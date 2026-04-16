from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from app.config import get_settings
from app.models import AggregatedScan, BBox, DefectInfo, FrameEvidence, FruitEvidence, ScanFruit


class ScanAggregator:
    """Aggregate multi-frame evidence into one deterministic ScanResult payload."""

    def __init__(self):
        self._settings = get_settings()

    def aggregate(
        self,
        frame_evidences: list[FrameEvidence],
        *,
        policy: str | None = None,
    ) -> AggregatedScan:
        if policy is None:
            policy = self._settings.AGGREGATION_POLICY
        if not frame_evidences:
            return AggregatedScan(representative_image_id="unknown", fruits=[], metadata={"frames": 0})

        representative_frame = self._select_representative_frame(frame_evidences)
        representative_image_id = representative_frame.image_id

        per_class_per_frame: dict[str, dict[str, list[FruitEvidence]]] = defaultdict(dict)
        for frame in frame_evidences:
            grouped: dict[str, list[FruitEvidence]] = defaultdict(list)
            for fruit in frame.fruits:
                grouped[fruit.fruit_class].append(fruit)
            for fruit_class in grouped:
                grouped[fruit_class].sort(key=lambda item: (item.bbox.x_min, item.bbox.y_min))
            per_class_per_frame[frame.image_id] = grouped

        all_classes = sorted(
            {
                fruit.fruit_class
                for frame in frame_evidences
                for fruit in frame.fruits
            }
        )

        aggregated_fruits: list[ScanFruit] = []
        for fruit_class in all_classes:
            counts = [
                len(per_class_per_frame[frame.image_id].get(fruit_class, []))
                for frame in frame_evidences
            ]
            target_count = self._majority_count(counts)
            if target_count <= 0:
                continue

            for index in range(target_count):
                candidates: list[FruitEvidence] = []
                for frame in frame_evidences:
                    class_fruits = per_class_per_frame[frame.image_id].get(fruit_class, [])
                    if index < len(class_fruits):
                        candidates.append(class_fruits[index])
                if not candidates:
                    continue

                aggregated_fruits.append(
                    self._aggregate_fruit_candidates(
                        candidates=candidates,
                        policy=policy,
                        fruit_id=f"{representative_image_id}-{fruit_class}-{index}",
                    )
                )

        aggregated_fruits.sort(key=lambda fruit: (fruit.fruit_class, fruit.bbox.x_min, fruit.bbox.y_min))
        return AggregatedScan(
            representative_image_id=representative_image_id,
            fruits=aggregated_fruits,
            metadata={
                "frames": len(frame_evidences),
                "policy": policy,
            },
        )

    def _select_representative_frame(self, frame_evidences: list[FrameEvidence]) -> FrameEvidence:
        ranked = sorted(
            frame_evidences,
            key=lambda frame: (
                sum(fruit.confidence for fruit in frame.fruits),
                len(frame.fruits),
                -frame.frame_id,
            ),
            reverse=True,
        )
        return ranked[0]

    def _majority_count(self, counts: list[int]) -> int:
        positive_counts = [count for count in counts if count > 0]
        if not positive_counts:
            return 0
        counter = Counter(positive_counts)
        return sorted(counter.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]

    def _aggregate_fruit_candidates(
        self,
        *,
        candidates: list[FruitEvidence],
        policy: str,
        fruit_id: str,
    ) -> ScanFruit:
        class_votes = Counter(item.fruit_class for item in candidates)
        fruit_class = sorted(class_votes.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]

        if policy == "best_frame_plus_vote":
            best = sorted(
                candidates,
                key=lambda item: (
                    item.confidence,
                    -item.bbox.x_min,
                    -item.bbox.y_min,
                ),
                reverse=True,
            )[0]
            confidence = best.confidence
            bbox = best.bbox
        else:
            confidence = float(mean(item.confidence for item in candidates))
            bbox = BBox(
                x_min=float(median(item.bbox.x_min for item in candidates)),
                y_min=float(median(item.bbox.y_min for item in candidates)),
                x_max=float(median(item.bbox.x_max for item in candidates)),
                y_max=float(median(item.bbox.y_max for item in candidates)),
            )

        defects = self._aggregate_defects(candidates=candidates, policy=policy)
        return ScanFruit(
            fruit_id=fruit_id,
            fruit_class=fruit_class,
            confidence=confidence,
            bbox=bbox,
            defects=defects,
        )

    def _aggregate_defects(self, *, candidates: list[FruitEvidence], policy: str) -> list[DefectInfo]:
        positive_frames = sum(1 for item in candidates if item.defects)
        if not positive_frames:
            return []

        total = len(candidates)
        if policy in {"vote", "best_frame_plus_vote"} and positive_frames * 2 <= total:
            return []
        if policy == "average" and (positive_frames / total) < 0.5:
            return []

        all_defects = [defect for item in candidates for defect in item.defects]
        best_defect = sorted(
            all_defects,
            key=lambda item: (item.confidence, item.type),
            reverse=True,
        )[0]
        avg_conf = float(mean(item.confidence for item in all_defects))

        return [
            DefectInfo(
                type=best_defect.type,
                confidence=avg_conf,
                # Keep segmentation from best evidence frame; we intentionally avoid polygon merge.
                segmentation=best_defect.segmentation,
            )
        ]
