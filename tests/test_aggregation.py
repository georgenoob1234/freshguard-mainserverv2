from __future__ import annotations

from app.core.aggregation import ScanAggregator
from app.models import BBox, DefectInfo, FrameEvidence, FruitEvidence


def _fruit(
    *,
    fruit_id: str,
    fruit_class: str = "apple",
    confidence: float = 0.9,
    defects: list[DefectInfo] | None = None,
) -> FruitEvidence:
    return FruitEvidence(
        source_fruit_id=fruit_id,
        fruit_class=fruit_class,
        confidence=confidence,
        bbox=BBox(x_min=10, y_min=20, x_max=110, y_max=220),
        defects=defects or [],
    )


def test_aggregation_vote_policy_uses_majority_for_defect_presence() -> None:
    aggregator = ScanAggregator()
    defect = DefectInfo(type="defect", confidence=0.8)

    frames = [
        FrameEvidence(frame_id=0, image_id="img-0", fruits=[_fruit(fruit_id="f0")]),
        FrameEvidence(
            frame_id=1,
            image_id="img-1",
            fruits=[_fruit(fruit_id="f1", confidence=0.8, defects=[defect])],
        ),
        FrameEvidence(
            frame_id=2,
            image_id="img-2",
            fruits=[_fruit(fruit_id="f2", confidence=0.7, defects=[defect])],
        ),
    ]

    result = aggregator.aggregate(frames, policy="vote")
    assert result.representative_image_id == "img-0"
    assert len(result.fruits) == 1
    assert len(result.fruits[0].defects) == 1
    assert result.fruits[0].defects[0].type == "defect"


def test_aggregation_handles_missing_frame_data() -> None:
    aggregator = ScanAggregator()
    frames = [
        FrameEvidence(
            frame_id=0,
            image_id="img-0",
            fruits=[_fruit(fruit_id="f0", confidence=0.85)],
        ),
        FrameEvidence(
            frame_id=1,
            image_id="img-1",
            fruits=[],
            frame_error="fruit detector timeout",
        ),
    ]

    result = aggregator.aggregate(frames, policy="vote")
    assert result.representative_image_id == "img-0"
    assert len(result.fruits) == 1
    assert result.fruits[0].fruit_class == "apple"


def test_representative_image_selection_is_deterministic() -> None:
    aggregator = ScanAggregator()
    frames = [
        FrameEvidence(frame_id=1, image_id="img-b", fruits=[_fruit(fruit_id="fb", confidence=0.9)]),
        FrameEvidence(frame_id=0, image_id="img-a", fruits=[_fruit(fruit_id="fa", confidence=0.9)]),
    ]

    result = aggregator.aggregate(frames, policy="vote")
    assert result.representative_image_id == "img-a"
