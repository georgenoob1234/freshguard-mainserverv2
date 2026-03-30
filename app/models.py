from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MachineState(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


class WeightEvent(BaseModel):
    grams: float = Field(ge=0)
    timestamp: datetime = Field(default_factory=utc_now)
    source_id: str | None = None
    seq: int | None = Field(default=None, ge=0)


class WeightIngressResponse(BaseModel):
    status: str = "accepted"
    state: MachineState
    session_id: str | None = None
    scan_id: str | None = None
    triggered_scan: bool = False
    reason: str


class StateTransition(BaseModel):
    from_state: MachineState
    to_state: MachineState


class StateMachineDecision(BaseModel):
    state: MachineState
    session_id: str | None = None
    scan_id: str | None = None
    triggered_scan: bool = False
    reason: str = "no_action"
    transition: StateTransition | None = None
    stable_weight: float | None = None
    close_session: bool = False


class ScanTriggerContext(BaseModel):
    session_id: str
    scan_id: str
    weight_grams: float
    trigger_reason: str
    operating_mode: Literal["scale", "shelf"] = "scale"
    triggered_at: datetime = Field(default_factory=utc_now)


class CameraCaptureImage(BaseModel):
    index: int = Field(ge=0)
    image_id: str
    image_url_or_path: str


class CameraCaptureResponse(BaseModel):
    image_id: str
    image_url_or_path: str
    timestamp: datetime
    images: list[CameraCaptureImage] | None = None


class FruitDetection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fruit_id: str
    fruit_class: str = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if len(bbox) != 4:
            raise ValueError("bbox must contain 4 coordinates")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox max coordinates must be larger than min coordinates")
        return bbox

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))


class FruitDetectionResponse(BaseModel):
    image_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: list[FruitDetection] = Field(
        default_factory=list,
        validation_alias=AliasChoices("detections", "fruits"),
    )

    @property
    def avg_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(mean(d.confidence for d in self.detections))


class Segmentation(BaseModel):
    polygon: list[tuple[float, float]]


class DefectInfo(BaseModel):
    type: str
    confidence: float = Field(ge=0, le=1)
    segmentation: Segmentation | None = None


class DefectDetectionResponse(BaseModel):
    image_id: str
    fruit_id: str
    defects: list[DefectInfo] = Field(default_factory=list)


class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def from_xyxy(cls, bbox: tuple[float, float, float, float]) -> "BBox":
        x1, y1, x2, y2 = bbox
        return cls(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


class ScanFruit(BaseModel):
    fruit_id: str
    fruit_class: str
    confidence: float = Field(ge=0, le=1)
    bbox: BBox
    defects: list[DefectInfo] = Field(default_factory=list)
    note: str | None = None


class ScanResult(BaseModel):
    session_id: str
    image_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    weight_grams: float = Field(ge=0)
    fruits: list[ScanFruit] = Field(default_factory=list)


class DroppedDetection(BaseModel):
    image_id: str
    fruit_id: str
    fruit_class: str
    confidence: float
    threshold: float
    reason: str


class FruitEvidence(BaseModel):
    source_fruit_id: str
    fruit_class: str
    confidence: float = Field(ge=0, le=1)
    bbox: BBox
    defects: list[DefectInfo] = Field(default_factory=list)
    note: str | None = None


class FrameEvidence(BaseModel):
    frame_id: int
    image_id: str
    captured_at: datetime = Field(default_factory=utc_now)
    fruits: list[FruitEvidence] = Field(default_factory=list)
    frame_error: str | None = None


class AggregatedScan(BaseModel):
    representative_image_id: str
    fruits: list[ScanFruit]
    metadata: dict[str, Any] = Field(default_factory=dict)
