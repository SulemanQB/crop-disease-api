"""Request and response models for the inference API."""

from pydantic import BaseModel, ConfigDict, Field


class ClassScore(BaseModel):
    """One class and its softmax probability."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(
        ...,
        alias="class",
        description="PlantVillage-style class name (crop, condition).",
        examples=["Tomato___Early_blight"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability for this class.",
    )


class PredictResponse(BaseModel):
    """Top class plus a ranked shortlist."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(
        ...,
        alias="class",
        description="Highest-probability class.",
        examples=["Tomato___Early_blight"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability of the top class.",
    )
    topk: list[ClassScore] = Field(
        ...,
        description="Top classes, highest confidence first.",
    )


class HealthResponse(BaseModel):
    """Liveness payload. The process only serves this after the checkpoint loads."""

    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    arch: str = Field(..., examples=["resnet18"])
    num_classes: int = Field(..., examples=[38])
