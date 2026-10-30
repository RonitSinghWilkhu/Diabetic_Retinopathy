from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    message: str

class PredictionResponse(BaseModel):
    analysis_id: int | None = None
    predicted_class: str
    predicted_class_index: int = Field(ge=0, le=4)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_name: str = "convnext_tiny"

class ImageContext(BaseModel):
    predicted_class: str
    predicted_class_index: int = Field(ge=0, le=4)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_name: str = "convnext_tiny"
    explanation: dict | None = None

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    image_context: ImageContext | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    disclaimer: str = (
        "This information is for educational purposes only "
    )