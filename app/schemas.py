from pydantic import BaseModel, Field
from typing import Optional, List


class PredictRequest(BaseModel):
    text: str = Field(..., description="News article text", min_length=20)
    language: Optional[str] = Field(
        None,
        description="ISO 639-1 code: hi, mr, gu, te. Auto-detected if omitted.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "नरेंद्र मोदी ने आज नई दिल्ली में एक महत्वपूर्ण बैठक की अध्यक्षता की।",
                    "language": "hi",
                }
            ]
        }
    }


class EntityEvidence(BaseModel):
    entity: str
    entity_type: str
    wikidata_verified: bool
    wikidata_description: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: str           # "FAKE" | "REAL"
    confidence: float         # 0.0 – 1.0
    fake_score: float
    real_score: float
    language_detected: str
    language_name: str
    entities_found: List[EntityEvidence]
    verified_count: int
    unverified_count: int
    explanation: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    supported_languages: List[str]
