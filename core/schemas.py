from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_TEXT_LENGTH = 5000
MAX_BATCH_SIZE = 100

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH)]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"text": "Сегодня хорошая погода, пойдём гулять в парк."}]})

    text: Text


class BatchRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"texts": ["Свергнем власть силой!", "Погуляем по парку, весна!"]}]})

    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Verdict(BaseModel):
    probability: float
    is_extremist: bool


class AnalyzeResponse(Verdict):
    threshold: float


class SentenceVerdict(Verdict):
    sentence: str


class SentencesResponse(AnalyzeResponse):
    extremist_count: int
    sentences: list[SentenceVerdict]


class BatchResponse(BaseModel):
    threshold: float
    total: int
    extremist_count: int
    results: list[Verdict]


class ModelInfoResponse(BaseModel):
    data_path: str
    samples: int
    vocabulary_size: int
    max_len: int
    train_accuracy: float
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class VersionResponse(BaseModel):
    name: str
    version: str
