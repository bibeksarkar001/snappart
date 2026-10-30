"""Request and response schemas. Requests forbid unknown fields; responses are what docs/api.md documents."""
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobCreate(Strict):
    part_name: str | None = Field(None, max_length=80)


class StartRequest(Strict):
    build_anyway: bool = False                     # start although the capture check said the footage is not good enough


class AnswersRequest(Strict):
    answers: dict[str, float | str] = Field(..., max_length=20)     # question id -> mm, a choice, or "skip" (I don't know)


class CorrectionRequest(Strict):
    correction: dict                                                 # validated against capture/corrections.py SCHEMA


class ReviewChoice(Strict):
    choice: str = Field(..., max_length=40)


class MarkRequest(Strict):
    point: list[float] = Field(..., min_length=3, max_length=3)     # part frame, mm (from the 3D viewer)
    radius_mm: float = Field(3.0, ge=0.5, le=20.0)
    normal: list[float] | None = Field(None, min_length=3, max_length=3)
    note: str | None = Field(None, max_length=200)


class ErrorBody(BaseModel):
    code: str
    message: str
    retry_after_s: int | None = None
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class JobError(BaseModel):
    code: str | None
    message: str | None


class Job(BaseModel):
    job_id: str
    state: str
    mode: str
    created_at: float
    updated_at: float
    state_changed_at: float
    warnings: list[str]
    error: JobError | None
    queue_position: int | None
    eta_s: float | None
    outputs: list[str]
    uploads: dict | None = None
    capture_check: dict | None = None


class JobCreated(BaseModel):
    job_id: str
    resume_code: str
    state: str
    mode: str
    limits: dict


class Event(BaseModel):
    seq: int
    type: str
    data: dict
    ts: float


class Events(BaseModel):
    job_id: str
    events: list[Event]
    last_seq: int


class ModelState(BaseModel):
    id: str
    running: bool
    reason: str
    calls_ok: int
    calls_failed: int
    last_ok_at: float | None


class MetaModels(BaseModel):
    mode: str
    job_id: str | None
    models: list[ModelState]
