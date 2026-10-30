"""Request and response schemas. Requests forbid unknown fields; responses are what docs/api.md documents."""
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobCreate(Strict):
    part_name: str | None = Field(None, max_length=80)


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


class JobCreated(BaseModel):
    job_id: str
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
