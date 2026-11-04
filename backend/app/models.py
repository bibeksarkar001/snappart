"""Request and response schemas. Requests forbid unknown fields; responses are what docs/api.md documents."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobCreate(Strict):
    part_name: str | None = Field(None, max_length=80)


class StartRequest(Strict):
    build_anyway: bool = False                     # start although the capture check said the footage is not good enough


class AnswersRequest(Strict):
    answers: dict[Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]{1,40}$")], float | Annotated[str, StringConstraints(max_length=40)]] \
        = Field(..., max_length=20)                                  # question id -> mm, a choice, or "skip" (I don't know)


class CorrectionRequest(Strict):
    correction: dict                                                 # validated against capture/corrections.py SCHEMA


class ReviewChoice(Strict):
    choice: str = Field(..., max_length=40)


def _finite(v):
    import math
    if isinstance(v, (list, tuple)):
        return all(_finite(x) for x in v)
    return not isinstance(v, float) or math.isfinite(v)


class MarkRequest(Strict):
    point: list[float] = Field(..., min_length=3, max_length=3)     # part frame, mm (from the 3D viewer)
    radius_mm: float = Field(3.0, ge=0.5, le=20.0)
    depth_mm: float = Field(3.0, ge=0.2, le=20.0)                   # how far Remove cuts into the part (sideways spots)
    normal: list[float] | None = Field(None, min_length=3, max_length=3)
    note: str | None = Field(None, max_length=200)

    @model_validator(mode="after")
    def _sane(self):
        if not (_finite(self.point) and _finite(self.radius_mm) and _finite(self.depth_mm) and _finite(self.normal or [])):
            raise ValueError("point, normal, radius and depth must be finite numbers")
        if self.normal is not None and sum(x * x for x in self.normal) < 1e-9:
            raise ValueError("normal must not be the zero vector")
        return self


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
    recorded: dict | None = None                   # replay mode only: which recorded run is shown (never a live call)


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
    recorded: dict | None = None                   # replay mode only


class ProposalRequest(Strict):
    text: str = Field(..., min_length=1, max_length=2000)            # untrusted; cut to 1000 characters before any model sees it
    item_id: str | None = Field(None, pattern=r"^[A-Z][0-9]{2}$")


class MoreFootage(Strict):
    request_id: str = Field(..., pattern=r"^film_[0-9]{1,3}$")
    action: str = Field(..., pattern=r"^(film|skip)$")                 # film: upload more frames of the area; skip: accept it as not seen

