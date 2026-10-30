"""All settings and limits, from the environment (prefix SNAPPART_, e.g. SNAPPART_MOCK=1). The Meta API key is read from MODEL_API_KEY as a
SecretStr: it never appears in repr, logs or responses. Owner decisions (host, retention, contact) have safe defaults here and are listed in
docs/ops.md."""
import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SNAPPART_", extra="ignore")

    environment: str = Field("development", pattern="^(development|test|production)$")
    mock: bool = False                                   # SNAPPART_MOCK=1: canned build, same API, exposed in every response header
    allow_mock_in_production: bool = False               # mock mode in production refuses to start unless this is set explicitly
    data_dir: Path = BACKEND / "storage"                 # DATA_DIR: jobs/{id}/ files + the SQLite database (git-ignored)
    db_name: str = "snappart.db"
    model_api_key: SecretStr | None = Field(None, validation_alias=AliasChoices("MODEL_API_KEY", "SNAPPART_MODEL_API_KEY"))
    meta_base_url: str = "https://api.meta.ai/v1"
    meta_models: tuple[str, ...] = ("sam-3.1", "muse-spark-1.3", "muse-voice-transcribe-1.0")
    model_running_window_s: int = 24 * 3600              # global "running" = a successful call within this window

    # upload limits
    videos_per_job: int = 3
    max_frames_per_video: int = 60
    max_frame_bytes: int = 3 * 1024 * 1024
    max_job_bytes: int = 250 * 1024 * 1024
    max_json_bytes: int = 64 * 1024
    max_audio_bytes: int = 32 * 1024 * 1024
    max_frame_long_edge_px: int = 1920

    # queue and builds
    max_queue_len: int = 500
    max_concurrent_builds: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 3) // 3))
    build_timeout_s: int = 12 * 60
    mock_build_s: float = 5.0                            # mock build duration (progress events spread over it)

    # retention (owner decision; default 7 days)
    retention_days: int = 7

    cors_origins: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _no_silent_mock_in_production(self):
        if self.mock and self.environment == "production" and not self.allow_mock_in_production:
            raise ValueError("SNAPPART_MOCK=1 in production needs SNAPPART_ALLOW_MOCK_IN_PRODUCTION=1 (mock mode is never enabled silently)")
        return self

    @property
    def mode(self) -> str:
        return "mock" if self.mock else "real"

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_name

    def public_limits(self) -> dict:
        return {"videos_per_job": self.videos_per_job, "max_frames_per_video": self.max_frames_per_video, "max_frame_bytes": self.max_frame_bytes,
                "max_job_bytes": self.max_job_bytes, "max_frame_long_edge_px": self.max_frame_long_edge_px, "max_audio_bytes": self.max_audio_bytes}


@lru_cache
def get_settings() -> Settings:
    return Settings()
