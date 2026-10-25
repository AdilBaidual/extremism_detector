import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
APP_NAME = "Extremism Detector"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    app_version: str = "1.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model_dir: Path = BASE_DIR / "model"
    training_data_path: Path = BASE_DIR / "data" / "training_data_mock.csv"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("model_dir", "training_data_path")
    @classmethod
    def resolve_from_project_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else BASE_DIR / value


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT)
