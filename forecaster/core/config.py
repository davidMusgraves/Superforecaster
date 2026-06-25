"""Centralized configuration, loaded from environment / .env.

Trimmed for the forecaster: no exchange/trading fields (those arrive later if we
add a trading layer). Import the module-level ``settings`` singleton.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parents of this file: forecaster/core/config.py -> repo root.
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage / artifacts ---
    db_path: str = Field(default="data/forecaster.db")
    calibrator_path: str = Field(default="data/calibrators.json")
    model_path: str = Field(default="data/forecast_model.json")

    # --- news / evidence ---
    news_user_agent: str = Field(default="Superforecaster (research)")

    # --- LLM (outside-view priors + headline classification) ---
    anthropic_api_key: str = Field(default="")
    llm_model: str = Field(default="claude-haiku-4-5-20251001")

    # --- behavior ---
    use_calibration: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    def _abs(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT_DIR / path

    @property
    def db_abs_path(self) -> Path:
        return self._abs(self.db_path)

    @property
    def calibrator_abs_path(self) -> Path:
        return self._abs(self.calibrator_path)

    @property
    def model_abs_path(self) -> Path:
        return self._abs(self.model_path)


settings = Settings()
