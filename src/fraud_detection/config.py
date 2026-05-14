"""Application settings via Pydantic. Override with env vars or .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class DataSettings(BaseSettings):
    """Data location and acquisition settings."""

    raw_dir: Path = Field(default=REPO_ROOT / "data" / "raw")
    processed_dir: Path = Field(default=REPO_ROOT / "data" / "processed")
    paysim_filename: str = "PS_20174392719_1491204439457_log.csv"
    paysim_kaggle_dataset: str = "ealaxi/paysim1"

    @property
    def paysim_path(self) -> Path:
        return self.raw_dir / self.paysim_filename


class ModelSettings(BaseSettings):
    """Model training and registry settings."""

    models_dir: Path = Field(default=REPO_ROOT / "models")
    random_seed: int = 42
    n_jobs: int = -1


class CostSettings(BaseSettings):
    """Cost-sensitive evaluation parameters.

    Defaults assume ~$500 loss per missed fraud and ~$10 per investigator
    review. These are placeholders; override per project.
    """

    model_config = SettingsConfigDict(env_prefix="COST_")

    false_negative: float = 500.0
    false_positive: float = 10.0


class MLflowSettings(BaseSettings):
    """MLflow tracking server connection."""

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fraud-detection"


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: str = "INFO"

    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)


settings = Settings()
