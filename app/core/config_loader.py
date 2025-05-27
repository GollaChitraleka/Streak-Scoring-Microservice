import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Tier:
    name: str
    threshold: int
    description: str

@dataclass
class StreakConfig:
    tiers: Dict[str, Tier]
    grace_period_days: int
    reset_after_days: int

@dataclass
class ValidationConfig:
    require_ai: bool
    threshold: Optional[Dict[str, Any]]

@dataclass
class ActionTypeConfig:
    enabled: bool
    validation: ValidationConfig

@dataclass
@dataclass
class AIModelConfig:
    model_file: str
    vectorizer_file: str
    scaler_file: Optional[str] = None  # add this with default None
    threshold: float = 0.6


@dataclass
class PersistenceConfig:
    enabled: bool
    type: str
    file_path: str

@dataclass
class AppConfig:
    version: str
    streak: StreakConfig
    action_types: Dict[str, ActionTypeConfig]
    ai_models: Dict[str, AIModelConfig]
    persistence: PersistenceConfig
    

def load_config(file_path: str) -> AppConfig:
    path = Path(__file__).resolve().parent.parent / file_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(path, "r") as f:
        data = json.load(f)

    tiers = {k: Tier(**v) for k, v in data["streak"]["tiers"].items()}
    streak = StreakConfig(
        tiers=tiers,
        grace_period_days=data["streak"]["grace_period_days"],
        reset_after_days=data["streak"]["reset_after_days"]
    )

    action_types = {
        k: ActionTypeConfig(
            enabled=v["enabled"],
            validation=ValidationConfig(**v["validation"])
        ) for k, v in data["action_types"].items()
    }

    ai_models = {
        k: AIModelConfig(**v) for k, v in data["ai_models"].items()
    }

    persistence = PersistenceConfig(**data["persistence"])

    return AppConfig(
        version=data["version"],
        streak=streak,
        action_types=action_types,
        ai_models=ai_models,
        persistence=persistence
    )
