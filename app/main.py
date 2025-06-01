from fastapi import FastAPI, HTTPException, Depends
from app.core.config_loader import load_config, AppConfig
from app.schemas.requests import StreakUpdateRequest
from app.schemas.responses import StreakUpdateResponse
from app.services.streak_service import StreakService
import logging
from pathlib import Path

from app.services.validators import ActionValidator

config_path = Path(__file__).parent / "config" / "config.json"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize application
app = FastAPI(
    title="Streak Scoring Microservice",
    description="Track and validate user engagement streaks across different action types",
    version="1.0.0"
)

# Global config object
config = None

def get_config() -> AppConfig:
    """Dependency to get config"""
    if not config:
        raise HTTPException(status_code=500, detail="Application not properly initialized")
    return config

def get_streak_service(config: AppConfig = Depends(get_config)) -> StreakService:
    """Dependency to get streak service"""
    return StreakService(config)

@app.on_event("startup")
async def startup_event():
    """Load configuration on startup"""
    global config
    try:
        config = load_config(str(config_path))
        logger.info(f"Configuration loaded successfully. Version: {config.version}")

        validator = ActionValidator(config)
        logger.info("Validator initialized during startup.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/version")
async def version(config: AppConfig = Depends(get_config)):
    """Version information endpoint"""
    return {
        "version": config.version,
        "config_version": config.version,
        "models": {
            model_name: {"threshold": model_config.threshold}
            for model_name, model_config in config.ai_models.items()
        }
    }

@app.post("/streak/update", response_model=StreakUpdateResponse)
async def update_streak(
    request: StreakUpdateRequest,
    streak_service: StreakService = Depends(get_streak_service)
):
    """Process a streak update request"""
    try:
        result = streak_service.process_streak_update(
            request.user_id,
            request.date_utc,
            [action.dict() for action in request.actions]
        )
        return result
    except Exception as e:
        logger.error(f"Error processing streak update: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process streak update: {str(e)}")
