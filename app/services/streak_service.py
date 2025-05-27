from typing import Dict, Any, List
from datetime import datetime
import logging
from app.core.config_loader import AppConfig
from app.core.streak_manager import StreakManager
from app.services.validators import ActionValidator

logger = logging.getLogger(__name__)

class StreakService:
    """Main service for processing streak updates"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.streak_manager = StreakManager(config)
        self.validator = ActionValidator(config)
        
    def process_streak_update(self, user_id: str, date_utc: datetime, 
                               actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a streak update request"""
        logger.info(f"Processing streak update for user: {user_id}")
        
        results = {}
        
        for action in actions:
            action_type = action["type"]
            metadata = action["metadata"]
            
            # Skip unsupported action types
            if action_type not in self.config.action_types:
                logger.warning(f"Skipping unsupported action type: {action_type}")
                continue
                
            # Skip disabled action types
            if not self.config.action_types[action_type].enabled:
                logger.info(f"Skipping disabled action type: {action_type}")
                continue
                
            # Validate the action
            is_valid, rejection_reason = self.validator.validate_action(action_type, metadata)
            
            # Update the streak based on validation result
            streak_result = self.streak_manager.update_streak(
                user_id,
                action_type,
                date_utc,
                is_valid,
                rejection_reason
            )
            
            results[action_type] = streak_result
            
        return {
            "user_id": user_id,
            "streaks": results
        }