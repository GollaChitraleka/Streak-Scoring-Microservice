from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional, List
import logging
from app.core.config_loader import AppConfig

logger = logging.getLogger(__name__)

class StreakManager:
    """Manages user streaks for different action types"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.storage = {}  # In-memory storage for development
        
    def _get_user_streaks(self, user_id: str) -> Dict[str, Any]:
        """Get streaks for a specific user, or initialize if not present"""
        if user_id not in self.storage:
            self.storage[user_id] = self._initialize_user_streaks()
        return self.storage[user_id]
        
    def _initialize_user_streaks(self) -> Dict[str, Any]:
        """Initialize streak data for a new user"""
        streaks = {}
        for action_type in self.config.action_types:
            if self.config.action_types[action_type].enabled:
                streaks[action_type] = {
                    "current_streak": 0,
                    "status": "inactive",
                    "tier": "none",
                    "last_action_date": None,
                    "next_deadline": None
                }
        return streaks
        
    def _calculate_tier(self, streak_count: int) -> str:
        """Calculate tier based on streak count"""
        tier = "none"
        for tier_name, tier_config in self.config.streak_settings.tiers.items():
            if streak_count >= tier_config.threshold:
                if tier_config.threshold > self.config.streak_settings.tiers[tier].threshold:
                    tier = tier_name
        return tier
        
    def _calculate_next_deadline(self, current_date: datetime) -> datetime:
        """Calculate next deadline for streak continuation"""
        # Next day at 23:59:59 UTC
        next_day = current_date.date() + timedelta(days=1)
        return datetime.combine(next_day, time(23, 59, 59))
        
    def update_streak(self, user_id: str, action_type: str, date: datetime, 
                       is_valid: bool, rejection_reason: Optional[str] = None) -> Dict[str, Any]:
        """Update streak for a specific action type"""
        # This is a placeholder for now - will be implemented in Week 4
        return {
            "current_streak": 1,
            "status": "active" if is_valid else "inactive",
            "tier": "none", 
            "validated": is_valid,
            "rejection_reason": rejection_reason if not is_valid else None,
            "next_deadline_utc": self._calculate_next_deadline(date) if is_valid else None
        }
        
    def process_actions(self, user_id: str, date: datetime, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a list of actions and update streaks"""
        # This is a placeholder for now - will be implemented in Week 4
        results = {}
        for action in actions:
            action_type = action["type"]
            # Assuming all actions are valid for now
            results[action_type] = self.update_streak(
                user_id, 
                action_type,
                date,
                is_valid=True
            )
        return results