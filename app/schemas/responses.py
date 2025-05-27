from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime

class StreakStatus(BaseModel):
    current_streak: int = Field(..., description="Current streak count")
    status: str = Field(..., description="Status of the streak (active, inactive)")
    tier: str = Field(..., description="Current tier (none, bronze, silver, gold)")
    next_deadline_utc: Optional[datetime] = Field(None, description="Deadline for next action to maintain streak")
    validated: Optional[bool] = Field(None, description="Whether the action was validated (for content-based actions)")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection if validation failed")

class StreakUpdateResponse(BaseModel):
    user_id: str = Field(..., description="User ID from the request")
    streaks: Dict[str, StreakStatus] = Field(..., description="Status of streaks for each action type")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "stu_2025",
                "streaks": {
                    "login": {
                        "current_streak": 4,
                        "status": "active",
                        "tier": "bronze",
                        "next_deadline_utc": "2024-07-06T23:59:59Z"
                    },
                    "quiz": {
                        "current_streak": 2,
                        "status": "active", 
                        "tier": "none",
                        "validated": True,
                        "next_deadline_utc": "2024-07-06T23:59:59Z"
                    }
                }
            }
        }
