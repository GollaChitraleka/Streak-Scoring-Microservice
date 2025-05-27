from pydantic import BaseModel, Field,RootModel
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

class LoginActionMetadata(BaseModel):
    pass  # No specific metadata required for login action

class QuizActionMetadata(BaseModel):
    quiz_id: str
    score: int = Field(..., ge=0)  # score must be >= 0
    time_taken_sec: int = Field(..., ge=0)  # time must be >= 0

class HelpPostActionMetadata(BaseModel):
    content: str
    word_count: int = Field(..., ge=0)
    contains_code: bool = False

class ActionMetadata(RootModel[Union[LoginActionMetadata, QuizActionMetadata, HelpPostActionMetadata]]):
    pass

class Action(BaseModel):
    type: str = Field(..., description="Type of action (login, quiz, help_post)")
    metadata: Dict[str, Any] = Field(..., description="Action-specific metadata")

class StreakUpdateRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    date_utc: datetime = Field(..., description="UTC timestamp of the action")
    actions: List[Action] = Field(..., description="List of actions to process")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "stu_2025",
                "date_utc": "2024-07-05T15:10:00Z",
                "actions": [
                    {
                        "type": "login",
                        "metadata": {}
                    },
                    {
                        "type": "quiz",
                        "metadata": {
                            "quiz_id": "quiz_8372",
                            "score": 7,
                            "time_taken_sec": 310
                        }
                    }
                ]
            }
        }
