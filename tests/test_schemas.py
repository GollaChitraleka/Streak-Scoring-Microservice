import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.schemas.requests import StreakUpdateRequest

# Valid test input
data = {
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

try:
    request_obj = StreakUpdateRequest(**data)
    print("✅ Validation Passed!")
    print(request_obj.json(indent=2))
except Exception as e:
    print("❌ Validation Failed:")
    print(str(e))
