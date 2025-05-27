import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_version():
    """Test the version endpoint"""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "config_version" in data
    assert "models" in data

def test_basic_login_streak():
    """Test a basic login streak update"""
    payload = {
        "user_id": "test_user_1",
        "date_utc": "2024-07-05T15:10:00Z",
        "actions": [
            {
                "type": "login",
                "metadata": {}
            }
        ]
    }
    
    response = client.post("/streak/update", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["user_id"] == "test_user_1"
    assert "login" in data["streaks"]
    assert data["streaks"]["login"]["current_streak"] == 1
    assert data["streaks"]["login"]["status"] == "active"
    assert data["streaks"]["login"]["tier"] == "none"
    assert "next_deadline_utc" in data["streaks"]["login"]