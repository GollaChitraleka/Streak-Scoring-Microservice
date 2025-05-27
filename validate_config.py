# validate_config.py

from app.core.config_loader import load_config
import pprint

if __name__ == "__main__":
    try:
        config = load_config("config/config.json")
        pprint.pprint(config.dict())  # Pretty-print the validated config
        print("✅ Config is valid.")
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
