from fastapi import FastAPI

app = FastAPI(
    title="Streak Scoring Microservice",
    description="Track and validate user engagement streaks across different action types",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/version")
async def version():
    """Version information endpoint"""
    return {
        "version": "1.0.0",
        "config_version": "1.0.0",
        "models": {}
    }
