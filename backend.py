from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# This defines what the AI expects to receive from your Kivy app
class Telemetry(BaseModel):
    device_type: str
    battery: float
    cpu_load: float
    memory: float

# --- NEW: ROOT ROUTE ---
# This fixes the {"detail":"Not Found"} error when you visit the main URL
@app.get("/")
async def home():
    return {
        "status": "AI System Online", 
        "message": "Neural Monitor Backend is live on Render",
        "version": "1.0.2"
    }

# --- PREDICTION ROUTE ---
@app.post("/predict")
async def predict(data: Telemetry):
    # AI Logic: Calculates failure risk based on real-time hardware stress
    # Formula: (CPU 60%) + (Memory 40%)
    risk_score = (data.cpu_load * 0.6) + (data.memory * 0.4)
    
    # Determine the status level
    if risk_score > 80:
        analysis = "CRITICAL: High Failure Probability"
    elif risk_score > 50:
        analysis = "WARNING: System Stress Detected"
    else:
        analysis = "NOMINAL: System Healthy"

    return {
        "risk": f"{min(risk_score, 99):.1f}%",
        "raw_metrics": {
            "CPU": f"{int(data.cpu_load)}%", 
            "RAM": f"{int(data.memory)}%", 
            "BATT": f"{int(data.battery)}%"
        },
        "analysis": analysis,
        "device": data.device_type
    }

if __name__ == "__main__":
    # This part allows you to run it locally for testing
    uvicorn.run(app, host="0.0.0.0", port=8000)