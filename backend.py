from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

class Telemetry(BaseModel):
    device_type: str
    battery: float
    cpu_load: float
    memory: float
    vibration: float = 0.0

@app.post("/predict")
async def predict(data: Telemetry):
    # Predictive Maintenance Logic
    risk_score = (data.cpu_load * 0.4) + (data.memory * 0.3) + (data.vibration * 0.3)
    analysis = "NOMINAL"
    
    if data.vibration > 8.0 or data.cpu_load > 90:
        analysis = "CRITICAL: FAILURE PREDICTED"
        risk_score = max(risk_score, 90)
    elif risk_score > 50:
        analysis = "WARNING: DEGRADED"

    return {
        "risk": f"{min(risk_score, 99.9):.1f}%",
        "analysis": analysis,
        "device": data.device_type
    }

@app.get("/monitor", response_class=HTMLResponse)
async def monitor_ui():
    # Use the HTML code provided in the previous message here
    return "<html>...</html>" 

import os

if __name__ == "__main__":
    # Render provides the port via an environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)