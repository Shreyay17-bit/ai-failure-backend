from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Telemetry(BaseModel):
    device_type: str
    battery: float
    cpu_load: float
    memory: float

@app.post("/predict")
async def predict(data: Telemetry):

    ram_weight = 0.4 if data.device_type != 'win' else 0.2
    
    risk = (data.cpu_load * 0.5) + (data.memory * ram_weight)

    if data.battery < 20:
        risk += 25

    risk = min(risk, 100)

    # smarter insights
    if risk > 80:
        insight = "CRITICAL: System instability imminent!"
    elif risk > 60:
        insight = "Warning: Performance degrading."
    elif data.cpu_load > 85:
        insight = "High CPU usage detected."
    else:
        insight = "All systems operating normally."

    return {
        "risk": f"{risk:.1f}%",
        "analysis": insight,
        "raw_metrics": {
            "CPU": f"{data.cpu_load}%",
            "RAM": f"{data.memory}%",
            "BATT": f"{data.battery}%"
        }
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)