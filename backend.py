import os, uvicorn, json
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.ensemble import IsolationForest

app = FastAPI()

# 1. THE ML MODEL
# We initialize it with some "normal" baseline data
# In a real project, you'd train this on thousands of rows of sensor data.
model = IsolationForest(contamination=0.1) # 10% expected anomaly rate
baseline_data = np.random.normal(0.1, 0.05, (100, 1)) # Simulated "Healthy" vibration
model.fit(baseline_data)

last_data = {"score": 0, "risk": "WAITING"}
history = []

@app.post("/predict")
async def predict(data: dict):
    global last_data
    
    vib = float(data.get("vibration", 0.0))
    
    # 2. ML INFERENCE
    # Convert single vibration point into a format the model understands
    current_val = np.array([[vib]])
    
    # decision_function returns a score: higher is more "normal"
    raw_ml_score = model.decision_function(current_val)[0]
    
    # Convert ML score to a 0-100% Health Scale
    health_score = int(max(0, min(100, (raw_ml_score + 0.5) * 100)))
    
    if health_score < 40:
        status, risk = "AI ALERT: ANOMALY DETECTED", "CRITICAL"
        analysis = f"Vibration ({vib}G) deviates significantly from learned baseline."
    else:
        status, risk = "AI STATUS: NORMAL", "NOMINAL"
        analysis = "Vibration pattern matches healthy operating signatures."

    last_data = {
        "vibration": vib,
        "score": health_score,
        "risk": risk,
        "status": status,
        "analysis": analysis,
        "device": data.get("device_type", "Mobile Probe")
    }
    history.append(health_score)
    return last_data

# ... (Keep the rest of your /monitor and /data routes the same)