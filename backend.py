import os, uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.ensemble import IsolationForest

app = FastAPI()

# --- THE AIML ENGINE ---
# We use Isolation Forest to detect 'Anomalous' vibration
model = IsolationForest(contamination=0.1) 
# Initial baseline training (Simulating 'Healthy' states)
baseline = np.random.normal(0.1, 0.05, (100, 1))
model.fit(baseline)

last_data = {"score": 100, "vibration": 0, "risk": "NOMINAL", "analysis": "System Initializing..."}

@app.get("/") # This fixes the "Not Found" error
async def home():
    return {"status": "AI Engine Online", "visual_dashboard": "/monitor"}

@app.post("/predict")
async def predict(data: dict):
    global last_data
    vib = float(data.get("vibration", 0.0))
    
    # AI INFERENCE
    # isolation_forest.decision_function gives a score: higher is more 'normal'
    score_raw = model.decision_function([[vib]])[0]
    health = int(max(0, min(100, (score_raw + 0.5) * 100)))
    
    risk = "CRITICAL" if health < 40 else "WARNING" if health < 75 else "NOMINAL"
    analysis = "AI detected anomalous vibration pattern!" if health < 40 else "System operating normally."

    last_data = {
        "vibration": round(vib, 3), "score": health, "risk": risk,
        "analysis": analysis, "device": data.get("device_type", "Handheld Probe")
    }
    return last_data

@app.get("/data")
async def get_data(): return last_data

@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return """
    <html>
    <head><title>AI MONITOR</title><style>body{background:#000;color:#0f0;text-align:center;font-family:monospace;padding:50px;}</style></head>
    <body>
        <h1>NEURAL ASSET MONITOR</h1>
        <div id='box' style='border:2px solid; padding:20px; display:inline-block;'>
            <h2 id='stat'>Loading AI Data...</h2>
            <p id='score' style='font-size:40px'></p>
            <p id='anal' style='color:#888'></p>
        </div>
        <script>
            async function upd(){
                const r = await fetch('/data'); const d = await r.json();
                document.getElementById('stat').innerText = d.risk + " (" + d.vibration + "G)";
                document.getElementById('score').innerText = d.score + "% HEALTH";
                document.getElementById('anal').innerText = d.analysis;
                document.getElementById('box').style.borderColor = d.risk == 'CRITICAL' ? 'red' : 'green';
            }
            setInterval(upd, 2000);
        </script>
    </body></html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)