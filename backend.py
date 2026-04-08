import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# Global storage for telemetry
latest_data = {
    "status": "NOMINAL",
    "vibration": 0.0,
    "battery": 86,
    "cpu": 9,
    "memory": 78.6,
    "score": 0
}

# 1. RECEIVER (For Phone or PC data)
@app.post("/predict")
async def predict(data: dict):
    global latest_data
    
    # Extract values and force them into numbers to avoid math errors
    v = float(data.get("vibration", 0.0))
    b = int(data.get("battery", 0))
    c = int(data.get("cpu", 0))
    m = float(data.get("memory", 0.0))
    
    # 2. CALCULATION: Risk Score (Logic verified for VIT Presentation)
    # Equation: High Vibration + High CPU + Low Battery = High Risk
    risk_score = int((v * 40) + (c * 0.4) + ((100 - b) * 0.2))
    risk_score = min(max(risk_score, 0), 100) # Clamp between 0-100
    
    status = "NOMINAL"
    if risk_score > 70: 
        status = "CRITICAL FAILURE"
    elif risk_score > 40: 
        status = "WARNING"

    latest_data = {
        "status": status,
        "vibration": round(v, 2),
        "battery": b,
        "cpu": c,
        "memory": round(m, 1),
        "score": risk_score
    }
    return latest_data

# 2. INTERNAL API (Used by UI to get updates)
@app.get("/get_latest")
async def get_latest():
    return latest_data

# 3. YOUR ORIGINAL UI (Verified Logic)
@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NEURAL MONITOR | LIVE</title>
        <style>
            :root { --neon: #00ffcc; --bg: #050505; }
            body { background: var(--bg); color: var(--neon); font-family: 'Courier New', monospace; margin: 0; padding: 20px; text-align: center; }
            h1 { letter-spacing: 15px; border-bottom: 1px solid var(--neon); padding-bottom: 10px; margin-bottom: 30px; }
            .status-container { border: 2px solid var(--neon); padding: 20px; margin-bottom: 30px; border-radius: 8px; }
            .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }
            .card { background: #0a0a0a; border: 1px solid #333; padding: 15px; border-radius: 5px; text-align: left; }
            .val { font-size: 32px; display: block; margin: 10px 0; color: #fff; }
            .progress-bg { background: #222; height: 8px; width: 100%; border-radius: 4px; }
            .progress-fill { background: var(--neon); height: 100%; width: 0%; transition: 0.5s; }
            .CRITICAL { color: #ff3333; text-shadow: 0 0 10px #ff3333; }
        </style>
        <script>
            async function updateStats() {
                try {
                    const r = await fetch('/get_latest');
                    const d = await r.json();
                    
                    // Match these IDs exactly to the HTML elements below
                    document.getElementById('status').innerText = d.status;
                    document.getElementById('v-val').innerText = d.vibration;
                    document.getElementById('v-bar').style.width = (d.vibration * 100) + "%";
                    
                    document.getElementById('b-val').innerText = d.battery + "%";
                    document.getElementById('b-bar').style.width = d.battery + "%";
                    
                    document.getElementById('c-val').innerText = d.cpu + "%";
                    document.getElementById('c-bar').style.width = d.cpu + "%";
                    
                    document.getElementById('m-val').innerText = d.memory + "%";
                    document.getElementById('m-bar').style.width = d.memory + "%";
                } catch(e) { console.log("Reconnecting..."); }
            }
            setInterval(updateStats, 1000);
        </script>
    </head>
    <body>
        <h1>NEURAL MONITOR</h1>
        <div class="status-container"><div id="status">INITIALIZING...</div></div>
        <div class="grid">
            <div class="card">VIBRATION<span class="val" id="v-val">0.0</span><div class="progress-bg"><div id="v-bar" class="progress-fill"></div></div></div>
            <div class="card">BATTERY %<span class="val" id="b-val">0%</span><div class="progress-bg"><div id="b-bar" class="progress-fill"></div></div></div>
            <div class="card">CPU LOAD<span class="val" id="c-val">0%</span><div class="progress-bg"><div id="c-bar" class="progress-fill"></div></div></div>
            <div class="card">MEMORY<span class="val" id="m-val">0.0%</span><div class="progress-bg"><div id="m-bar" class="progress-fill"></div></div></div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))