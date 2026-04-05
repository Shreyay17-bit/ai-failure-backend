import os, uvicorn, time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.ensemble import IsolationForest

app = FastAPI()

# AI Engine Configuration
model = IsolationForest(contamination=0.05, random_state=42)
# Training with 'Healthy' baseline (0.08G avg with slight noise)
baseline = np.random.normal(0.08, 0.02, (200, 1))
model.fit(baseline)

# Global State for Telemetry
history = []
last_data = {
    "score": 100, "vibration": 0, "risk": "NOMINAL", 
    "analysis": "Neural Engine Online", "peak_g": 0, "z_score": 0, "latency": 0
}

@app.get("/")
async def home(): return {"status": "AI Engine Online", "dashboard": "/monitor"}

@app.post("/predict")
async def predict(data: dict):
    global last_data, history
    start_time = time.time()
    
    vib = float(data.get("vibration", 0.0))
    history.append(vib)
    if len(history) > 50: history.pop(0)

    # --- ADVANCED PARAMETER CALCULATION ---
    # 1. AI Decision Function (The 'Neural' Score)
    score_raw = model.decision_function([[vib]])[0]
    health = int(max(0, min(100, (score_raw + 0.5) * 100)))
    
    # 2. Z-Score (Statistical Deviation)
    mean_h = np.mean(history)
    std_h = np.std(history) if len(history) > 1 else 1
    z_score = abs((vib - mean_h) / std_h) if std_h > 0 else 0
    
    # 3. Peak-to-Peak (Stress Testing)
    peak_g = max(history)

    # 4. Processing Latency
    latency = round((time.time() - start_time) * 1000, 2)

    # Logic for Diagnostic Message
    if health < 45 or z_score > 3:
        risk, msg = "CRITICAL", f"High Sigma Deviation ({round(z_score, 2)}σ). Structural integrity compromised."
    elif health < 75:
        risk, msg = "WARNING", "Sub-threshold anomalies detected. Inspect mechanical bearings."
    else:
        risk, msg = "NOMINAL", "Pattern matches learned 'Healthy' dataset. Zero degradation."

    last_data = {
        "vibration": round(vib, 3), "score": health, "risk": risk,
        "analysis": msg, "peak_g": round(peak_g, 3), 
        "z_score": round(z_score, 2), "latency": f"{latency}ms"
    }
    return last_data

@app.get("/data")
async def get_data(): return last_data

@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ADVANCED NEURAL DIAGNOSTICS</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { background: #05080a; color: #00e5ff; font-family: 'Segoe UI', monospace; margin: 0; padding: 20px; }
            .dashboard { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; max-width: 1400px; margin: auto; }
            .card { background: #0d1117; border: 1px solid #1f2937; padding: 20px; border-radius: 10px; text-align: center; }
            .large-card { grid-column: span 2; }
            .val { font-size: 40px; font-weight: bold; margin: 10px 0; color: #fff; }
            .label { color: #8b949e; text-transform: uppercase; font-size: 12px; letter-spacing: 1.5px; }
            .CRITICAL { color: #ff4444 !important; border: 1px solid #ff4444; }
            .NOMINAL { color: #00ffcc !important; }
            #analysis { font-size: 14px; color: #8b949e; margin-top: 15px; border-top: 1px solid #222; padding-top: 10px; }
        </style>
    </head>
    <body>
        <h1 style="text-align:center; font-weight: 300; letter-spacing: 10px;">X-NEURAL ANALYTICS v4.0</h1>
        <div class="dashboard">
            <div class="card" id="main_border">
                <div class="label">System Health Index</div>
                <div id="health" class="val">--%</div>
                <div id="risk" style="font-weight:bold;">CONNECTING...</div>
            </div>
            <div class="card">
                <div class="label">Statistical Deviation (Z-Score)</div>
                <div id="zscore" class="val">0.0</div>
                <div class="label">Sigma Unit</div>
            </div>
            <div class="card">
                <div class="label">Peak Stress (Max G)</div>
                <div id="peak" class="val">0.00</div>
                <div id="latency" style="color:#555">0ms latency</div>
            </div>
            <div class="card large-card">
                <div class="label">Neural Pattern Tracking</div>
                <canvas id="chart" height="100"></canvas>
            </div>
            <div class="card">
                <div class="label">AI Logic Core Output</div>
                <p id="analysis">Initializing predictive tensors...</p>
            </div>
        </div>
        <script>
            const ctx = document.getElementById('chart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: { labels: Array(40).fill(''), datasets: [{ 
                    label: 'Vibration Impulse', data: [], borderColor: '#00e5ff', 
                    borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(0, 229, 255, 0.05)', tension: 0.3 
                }]},
                options: { animation: false, scales: { y: { beginAtZero: true, grid: { color: '#161b22' } }, x: { display: false } } }
            });

            async function update() {
                try {
                    const r = await fetch('/data');
                    const d = await r.json();
                    document.getElementById('health').innerText = d.score + "%";
                    document.getElementById('risk').innerText = d.risk;
                    document.getElementById('zscore').innerText = d.z_score;
                    document.getElementById('peak').innerText = d.peak_g;
                    document.getElementById('latency').innerText = d.latency + " inference";
                    document.getElementById('analysis').innerText = d.analysis;
                    document.getElementById('main_border').className = "card " + (d.risk === 'CRITICAL' ? 'CRITICAL' : '');
                    
                    chart.data.datasets[0].data.push(d.vibration);
                    if(chart.data.datasets[0].data.length > 40) chart.data.datasets[0].data.shift();
                    chart.update();
                } catch(e) {}
            }
            setInterval(update, 1000);
        </script>
    </body>
    </html>
    """