import time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = FastAPI()

# ---------------- AI MODELS ----------------
model = IsolationForest(contamination=0.05, random_state=42)
scaler = StandardScaler()
trend_model = LinearRegression()

baseline = np.random.normal(0.08, 0.02, (200, 1))
model.fit(baseline)
scaler.fit(baseline)

# ---------------- GLOBAL DATA ----------------
history = []

last_data = {
    "score": 100,
    "vibration": 0,
    "risk": "NOMINAL",
    "analysis": "System Ready",
    "peak_g": 0,
    "z_score": 0,
    "latency": "0ms",
    "velocity": 0,
    "acceleration": 0,
    "trend": "STABLE",
    "frequency": 0
}

# ---------------- ROUTES ----------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return await monitor()


@app.post("/predict")
async def predict(data: dict):
    global last_data, history

    start_time = time.time()
    vib = float(data.get("vibration", 0.0))

    history.append(vib)
    if len(history) > 50:
        history.pop(0)

    # ---------- AI HEALTH ----------
    score_raw = model.decision_function([[vib]])[0]
    health = int(max(0, min(100, (score_raw + 0.5) * 100)))

    # ---------- Z SCORE ----------
    mean_h = np.mean(history)
    std_h = np.std(history) if len(history) > 1 else 1
    z_score = abs((vib - mean_h) / std_h) if std_h > 0 else 0

    # ---------- PEAK ----------
    peak_g = max(history)

    # ---------- LATENCY ----------
    latency = round((time.time() - start_time) * 1000, 2)

    # ---------- VELOCITY & ACCELERATION ----------
    velocity = 0
    acceleration = 0

    if len(history) > 1:
        velocity = history[-1] - history[-2]

    if len(history) > 2:
        acceleration = history[-1] - 2*history[-2] + history[-3]

    # ---------- TREND ----------
    trend = "STABLE"
    if len(history) >= 5:
        x = np.arange(len(history)).reshape(-1, 1)
        y = np.array(history)
        trend_model.fit(x, y)
        slope = trend_model.coef_[0]

        if slope > 0.002:
            trend = "INCREASING"
        elif slope < -0.002:
            trend = "DECREASING"

    # ---------- FREQUENCY ----------
    dominant_freq = 0
    if len(history) >= 10:
        fft_vals = np.fft.fft(history)
        freqs = np.fft.fftfreq(len(history))
        dominant_freq = abs(freqs[np.argmax(np.abs(fft_vals))])

    # ---------- RISK ----------
    if health < 45 or z_score > 3:
        risk = "CRITICAL"
        msg = f"High deviation ({round(z_score,2)}σ). Failure risk detected."
    elif health < 75:
        risk = "WARNING"
        msg = "Anomalies detected. Maintenance suggested."
    else:
        risk = "NOMINAL"
        msg = "System operating normally."

    last_data = {
        "vibration": round(vib, 3),
        "score": health,
        "risk": risk,
        "analysis": msg,
        "peak_g": round(peak_g, 3),
        "z_score": round(z_score, 2),
        "latency": f"{latency}ms",
        "velocity": round(velocity, 4),
        "acceleration": round(acceleration, 4),
        "trend": trend,
        "frequency": round(dominant_freq, 4)
    }

    return last_data


@app.get("/data")
async def get_data():
    return last_data


# ---------------- DASHBOARD ----------------

@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return """
<!DOCTYPE html>
<html>
<head>
<title>AI Diagnostics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body { background:#0a0f1c; color:white; font-family:Arial; text-align:center; }
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:15px; padding:20px; }
.card { background:#111; padding:20px; border-radius:10px; }
.CRITICAL { color:red; }
.WARNING { color:orange; }
.NOMINAL { color:lightgreen; }
</style>
</head>

<body>

<h1>AI SYSTEM MONITOR</h1>

<div class="grid">
<div class="card"><h2 id="health">--%</h2><p>Health</p></div>
<div class="card"><h2 id="vib">0</h2><p>Vibration</p></div>
<div class="card"><h2 id="z">0</h2><p>Z Score</p></div>
<div class="card"><h2 id="trend">--</h2><p>Trend</p></div>
<div class="card"><h2 id="freq">0</h2><p>Frequency</p></div>
<div class="card"><h2 id="risk">--</h2><p>Status</p></div>
</div>

<canvas id="chart"></canvas>

<script>
const ctx = document.getElementById('chart').getContext('2d');

const chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Vibration', data: [] }] },
    options: { animation: false }
});

async function update(){
    const r = await fetch('/data');
    const d = await r.json();

    document.getElementById('health').innerText = (d.score || "--") + "%";
    document.getElementById('vib').innerText = d.vibration || 0;
    document.getElementById('z').innerText = d.z_score || 0;
    document.getElementById('trend').innerText = d.trend || "--";
    document.getElementById('freq').innerText = d.frequency || 0;
    document.getElementById('risk').innerText = d.risk || "--";

    chart.data.datasets[0].data.push(d.vibration || 0);
    if(chart.data.datasets[0].data.length > 40)
        chart.data.datasets[0].data.shift();

    chart.update();
}

setInterval(update, 1000);
</script>

</body>
</html>
"""