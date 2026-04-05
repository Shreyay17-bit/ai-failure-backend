import time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

app = FastAPI()

# AI MODELS
model = IsolationForest(contamination=0.05, random_state=42)
trend_model = LinearRegression()

baseline = np.random.normal(0.08, 0.02, (200, 1))
model.fit(baseline)

history = []

last_data = {
    "score": 100,
    "vibration": 0,
    "risk": "NOMINAL",
    "analysis": "System Ready",
    "z_score": 0,
    "trend": "STABLE",
    "frequency": 0,
    "fault": "NONE",
    "battery": 0
}

@app.get("/", response_class=HTMLResponse)
async def home():
    return await monitor()


@app.post("/predict")
async def predict(data: dict):
    global history, last_data

    vib = float(data.get("vibration", 0))
    battery = data.get("battery", 0)

    history.append(vib)
    if len(history) > 50:
        history.pop(0)

    # HEALTH
    score_raw = model.decision_function([[vib]])[0]
    health = int(max(0, min(100, (score_raw + 0.5) * 100)))

    # Z SCORE
    mean = np.mean(history)
    std = np.std(history) if len(history) > 1 else 1
    z = abs((vib - mean) / std) if std > 0 else 0

    # TREND
    trend = "STABLE"
    if len(history) >= 5:
        x = np.arange(len(history)).reshape(-1, 1)
        trend_model.fit(x, history)
        slope = trend_model.coef_[0]

        if slope > 0.002:
            trend = "INCREASING"
        elif slope < -0.002:
            trend = "DECREASING"

    # FREQUENCY
    freq = 0
    if len(history) >= 10:
        fft = np.fft.fft(history)
        freqs = np.fft.fftfreq(len(history))
        freq = abs(freqs[np.argmax(np.abs(fft))])

    # 🔥 FAULT CLASSIFICATION
    fault = "NORMAL"

    if z > 3:
        fault = "SEVERE IMBALANCE"
    elif z > 2:
        fault = "MISALIGNMENT"
    elif freq > 0.2:
        fault = "BEARING FAULT"
    elif trend == "INCREASING":
        fault = "WEAR DETECTED"

    # RISK
    if health < 45 or z > 3:
        risk = "CRITICAL"
    elif health < 75:
        risk = "WARNING"
    else:
        risk = "NOMINAL"

    last_data = {
        "vibration": round(vib, 3),
        "score": health,
        "risk": risk,
        "z_score": round(z, 2),
        "trend": trend,
        "frequency": round(freq, 3),
        "fault": fault,
        "battery": battery
    }

    return last_data


@app.get("/data")
async def get_data():
    return last_data


@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return """
<!DOCTYPE html>
<html>
<head>
<title>AI FUTURE DASHBOARD</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body {
    margin:0;
    font-family:Orbitron, sans-serif;
    background: radial-gradient(circle, #05080a, #000);
    color:#00ffe7;
    text-align:center;
}

h1 {
    margin:20px;
    letter-spacing:5px;
}

.grid {
    display:grid;
    grid-template-columns: repeat(auto-fit, minmax(220px,1fr));
    gap:20px;
    padding:20px;
}

.card {
    background: rgba(0,255,255,0.05);
    border:1px solid rgba(0,255,255,0.2);
    border-radius:15px;
    padding:20px;
    box-shadow:0 0 20px rgba(0,255,255,0.2);
}

.val {
    font-size:30px;
}

.CRITICAL { color:red; }
.WARNING { color:orange; }
.NOMINAL { color:#00ffcc; }

.qr {
    margin-top:20px;
}
</style>
</head>

<body>

<h1>AI NEURAL MONITOR</h1>

<div class="grid">
<div class="card"><div class="val" id="health">--%</div><div>Health</div></div>
<div class="card"><div class="val" id="vib">0</div><div>Vibration</div></div>
<div class="card"><div class="val" id="z">0</div><div>Z Score</div></div>
<div class="card"><div class="val" id="trend">--</div><div>Trend</div></div>
<div class="card"><div class="val" id="freq">0</div><div>Frequency</div></div>
<div class="card"><div class="val" id="fault">--</div><div>Fault Type</div></div>
<div class="card"><div class="val" id="battery">--%</div><div>Battery</div></div>
<div class="card"><div class="val" id="risk">--</div><div>Status</div></div>
</div>

<canvas id="chart"></canvas>

<h3>Scan to Monitor on Phone</h3>
<img class="qr" src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=https://ai-failure-backend.onrender.com">

<script>
const ctx = document.getElementById('chart').getContext('2d');

const chart = new Chart(ctx, {
    type:'line',
    data:{labels:[],datasets:[{label:'Vibration',data:[]}]},
    options:{animation:false}
});

async function update(){
    const r = await fetch('/data');
    const d = await r.json();

    health.innerText = d.score + "%";
    vib.innerText = d.vibration;
    z.innerText = d.z_score;
    trend.innerText = d.trend;
    freq.innerText = d.frequency;
    fault.innerText = d.fault;
    battery.innerText = d.battery + "%";
    risk.innerText = d.risk;

    chart.data.datasets[0].data.push(d.vibration);
    if(chart.data.datasets[0].data.length>40)
        chart.data.datasets[0].data.shift();

    chart.update();
}
setInterval(update,1000);
</script>

</body>
</html>
"""