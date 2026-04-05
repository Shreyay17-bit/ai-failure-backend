import time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

app = FastAPI()

# ---------------- AI MODELS ----------------
model = IsolationForest(contamination=0.05, random_state=42)
trend_model = LinearRegression()

baseline = np.random.normal(0.08, 0.02, (200, 1))
model.fit(baseline)

# ---------------- DATA ----------------
history = []
battery_history = []
time_history = []

last_data = {
    "score": 100,
    "vibration": 0,
    "z_score": 0,
    "trend": "STABLE",
    "frequency": 0,
    "fault": "NORMAL",
    "battery": -1,
    "battery_drain": "N/A",
    "risk": "NOMINAL"
}


@app.get("/", response_class=HTMLResponse)
async def home():
    return await monitor()


@app.post("/predict")
async def predict(data: dict):
    global history, battery_history, time_history, last_data

    vib = float(data.get("vibration", 0))
    battery = int(data.get("battery", -1))

    history.append(vib)
    if len(history) > 50:
        history.pop(0)

    # ---------- HEALTH ----------
    score_raw = model.decision_function([[vib]])[0]
    health = int(max(0, min(100, (score_raw + 0.5) * 100)))

    # ---------- Z SCORE ----------
    mean = np.mean(history)
    std = np.std(history) if len(history) > 1 else 1
    z = abs((vib - mean) / std)

    # ---------- TREND ----------
    trend = "STABLE"
    if len(history) >= 5:
        x = np.arange(len(history)).reshape(-1, 1)
        trend_model.fit(x, history)
        slope = trend_model.coef_[0]

        if slope > 0.002:
            trend = "INCREASING"
        elif slope < -0.002:
            trend = "DECREASING"

    # ---------- FREQUENCY ----------
    freq = 0
    if len(history) >= 10:
        fft = np.fft.fft(history)
        freqs = np.fft.fftfreq(len(history))
        freq = abs(freqs[np.argmax(np.abs(fft))])

    # ---------- FAULT CLASSIFICATION ----------
    fault = "NORMAL"
    if z > 3:
        fault = "SEVERE IMBALANCE"
    elif z > 2:
        fault = "MISALIGNMENT"
    elif freq > 0.2:
        fault = "BEARING FAULT"
    elif trend == "INCREASING":
        fault = "WEAR DETECTED"

    # ---------- BATTERY PREDICTION ----------
    current_time = time.time()

    if battery >= 0:
        battery_history.append(battery)
        time_history.append(current_time)

        if len(battery_history) > 20:
            battery_history.pop(0)
            time_history.pop(0)

    battery_drain = "N/A"

    if len(battery_history) >= 5:
        x = np.array(time_history).reshape(-1, 1)
        y = np.array(battery_history)

        trend_model.fit(x, y)
        slope = trend_model.coef_[0]

        if slope < 0:
            minutes_left = abs(battery / slope) / 60
            battery_drain = f"{int(minutes_left)} min remaining"

    # ---------- RISK ----------
    if health < 45 or z > 3:
        risk = "CRITICAL"
    elif health < 75:
        risk = "WARNING"
    else:
        risk = "NOMINAL"

    last_data = {
        "score": health,
        "vibration": round(vib, 3),
        "z_score": round(z, 2),
        "trend": trend,
        "frequency": round(freq, 3),
        "fault": fault,
        "battery": battery,
        "battery_drain": battery_drain,
        "risk": risk
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
    background:black;
    color:#00ffe7;
    font-family:sans-serif;
    text-align:center;
}

.grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:15px;
    padding:20px;
}

.card {
    background:#111;
    padding:20px;
    border-radius:10px;
}

button {
    padding:15px;
    background:#00ffe7;
    border:none;
    margin:20px;
    cursor:pointer;
}
</style>
</head>

<body>

<h1>AI NEURAL MONITOR</h1>

<button onclick="startSensors()">START SENSOR</button>

<div class="grid">
<div class="card"><h2 id="health">--</h2>Health</div>
<div class="card"><h2 id="vib">--</h2>Vibration</div>
<div class="card"><h2 id="z">--</h2>Z Score</div>
<div class="card"><h2 id="trend">--</h2>Trend</div>
<div class="card"><h2 id="freq">--</h2>Frequency</div>
<div class="card"><h2 id="fault">--</h2>Fault</div>
<div class="card"><h2 id="battery">--</h2>Battery</div>
<div class="card"><h2 id="drain">--</h2>Battery Life</div>
<div class="card"><h2 id="risk">--</h2>Status</div>
</div>

<h3>Scan QR</h3>
<img src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=https://ai-failure-backend.onrender.com">

<script>
let batteryLevel = -1;

// BATTERY API
if (navigator.getBattery) {
    navigator.getBattery().then(function(battery) {
        function updateBattery() {
            batteryLevel = Math.round(battery.level * 100);
        }
        updateBattery();
        battery.addEventListener('levelchange', updateBattery);
    });
}

// SENSOR START
function startSensors() {
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
        DeviceMotionEvent.requestPermission()
        .then(p => { if(p === 'granted') activate(); });
    } else {
        activate();
    }
}

function activate() {
    window.addEventListener("devicemotion", function(e) {
        let a = e.accelerationIncludingGravity;
        if(!a) return;

        let vib = Math.sqrt(a.x*a.x + a.y*a.y + a.z*a.z) - 9.81;

        fetch('/predict', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({
                vibration: vib,
                battery: batteryLevel
            })
        });
    });
}

// UI UPDATE
async function update(){
    let r = await fetch('/data');
    let d = await r.json();

    health.innerText = d.score + "%";
    vib.innerText = d.vibration;
    z.innerText = d.z_score;
    trend.innerText = d.trend;
    freq.innerText = d.frequency;
    fault.innerText = d.fault;
    battery.innerText = d.battery >= 0 ? d.battery + "%" : "N/A";
    drain.innerText = d.battery_drain;
    risk.innerText = d.risk;
}

setInterval(update,1000);
</script>

</body>
</html>
"""