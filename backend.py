import os, time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import IsolationForest, RandomForestClassifier

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── ML SETUP ──
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

rng = np.random.default_rng(42)
baseline = np.column_stack([
    rng.normal(0.02, 0.01, 300),
    rng.normal(45, 20, 300),
    rng.normal(55, 15, 300),
    rng.normal(80, 15, 300),
    rng.normal(45, 8, 300),
    rng.normal(50, 20, 300),
])
iso.fit(baseline)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(baseline, [0]*300)

# ── STORAGE ──
last = {}

# ── ML FUNCTION ──
def run_ml(vib, cpu, mem, bat, temp, disk):
    feat = np.array([[vib, cpu, mem, bat, temp, disk]])

    health = int(np.clip((iso.decision_function(feat)[0]+0.3)*160,0,100))

    score = int(min(100, (1-health/100)*100))

    risk = "CRITICAL" if score > 60 else "WARNING" if score > 30 else "NOMINAL"

    return {
        "health": health,
        "score": score,
        "risk": risk,
        "cpu": cpu,
        "memory": mem,
        "battery": bat,
        "temp": temp,
        "disk": disk
    }

# ── API ──
@app.post("/predict")
async def predict(data: dict):
    global last
    last = run_ml(
        float(data.get("vibration",0.02)),
        float(data.get("cpu",50)),
        float(data.get("memory",50)),
        float(data.get("battery",-1)),
        float(data.get("temp",40)),
        float(data.get("disk",50)),
    )
    return last

@app.get("/data")
async def get_data():
    return last if last else {"error":"Waiting for data..."}

# ── FULL FRONTEND INSIDE BACKEND ──
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Nexus PM</title>
<style>
body{background:black;color:white;font-family:Arial;text-align:center}
h1{color:#00ff9d}
.box{margin:10px;font-size:22px}
button{padding:10px 20px;background:#00ff9d;border:none;cursor:pointer}
</style>
</head>
<body>

<h1>🔥 NEXUS PM (ALL-IN-ONE)</h1>

<div class="box" id="cpu">CPU: --%</div>
<div class="box" id="mem">Memory: --%</div>
<div class="box" id="temp">Temp: --°C</div>
<div class="box" id="risk">Risk: --</div>

<button onclick="startPhone()">Enable Phone Sensor</button>

<script>

// 🔁 FETCH BACKEND DATA
async function fetchData(){
  try{
    const res = await fetch('/data');
    const d = await res.json();
    if(d.error) return;

    document.getElementById("cpu").innerText = "CPU: " + d.cpu + "%";
    document.getElementById("mem").innerText = "Memory: " + d.memory + "%";
    document.getElementById("temp").innerText = "Temp: " + d.temp + "°C";
    document.getElementById("risk").innerText = "Risk: " + d.risk;
  }catch(e){}
}

setInterval(fetchData,2000);


// 📱 ANDROID SENSOR FIX
let lastVib = 0;

function startPhone(){
  window.addEventListener('devicemotion', async e => {
    const a = e.accelerationIncludingGravity;
    if(!a) return;

    let raw = Math.sqrt(a.x*a.x + a.y*a.y + a.z*a.z) - 9.81;

    let vib = 0.8 * lastVib + 0.2 * raw;
    lastVib = vib;

    if(vib < 0.02) vib = 0;

    await fetch('/predict',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        vibration: vib,
        cpu:50,
        memory:60,
        temp:36,
        disk:50
      })
    });
  });
}

</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML

# ── RUN ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)