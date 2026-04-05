"""
Nexus PM - Predictive Maintenance Backend
3 modes: Live System | Industrial | Custom Input
ML: Isolation Forest + Random Forest + Linear Regression + Z-score + FFT
"""
import os, time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression

app = FastAPI(title="Nexus PM API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── ML MODELS ─────────────────────────────────────────────────────────────────

# 1. Isolation Forest — unsupervised anomaly detection on 6 features
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
baseline = np.column_stack([
    np.random.normal(0.08, 0.02, 500),
    np.random.normal(60,   15,   500),
    np.random.normal(55,   15,   500),
    np.random.normal(75,   15,   500),
    np.random.normal(35,    5,   500),
    np.random.normal(50,   20,   500),
])
iso.fit(baseline)

# 2. Random Forest — supervised fault classification
rf = RandomForestClassifier(n_estimators=100, random_state=42)
_faults = ["NORMAL","BEARING_FAULT","MISALIGNMENT","IMBALANCE","OVERHEATING","WEAR"]
_Xf, _yf = [], []
for i, f in enumerate(_faults):
    for _ in range(80):
        row = [
            np.random.normal([0.05,0.08,0.15,0.20,0.05,0.12][i], 0.01),
            np.random.normal([40,50,70,80,95,60][i], 8),
            np.random.normal([40,55,65,70,85,60][i], 8),
            np.random.normal([85,75,65,60,50,70][i], 8),
            np.random.normal([32,36,38,40,55,42][i], 3),
            np.random.normal([40,50,55,60,70,65][i], 8),
        ]
        _Xf.append(row); _yf.append(i)
rf.fit(_Xf, _yf)

# 3. Linear Regression — trend + battery life prediction
trend_model = LinearRegression()

# ── STORAGE ───────────────────────────────────────────────────────────────────
history = {k: [] for k in ["vibration","cpu","memory","battery","temp","disk","score","time"]}
MAX_HIST = 60
last_result = {}

def push(key, val):
    history[key].append(val)
    if len(history[key]) > MAX_HIST:
        history[key].pop(0)

# ── CORE ML ENGINE ────────────────────────────────────────────────────────────
def run_ml(vib, cpu, mem, bat, temp, disk, device="system"):
    t = time.time()
    feat = np.array([[vib, cpu, mem, bat, temp, disk]])

    # Isolation Forest anomaly score → health %
    iso_score = iso.decision_function(feat)[0]
    health = int(np.clip((iso_score + 0.5) * 130, 0, 100))

    # Push to history
    push("vibration", vib); push("cpu", cpu); push("memory", mem)
    push("battery", bat); push("temp", temp); push("disk", disk); push("time", t)

    # Z-score on vibration
    vib_arr = np.array(history["vibration"])
    z = 0.0
    if len(vib_arr) > 2:
        z = abs((vib - vib_arr.mean()) / (vib_arr.std() + 1e-9))

    # Vibration trend via Linear Regression
    trend, slope = "STABLE", 0.0
    if len(vib_arr) >= 5:
        x = np.arange(len(vib_arr)).reshape(-1, 1)
        trend_model.fit(x, vib_arr)
        slope = float(trend_model.coef_[0])
        trend = "RISING" if slope > 0.002 else "FALLING" if slope < -0.002 else "STABLE"

    # FFT dominant frequency
    freq = 0.0
    if len(vib_arr) >= 10:
        fft = np.abs(np.fft.fft(vib_arr))
        freqs = np.fft.fftfreq(len(vib_arr))
        freq = float(abs(freqs[np.argmax(fft[1:]) + 1]))

    # Random Forest fault classification
    fault_idx = int(rf.predict(feat)[0])
    fault = _faults[fault_idx]
    fault_proba = float(rf.predict_proba(feat)[0][fault_idx])
    if z > 3.5:   fault = "IMBALANCE"
    elif z > 2.5: fault = "MISALIGNMENT"
    if temp > 70: fault = "OVERHEATING"

    # RUL estimate
    rul_hours = None
    if len(vib_arr) >= 10 and slope > 0:
        steps_to_critical = max(0, (0.8 - vib) / (slope + 1e-9))
        rul_hours = round(steps_to_critical * 2 / 60, 1)

    # Battery life via Linear Regression
    bat_life = "N/A"
    bat_arr = np.array(history["battery"])
    t_arr   = np.array(history["time"])
    if len(bat_arr) >= 5 and bat >= 0:
        trend_model.fit(t_arr.reshape(-1,1), bat_arr)
        drain_per_sec = trend_model.coef_[0]
        if drain_per_sec < 0:
            mins = abs(bat / drain_per_sec) / 60
            bat_life = f"{int(mins)} min" if mins < 120 else f"{mins/60:.1f} hr"

    # Risk score
    risk_score = 0
    if health < 40:   risk_score += 40
    elif health < 65: risk_score += 20
    if z > 3:         risk_score += 25
    elif z > 2:       risk_score += 12
    if temp > 70:     risk_score += 20
    elif temp > 55:   risk_score += 10
    if bat < 15:      risk_score += 15
    elif bat < 30:    risk_score += 7
    if cpu > 90:      risk_score += 10
    if mem > 90:      risk_score += 8
    risk_score = min(risk_score, 100)
    risk = "CRITICAL" if risk_score >= 60 else "WARNING" if risk_score >= 30 else "NOMINAL"

    # Recommendations
    recs = []
    if temp > 70:       recs.append("🔥 Reduce load — thermal critical")
    if bat < 20:        recs.append("🔋 Connect charger immediately")
    if cpu > 85:        recs.append("⚙️ Close background processes")
    if mem > 85:        recs.append("💾 Clear memory cache")
    if z > 2.5:         recs.append("📳 Abnormal vibration — inspect mounting")
    if trend == "RISING": recs.append("📈 Vibration rising — check bearings")
    if disk > 90:       recs.append("💿 Disk nearly full — free space")
    if not recs:        recs.append("✅ All parameters nominal")

    push("score", risk_score)

    return {
        "health": health, "score": risk_score, "risk": risk,
        "fault": fault, "fault_confidence": round(fault_proba * 100, 1),
        "vibration": round(vib, 4), "z_score": round(z, 3),
        "trend": trend, "slope": round(slope, 5),
        "frequency": round(freq, 4), "rul_hours": rul_hours,
        "cpu": round(cpu, 1), "memory": round(mem, 1),
        "battery": round(bat, 1), "temp": round(temp, 1), "disk": round(disk, 1),
        "battery_life": bat_life, "device": device,
        "recommendations": recs,
        "history": {k: history[k][-30:] for k in ["score","vibration","cpu","memory","temp"]},
        "timestamp": round(t * 1000),
    }

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return await monitor()

@app.post("/predict")
async def predict(data: dict):
    global last_result
    last_result = run_ml(
        float(data.get("vibration", 0.05)),
        float(data.get("cpu", data.get("cpu_load", 50))),
        float(data.get("memory", 50)),
        float(data.get("battery", 75)),
        float(data.get("temp", data.get("temperature", 35))),
        float(data.get("disk", 50)),
        str(data.get("device", data.get("device_type", "system")))
    )
    return last_result

@app.get("/data")
async def get_data():
    return last_result if last_result else {"error": "No data yet"}

@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return DASHBOARD_HTML

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NEXUS PM — Predictive Maintenance</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#04080f;--bg2:#080f1a;--bg3:#0d1825;--bg4:#122030;--cyan:#00e5ff;--green:#00ff9d;--yellow:#ffe600;--red:#ff2d55;--orange:#ff6b00;--text:#c8dff0;--text2:#5a7a99;--text3:#2a3f55;--r:8px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,.012) 2px,rgba(0,229,255,.012) 4px);pointer-events:none;z-index:0}
.shell{max-width:1400px;margin:0 auto;padding:14px;position:relative;z-index:1}
.mono{font-family:'Share Tech Mono',monospace}
.topbar{display:flex;align-items:center;gap:12px;padding:10px 18px;background:var(--bg2);border:1px solid #0a2a40;border-radius:var(--r);margin-bottom:14px;flex-wrap:wrap;gap:10px}
.logo{font-family:'Share Tech Mono',monospace;font-size:15px;color:var(--cyan);letter-spacing:3px}
.logo span{color:var(--green)}
.mode-tabs{display:flex;gap:6px;flex:1;justify-content:center}
.mtab{padding:5px 16px;border:1px solid var(--text3);border-radius:5px;cursor:pointer;font-size:13px;font-weight:600;letter-spacing:1px;background:transparent;color:var(--text2);transition:.2s;font-family:'Rajdhani',sans-serif}
.mtab.active{border-color:var(--cyan);color:var(--cyan);background:rgba(0,229,255,.08)}
.pulse-dot{width:9px;height:9px;border-radius:50%;background:var(--green);animation:pdot 1.5s infinite;flex-shrink:0}
@keyframes pdot{0%,100%{box-shadow:0 0 0 0 rgba(0,255,157,.5)}70%{box-shadow:0 0 0 8px rgba(0,255,157,0)}}
.risk-banner{padding:12px 18px;border-radius:var(--r);margin-bottom:14px;display:flex;align-items:center;gap:16px;border:1px solid;transition:all .4s;flex-wrap:wrap}
.risk-banner.NOMINAL{background:rgba(0,255,157,.06);border-color:rgba(0,255,157,.3)}
.risk-banner.WARNING{background:rgba(255,230,0,.06);border-color:rgba(255,230,0,.4);animation:wpulse 2s infinite}
.risk-banner.CRITICAL{background:rgba(255,45,85,.1);border-color:rgba(255,45,85,.6);animation:cpulse .8s infinite}
@keyframes wpulse{0%,100%{border-color:rgba(255,230,0,.4)}50%{border-color:rgba(255,230,0,.9)}}
@keyframes cpulse{0%,100%{border-color:rgba(255,45,85,.6)}50%{border-color:rgba(255,45,85,1)}}
.risk-icon{font-size:30px}
.risk-text h2{font-size:22px;font-weight:700;letter-spacing:3px}
.risk-text p{font-size:13px;color:var(--text2);margin-top:2px}
.risk-score-big{margin-left:auto;text-align:right}
.risk-score-big .num{font-family:'Share Tech Mono',monospace;font-size:46px;line-height:1}
.risk-score-big .lbl{font-size:10px;color:var(--text3);letter-spacing:2px}
.main-grid{display:grid;grid-template-columns:260px 1fr 280px;gap:14px;margin-bottom:14px}
@media(max-width:1100px){.main-grid{grid-template-columns:1fr 1fr}}
@media(max-width:700px){.main-grid{grid-template-columns:1fr}}
.card{background:var(--bg2);border:1px solid #0a2a40;border-radius:var(--r);padding:16px;animation:fadein .4s ease both}
@keyframes fadein{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.card-title{font-size:10px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.card-title::before{content:'';width:3px;height:12px;background:var(--cyan);border-radius:2px}
.gauge-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.gauge-wrap{display:flex;flex-direction:column;align-items:center;gap:4px}
.gauge-svg{width:110px;height:65px;overflow:visible}
.gauge-label{font-size:11px;color:var(--text2);letter-spacing:1px;text-align:center}
.gauge-val{font-family:'Share Tech Mono',monospace;font-size:17px;font-weight:700;text-align:center}
.big-gauge-wrap{display:flex;flex-direction:column;align-items:center;padding:8px 0}
.big-gauge-svg{width:200px;height:120px;overflow:visible}
.big-gauge-num{font-family:'Share Tech Mono',monospace;font-size:46px;font-weight:700;text-align:center;line-height:1;margin-top:-6px}
.big-gauge-sub{font-size:12px;color:var(--text2);text-align:center;margin-top:4px;letter-spacing:1px}
.metric-row{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--bg4)}
.metric-row:last-child{border-bottom:none}
.metric-icon{width:28px;height:28px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.metric-info{flex:1;min-width:0}
.metric-name{font-size:12px;color:var(--text2);font-weight:600;letter-spacing:.5px}
.metric-val{font-family:'Share Tech Mono',monospace;font-size:16px;font-weight:700;line-height:1.2}
.metric-bar{height:3px;background:var(--bg4);border-radius:2px;margin-top:4px}
.metric-bar-fill{height:100%;border-radius:2px;transition:width .6s ease}
.fault-box{border-radius:var(--r);padding:14px 16px;margin-bottom:12px;border:1px solid}
.fault-name{font-size:18px;font-weight:700;letter-spacing:2px;margin-bottom:4px}
.fault-conf{font-size:12px;color:var(--text2)}
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.stat-box{background:var(--bg3);border:1px solid var(--bg4);border-radius:6px;padding:10px 12px}
.stat-label{font-size:10px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px}
.stat-val{font-family:'Share Tech Mono',monospace;font-size:15px;font-weight:700}
.spark-container{position:relative;height:80px;margin-top:8px}
.rec-item{display:flex;align-items:flex-start;gap:8px;padding:7px 0;border-bottom:1px solid var(--bg4);font-size:13px;font-weight:500}
.rec-item:last-child{border-bottom:none}
.inp-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.inp-group{display:flex;flex-direction:column;gap:5px}
.inp-group label{font-size:11px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase}
.inp-group input{background:var(--bg3);border:1px solid var(--bg4);border-radius:6px;color:var(--text);font-family:'Share Tech Mono',monospace;font-size:14px;padding:8px 12px;outline:none;transition:.2s;width:100%}
.inp-group input:focus{border-color:var(--cyan);box-shadow:0 0 0 2px rgba(0,229,255,.1)}
.inp-group .hint{font-size:10px;color:var(--text3)}
.submit-btn{width:100%;padding:12px;background:rgba(0,229,255,.1);border:1px solid var(--cyan);border-radius:var(--r);color:var(--cyan);font-family:'Rajdhani',sans-serif;font-size:15px;font-weight:700;letter-spacing:2px;cursor:pointer;transition:.2s}
.submit-btn:hover{background:rgba(0,229,255,.2)}
.start-btn{padding:14px 32px;background:rgba(0,255,157,.1);border:1px solid var(--green);border-radius:var(--r);color:var(--green);font-family:'Rajdhani',sans-serif;font-size:16px;font-weight:700;letter-spacing:2px;cursor:pointer;transition:.2s;margin:8px}
.start-btn:hover{background:rgba(0,255,157,.2)}
.bottom-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
@media(max-width:900px){.bottom-grid{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.bottom-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="shell">

<div class="topbar">
  <div class="logo mono">NEXUS<span>/PM</span></div>
  <div class="mode-tabs">
    <button class="mtab active" onclick="setMode('live')" id="tab-live">⚡ LIVE SYSTEM</button>
    <button class="mtab" onclick="setMode('industrial')" id="tab-industrial">🏭 INDUSTRIAL</button>
    <button class="mtab" onclick="setMode('custom')" id="tab-custom">✏️ CUSTOM INPUT</button>
  </div>
  <div class="pulse-dot"></div>
  <div class="mono" style="font-size:11px;color:var(--text3)" id="clock">--:--:--</div>
</div>

<div class="risk-banner NOMINAL" id="risk-banner">
  <div class="risk-icon" id="risk-icon">✅</div>
  <div class="risk-text">
    <h2 id="risk-title" style="color:var(--green)">NOMINAL</h2>
    <p id="risk-desc">Awaiting sensor data</p>
  </div>
  <div class="risk-score-big">
    <div class="num" id="big-score" style="color:var(--green)">--</div>
    <div class="lbl">RISK SCORE / 100</div>
  </div>
</div>

<!-- LIVE MODE -->
<div id="mode-live">
  <div id="phone-prompt" class="card" style="margin-bottom:14px;text-align:center;padding:24px;display:none">
    <div class="card-title" style="justify-content:center">PHONE SENSOR ACCESS</div>
    <div style="font-size:48px;margin:8px 0">📱</div>
    <p style="color:var(--text2);font-size:13px;margin-bottom:16px">Tap to grant motion + battery sensor access</p>
    <button class="start-btn" onclick="startPhoneSensors()">▶ GRANT ACCESS</button>
  </div>

  <div class="main-grid">
    <div style="display:flex;flex-direction:column;gap:14px">
      <div class="card">
        <div class="card-title">Health Index</div>
        <div class="big-gauge-wrap">
          <svg class="big-gauge-svg" viewBox="0 0 200 120">
            <defs>
              <linearGradient id="ag" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#00ff9d"/>
                <stop offset="50%" stop-color="#ffe600"/>
                <stop offset="100%" stop-color="#ff2d55"/>
              </linearGradient>
            </defs>
            <path d="M20,110 A80,80 0 0,1 180,110" fill="none" stroke="#0d1825" stroke-width="18" stroke-linecap="round"/>
            <path id="big-arc" d="M20,110 A80,80 0 0,1 180,110" fill="none" stroke="url(#ag)" stroke-width="18" stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="251.2" style="transition:stroke-dashoffset .8s ease"/>
          </svg>
          <div class="big-gauge-num" id="health-num" style="color:var(--green)">--</div>
          <div class="big-gauge-sub" id="health-lbl">AWAITING DATA</div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Fault Classification</div>
        <div class="fault-box" id="fault-box" style="background:rgba(0,255,157,.06);border-color:rgba(0,255,157,.3)">
          <div class="fault-name" id="fault-name" style="color:var(--green)">NORMAL</div>
          <div class="fault-conf" id="fault-conf">Confidence: -- | Model: Random Forest</div>
        </div>
        <div class="stats-grid">
          <div class="stat-box"><div class="stat-label">Z-Score</div><div class="stat-val" id="z-score" style="color:var(--cyan)">--</div></div>
          <div class="stat-box"><div class="stat-label">Frequency</div><div class="stat-val" id="freq-val" style="color:var(--cyan)">--</div></div>
          <div class="stat-box"><div class="stat-label">Trend</div><div class="stat-val" id="trend-val">--</div></div>
          <div class="stat-box"><div class="stat-label">RUL Est.</div><div class="stat-val" id="rul-val" style="color:var(--yellow)">--</div></div>
        </div>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:14px">
      <div class="card">
        <div class="card-title">Parameter Gauges</div>
        <div class="gauge-grid" id="gauge-grid"><div style="color:var(--text3);font-size:12px;grid-column:span 2">Awaiting data...</div></div>
      </div>
      <div class="card">
        <div class="card-title">Live Metrics</div>
        <div id="metrics-list"><div style="color:var(--text3);font-size:12px">Awaiting data...</div></div>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:14px">
      <div class="card">
        <div class="card-title">AI Recommendations</div>
        <div id="recs-list"><div style="color:var(--text3);font-size:13px">Awaiting analysis...</div></div>
      </div>
      <div class="card">
        <div class="card-title">Battery Intelligence</div>
        <div style="text-align:center;padding:8px 0">
          <div class="mono" style="font-size:38px;color:var(--green)" id="bat-pct">--%</div>
          <div style="font-size:12px;color:var(--text2);margin-top:4px" id="bat-life">Life: --</div>
        </div>
        <div style="height:6px;background:var(--bg4);border-radius:3px;margin-bottom:12px"><div id="bat-bar" style="height:100%;border-radius:3px;width:0%;background:var(--green);transition:width .6s"></div></div>
        <div class="stats-grid">
          <div class="stat-box"><div class="stat-label">Vibration</div><div class="stat-val mono" id="vib-val" style="color:var(--cyan)">--</div></div>
          <div class="stat-box"><div class="stat-label">Device</div><div class="stat-val" id="dev-val" style="font-size:12px">--</div></div>
        </div>
      </div>
    </div>
  </div>

  <div class="bottom-grid">
    <div class="card"><div class="card-title">Risk Score History</div><div class="spark-container"><canvas id="c-score"></canvas></div></div>
    <div class="card"><div class="card-title">Vibration History</div><div class="spark-container"><canvas id="c-vib"></canvas></div></div>
    <div class="card"><div class="card-title">CPU + Memory</div><div class="spark-container"><canvas id="c-cpu"></canvas></div></div>
  </div>
</div>

<!-- INDUSTRIAL MODE -->
<div id="mode-industrial" style="display:none">
  <div class="card">
    <div class="card-title">Industrial Machine Parameters</div>
    <div class="inp-grid">
      <div class="inp-group"><label>Vibration (m/s²)</label><input type="number" id="ind-vib" value="0.08" step="0.01" min="0"/><div class="hint">Normal &lt;0.15 | Warning &gt;0.5 | Critical &gt;0.8</div></div>
      <div class="inp-group"><label>Temperature (°C)</label><input type="number" id="ind-temp" value="38" step="1"/><div class="hint">Normal &lt;55 | Warning 55–70 | Critical &gt;70</div></div>
      <div class="inp-group"><label>RPM / Speed (%)</label><input type="number" id="ind-cpu" value="65" step="1" min="0" max="100"/><div class="hint">Machine load percentage</div></div>
      <div class="inp-group"><label>Pressure / Load (%)</label><input type="number" id="ind-mem" value="55" step="1" min="0" max="100"/><div class="hint">Hydraulic or mechanical load</div></div>
      <div class="inp-group"><label>Power Supply (%)</label><input type="number" id="ind-bat" value="90" step="1" min="0" max="100"/><div class="hint">Input power health</div></div>
      <div class="inp-group"><label>Wear Index (%)</label><input type="number" id="ind-disk" value="40" step="1" min="0" max="100"/><div class="hint">Cumulative wear / cycle count</div></div>
    </div>
    <button class="submit-btn" onclick="submitIndustrial()">⚙️ RUN ML ANALYSIS</button>
  </div>
</div>

<!-- CUSTOM MODE -->
<div id="mode-custom" style="display:none">
  <div class="card">
    <div class="card-title">Custom Parameter Input</div>
    <div class="inp-grid">
      <div class="inp-group"><label>Vibration / Anomaly Signal</label><input type="number" id="cv-in" value="0.08" step="0.001"/></div>
      <div class="inp-group"><label>CPU / Primary Load (%)</label><input type="number" id="cc-in" value="50" step="1" min="0" max="100"/></div>
      <div class="inp-group"><label>Memory / Secondary Load (%)</label><input type="number" id="cm-in" value="50" step="1" min="0" max="100"/></div>
      <div class="inp-group"><label>Battery / Power (%)</label><input type="number" id="cb-in" value="80" step="1" min="0" max="100"/></div>
      <div class="inp-group"><label>Temperature (°C)</label><input type="number" id="ct-in" value="35" step="1"/></div>
      <div class="inp-group"><label>Disk / Wear (%)</label><input type="number" id="cd-in" value="50" step="1" min="0" max="100"/></div>
    </div>
    <button class="submit-btn" onclick="submitCustom()">🧠 ANALYZE WITH ML</button>
  </div>
</div>

</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
let mode='live', charts={}, phoneActive=false, batteryLevel=-1;

setInterval(()=>{document.getElementById('clock').textContent=new Date().toLocaleTimeString('en',{hour12:false})},1000);

function setMode(m){
  mode=m;
  ['live','industrial','custom'].forEach(k=>{
    document.getElementById('mode-'+k).style.display=k===m?'block':'none';
    document.getElementById('tab-'+k).classList.toggle('active',k===m);
  });
  if(m==='live'&&/Mobi|Android/i.test(navigator.userAgent)&&!phoneActive)
    document.getElementById('phone-prompt').style.display='block';
}

if(/Mobi|Android/i.test(navigator.userAgent))document.getElementById('phone-prompt').style.display='block';

if(navigator.getBattery)navigator.getBattery().then(b=>{
  batteryLevel=Math.round(b.level*100);
  b.addEventListener('levelchange',()=>{batteryLevel=Math.round(b.level*100);});
}).catch(()=>{});

function colorFor(v,lo,hi){return v>=hi?'#ff2d55':v>=lo?'#ffe600':'#00ff9d'}

function render(d){
  if(!d||d.error)return;
  const colors={NOMINAL:'#00ff9d',WARNING:'#ffe600',CRITICAL:'#ff2d55'};
  const icons={NOMINAL:'✅',WARNING:'⚠️',CRITICAL:'🚨'};
  const descs={NOMINAL:'All systems operating within normal parameters',WARNING:'Elevated risk — schedule maintenance soon',CRITICAL:'Multiple failure indicators — immediate action required'};
  const col=colors[d.risk]||'#00ff9d';

  // Banner
  document.getElementById('risk-banner').className='risk-banner '+(d.risk||'NOMINAL');
  document.getElementById('risk-icon').textContent=icons[d.risk]||'✅';
  document.getElementById('risk-title').textContent=d.risk||'--';
  document.getElementById('risk-title').style.color=col;
  document.getElementById('risk-desc').textContent=descs[d.risk]||'';
  document.getElementById('big-score').textContent=d.score;
  document.getElementById('big-score').style.color=col;

  // Big gauge
  document.getElementById('big-arc').style.strokeDashoffset=251.2*(1-d.health/100);
  document.getElementById('health-num').textContent=d.health+'%';
  document.getElementById('health-num').style.color=col;
  document.getElementById('health-lbl').textContent=d.risk+' — '+(d.fault||'').replace(/_/g,' ');

  // Fault
  const fBg={NORMAL:'rgba(0,255,157,.06)',BEARING_FAULT:'rgba(255,107,0,.1)',MISALIGNMENT:'rgba(255,230,0,.08)',IMBALANCE:'rgba(255,45,85,.1)',OVERHEATING:'rgba(255,45,85,.15)',WEAR:'rgba(255,230,0,.1)'};
  const fBd={NORMAL:'rgba(0,255,157,.3)',BEARING_FAULT:'rgba(255,107,0,.5)',MISALIGNMENT:'rgba(255,230,0,.4)',IMBALANCE:'rgba(255,45,85,.5)',OVERHEATING:'rgba(255,45,85,.7)',WEAR:'rgba(255,230,0,.5)'};
  const fTx={NORMAL:'#00ff9d',BEARING_FAULT:'#ff6b00',MISALIGNMENT:'#ffe600',IMBALANCE:'#ff2d55',OVERHEATING:'#ff2d55',WEAR:'#ffe600'};
  const fb=document.getElementById('fault-box');
  fb.style.background=fBg[d.fault]||fBg.NORMAL;
  fb.style.borderColor=fBd[d.fault]||fBd.NORMAL;
  document.getElementById('fault-name').textContent=(d.fault||'NORMAL').replace(/_/g,' ');
  document.getElementById('fault-name').style.color=fTx[d.fault]||'#00ff9d';
  document.getElementById('fault-conf').textContent=`Confidence: ${d.fault_confidence}% | Model: Random Forest`;

  // Stats
  const ze=document.getElementById('z-score');
  ze.textContent=d.z_score;ze.style.color=d.z_score>3?'#ff2d55':d.z_score>2?'#ffe600':'#00e5ff';
  document.getElementById('freq-val').textContent=d.frequency+' Hz';
  const te=document.getElementById('trend-val');
  te.textContent=d.trend;te.style.color=d.trend==='RISING'?'#ff2d55':d.trend==='FALLING'?'#ffe600':'#00ff9d';
  document.getElementById('rul-val').textContent=d.rul_hours!=null?d.rul_hours+'h':'N/A';
  document.getElementById('vib-val').textContent=d.vibration;
  document.getElementById('dev-val').textContent=d.device||'--';

  // Battery
  const bc=colorFor(100-d.battery,60,80);
  document.getElementById('bat-pct').textContent=d.battery>=0?d.battery+'%':'N/A';
  document.getElementById('bat-pct').style.color=d.battery<20?'#ff2d55':d.battery<40?'#ffe600':'#00ff9d';
  document.getElementById('bat-life').textContent='Est. Life: '+d.battery_life;
  document.getElementById('bat-bar').style.width=Math.max(0,d.battery)+'%';
  document.getElementById('bat-bar').style.background=d.battery<20?'#ff2d55':d.battery<40?'#ffe600':'#00ff9d';

  // Recs
  document.getElementById('recs-list').innerHTML=(d.recommendations||[]).map(r=>`<div class="rec-item">${r}</div>`).join('')||'<div style="color:var(--text3);font-size:13px">No issues detected</div>';

  // Gauge grid
  const gauges=[
    {label:'CPU',val:d.cpu,unit:'%',pct:d.cpu,col:colorFor(d.cpu,70,90)},
    {label:'MEMORY',val:d.memory,unit:'%',pct:d.memory,col:colorFor(d.memory,70,85)},
    {label:'TEMP',val:d.temp,unit:'°C',pct:Math.min(d.temp,100),col:colorFor(d.temp,55,70)},
    {label:'DISK',val:d.disk,unit:'%',pct:d.disk,col:colorFor(d.disk,80,92)},
  ];
  document.getElementById('gauge-grid').innerHTML=gauges.map((g,i)=>`
    <div class="gauge-wrap">
      <svg class="gauge-svg" viewBox="0 0 110 65" style="overflow:visible">
        <path d="M10,60 A45,45 0 0,1 100,60" fill="none" stroke="#0d1825" stroke-width="10" stroke-linecap="round"/>
        <path d="M10,60 A45,45 0 0,1 100,60" fill="none" stroke="${g.col}" stroke-width="10" stroke-linecap="round"
          stroke-dasharray="141.4" stroke-dashoffset="${141.4*(1-g.pct/100)}" style="transition:stroke-dashoffset .6s"/>
      </svg>
      <div class="gauge-val" style="color:${g.col}">${g.val.toFixed(1)}${g.unit}</div>
      <div class="gauge-label">${g.label}</div>
    </div>`).join('');

  // Metrics
  const metrics=[
    {icon:'⚙️',name:'CPU Load',val:d.cpu+'%',bar:d.cpu,col:colorFor(d.cpu,70,90)},
    {icon:'💾',name:'Memory',val:d.memory+'%',bar:d.memory,col:colorFor(d.memory,70,85)},
    {icon:'🌡️',name:'Temperature',val:d.temp+'°C',bar:Math.min(d.temp,100),col:colorFor(d.temp,55,70)},
    {icon:'💿',name:'Disk',val:d.disk+'%',bar:d.disk,col:colorFor(d.disk,80,92)},
  ];
  document.getElementById('metrics-list').innerHTML=metrics.map(m=>`
    <div class="metric-row">
      <div class="metric-icon" style="background:${m.col}22">${m.icon}</div>
      <div class="metric-info">
        <div class="metric-name">${m.name}</div>
        <div class="metric-val" style="color:${m.col}">${m.val}</div>
        <div class="metric-bar"><div class="metric-bar-fill" style="width:${m.bar}%;background:${m.col}"></div></div>
      </div>
    </div>`).join('');

  // Charts
  if(d.history){
    const h=d.history, labels=h.score.map((_,i)=>i);
    function pushChart(c,datasets){c.data.labels=labels;datasets.forEach((data,i)=>{c.data.datasets[i].data=data;});c.update('none');}
    pushChart(charts.score,[h.score]);
    pushChart(charts.vib,[h.vibration.map(v=>Math.min(v*100,100))]);
    pushChart(charts.cpu,[h.cpu,h.memory]);
  }
}

function initCharts(){
  const base={responsive:true,maintainAspectRatio:false,animation:{duration:300},plugins:{legend:{display:false}},scales:{x:{display:false},y:{display:true,min:0,max:100,grid:{color:'#0d1825'},ticks:{color:'#2a3f55',font:{size:9},maxTicksLimit:4}}}};
  charts.score=new Chart(document.getElementById('c-score'),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#ff2d55',borderWidth:2,pointRadius:0,fill:true,backgroundColor:'#ff2d5518',tension:.4}]},options:base});
  charts.vib=new Chart(document.getElementById('c-vib'),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#00e5ff',borderWidth:2,pointRadius:0,fill:true,backgroundColor:'#00e5ff18',tension:.4}]},options:{...base,scales:{...base.scales,y:{...base.scales.y,max:undefined,min:undefined}}}});
  charts.cpu=new Chart(document.getElementById('c-cpu'),{type:'line',data:{labels:[],datasets:[{label:'CPU',data:[],borderColor:'#00e5ff',borderWidth:2,pointRadius:0,fill:false,tension:.4},{label:'Memory',data:[],borderColor:'#a78bfa',borderWidth:2,pointRadius:0,fill:false,tension:.4}]},options:{...base,plugins:{legend:{display:true,labels:{color:'#5a7a99',font:{size:10},boxWidth:10}}}}});
}

async function poll(){
  try{const r=await fetch('/data');const d=await r.json();if(d&&!d.error)render(d);}catch(e){}
}
setInterval(poll,2000);

let autoInterval=setInterval(async()=>{
  if(mode!=='live')return;
  const mem=performance.memory?Math.round(performance.memory.usedJSHeapSize/performance.memory.jsHeapSizeLimit*100):50;
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vibration:0.05,cpu:50,memory:mem,battery:batteryLevel>=0?batteryLevel:80,temp:38,disk:50,device:navigator.platform||'Desktop'})
  }).catch(()=>{});
  if(r){const d=await r.json();render(d);}
},4000);

function startPhoneSensors(){
  function activate(){
    phoneActive=true;document.getElementById('phone-prompt').style.display='none';
    let st=null;
    window.addEventListener('devicemotion',e=>{
      const a=e.accelerationIncludingGravity;if(!a)return;
      const vib=Math.max(0,Math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z)-9.81);
      clearTimeout(st);st=setTimeout(()=>{
        fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({vibration:parseFloat(vib.toFixed(4)),battery:batteryLevel,cpu:50,memory:60,temp:35,disk:50,device:'Phone'})
        }).then(r=>r.json()).then(render).catch(()=>{});
      },300);
    });
  }
  if(typeof DeviceMotionEvent!=='undefined'&&typeof DeviceMotionEvent.requestPermission==='function'){
    DeviceMotionEvent.requestPermission().then(p=>{if(p==='granted')activate();}).catch(()=>{});
  }else{activate();}
}

async function submitIndustrial(){
  const d={vibration:+document.getElementById('ind-vib').value,cpu:+document.getElementById('ind-cpu').value,memory:+document.getElementById('ind-mem').value,battery:+document.getElementById('ind-bat').value,temp:+document.getElementById('ind-temp').value,disk:+document.getElementById('ind-disk').value,device:'Industrial'};
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  render(await r.json());setMode('live');
}

async function submitCustom(){
  const d={vibration:+document.getElementById('cv-in').value,cpu:+document.getElementById('cc-in').value,memory:+document.getElementById('cm-in').value,battery:+document.getElementById('cb-in').value,temp:+document.getElementById('ct-in').value,disk:+document.getElementById('cd-in').value,device:'Custom'};
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  render(await r.json());setMode('live');
}

initCharts();
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
