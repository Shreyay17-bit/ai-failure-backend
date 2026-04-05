import os, time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression

app = FastAPI()

# ── ML MODELS ──────────────────────────────────────────────────
iforest = IsolationForest(contamination=0.05, random_state=42)
rf_clf  = RandomForestClassifier(n_estimators=100, random_state=42)
lr_bat  = LinearRegression()

rng = np.random.default_rng(42)
N   = 500

# Training data
X_normal = np.column_stack([
    rng.uniform(0.0, 0.3,  N),   # vibration
    rng.uniform(10,  70,   N),   # cpu
    rng.uniform(20,  75,   N),   # ram
    rng.uniform(40,  100,  N),   # battery
    rng.uniform(30,  65,   N),   # temp
    rng.uniform(10,  80,   N),   # disk
])
iforest.fit(X_normal)

X_clf = np.vstack([X_normal,
    np.column_stack([rng.uniform(0.8,2.0,50), rng.uniform(10,70,50), rng.uniform(20,75,50), rng.uniform(40,100,50), rng.uniform(30,65,50), rng.uniform(10,80,50)]),
    np.column_stack([rng.uniform(0.0,0.3,50), rng.uniform(10,70,50), rng.uniform(20,75,50), rng.uniform(40,100,50), rng.uniform(70,100,50), rng.uniform(10,80,50)]),
    np.column_stack([rng.uniform(0.4,0.8,50), rng.uniform(10,70,50), rng.uniform(20,75,50), rng.uniform(40,100,50), rng.uniform(30,65,50), rng.uniform(10,80,50)]),
    np.column_stack([rng.uniform(0.0,0.3,50), rng.uniform(80,100,50), rng.uniform(85,100,50), rng.uniform(40,100,50), rng.uniform(30,65,50), rng.uniform(10,80,50)]),
    np.column_stack([rng.uniform(0.0,0.3,50), rng.uniform(10,70,50), rng.uniform(20,75,50), rng.uniform(5,20,50), rng.uniform(30,65,50), rng.uniform(10,80,50)]),
])
y_clf = np.array([0]*N + [1]*50 + [2]*50 + [3]*50 + [4]*50 + [5]*50)
rf_clf.fit(X_clf, y_clf)

t = np.arange(N).reshape(-1, 1)
bat_vals = 100 - (t.flatten() * 0.05) + rng.normal(0, 2, N)
lr_bat.fit(t, bat_vals)

FAULT_NAMES = ["NORMAL","BEARING_FAULT","OVERHEATING","MISALIGNMENT","WEAR_OVERLOAD","LOW_POWER"]

all_devices: dict = {}
history: list    = []

# ── PREDICT ENDPOINT ──────────────────────────────────────────
@app.post("/predict")
async def predict(data: dict):
    global history
    vib   = float(data.get("vibration", 0.1))
    cpu   = float(data.get("cpu_load",  30))
    ram   = float(data.get("memory",    40))
    bat   = float(data.get("battery",   90))
    temp  = float(data.get("temperature", 45))
    disk  = float(data.get("disk_usage",  50))
    dev   = str(data.get("device_type", "Unknown"))

    vec   = [[vib, cpu, ram, bat, temp, disk]]
    score = iforest.score_samples(vec)[0]
    anom  = max(0, min(100, int((1 - (score + 0.5)) * 100)))
    fault = FAULT_NAMES[int(rf_clf.predict(vec)[0])]
    proba = float(rf_clf.predict_proba(vec)[0].max()) * 100
    slope = float(lr_bat.coef_[0])
    rul   = max(0, int(bat / max(0.01, abs(slope)))) if slope < 0 else 999

    if anom >= 70:
        risk, status = "CRITICAL", "CRITICAL FAILURE"
        recs = ["HALT OPERATIONS — immediate inspection required",
                f"Fault signature: {fault} detected at {proba:.0f}% confidence",
                "Check bearing assemblies and lubrication channels",
                "Isolate affected subsystem before restart"]
    elif anom >= 40:
        risk, status = "WARNING", "MAINTENANCE ADVISORY"
        recs = [f"Schedule maintenance within 24h — {fault} pattern emerging",
                "Monitor vibration trend — currently elevated",
                f"Battery RUL estimate: {rul}h at current drain rate",
                "Inspect cooling system — thermal margins shrinking"]
    else:
        risk, status = "NOMINAL", "ALL SYSTEMS NOMINAL"
        recs = ["Continue standard monitoring intervals",
                f"Next scheduled check in {min(168, rul)}h",
                "No anomalies detected by Isolation Forest",
                "System operating within design envelope"]

    trend = "RISING" if slope > 0.01 else "FALLING" if slope < -0.01 else "STABLE"
    result = dict(vibration=round(vib,3), cpu_load=cpu, memory=ram,
                  battery=bat, temperature=temp, disk_usage=disk,
                  device=dev, anomaly_score=anom, fault=fault,
                  fault_confidence=round(proba,1), risk=risk, status=status,
                  rul_hours=rul, trend=trend, recommendations=recs,
                  z_score=round((vib-0.15)/0.1, 2))

    all_devices[dev] = result
    history.append(anom)
    if len(history) > 30:
        history.pop(0)
    return result

@app.get("/data")
async def data():
    latest = list(all_devices.values())[-1] if all_devices else {}
    return {**latest, "history": history, "all_devices": all_devices}

@app.get("/")
async def root():
    return {"status": "online", "version": "2.0.0"}

# ── MONITOR DASHBOARD ─────────────────────────────────────────
@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEXUS — Predictive Maintenance</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:      #0c0b08;
  --surface: #131210;
  --border:  #2a2720;
  --amber:   #e8a020;
  --amber2:  #c47a10;
  --amber-lo:#3a2800;
  --red:     #d63c2a;
  --red-lo:  #3a0e08;
  --green:   #4a8c5c;
  --green-lo:#0d2218;
  --text:    #d4c8a8;
  --muted:   #6b6048;
  --mono:    'Share Tech Mono', monospace;
  --display: 'Bebas Neue', cursive;
  --body:    'DM Sans', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { font-size: 14px; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--body);
  min-height: 100vh;
  overflow-x: hidden;
}

/* Scanline overlay */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.06) 2px,
    rgba(0,0,0,0.06) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── HEADER ── */
header {
  display: flex;
  align-items: stretch;
  border-bottom: 2px solid var(--border);
  background: var(--surface);
}

.header-brand {
  padding: 18px 32px;
  border-right: 2px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.brand-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 4px;
  color: var(--muted);
  text-transform: uppercase;
}

.brand-name {
  font-family: var(--display);
  font-size: 38px;
  color: var(--amber);
  line-height: 1;
  letter-spacing: 3px;
}

.header-status {
  flex: 1;
  display: flex;
  align-items: center;
  padding: 0 32px;
  gap: 32px;
}

.status-pill {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 2px;
  color: var(--muted);
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74,140,92,0.4); }
  50%       { opacity: 0.7; box-shadow: 0 0 0 4px rgba(74,140,92,0); }
}

.header-time {
  margin-left: auto;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  text-align: right;
  line-height: 1.8;
}

/* ── LAYOUT ── */
.shell {
  display: grid;
  grid-template-columns: 320px 1fr;
  min-height: calc(100vh - 80px);
}

.left-panel {
  border-right: 2px solid var(--border);
  display: flex;
  flex-direction: column;
}

/* ── SECTION HEADERS ── */
.sec-hdr {
  padding: 12px 20px;
  border-bottom: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 4px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--surface);
  text-transform: uppercase;
}

.sec-hdr::before {
  content: '';
  display: block;
  width: 3px;
  height: 10px;
  background: var(--amber);
}

/* ── RISK METER ── */
.risk-block {
  padding: 28px 24px 24px;
  border-bottom: 1px solid var(--border);
}

.risk-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 3px;
  color: var(--muted);
  margin-bottom: 16px;
}

.risk-arc-wrap {
  position: relative;
  width: 200px;
  height: 120px;
  margin: 0 auto 16px;
}

.risk-arc-wrap svg { width: 100%; height: 100%; overflow: visible; }

.risk-score-big {
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  text-align: center;
}

.risk-num {
  font-family: var(--display);
  font-size: 56px;
  line-height: 1;
  color: var(--amber);
  transition: color 0.5s;
}

.risk-unit {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  color: var(--muted);
}

.risk-badge {
  display: block;
  margin: 14px auto 0;
  width: fit-content;
  padding: 6px 18px;
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 3px;
  border: 1px solid var(--amber);
  color: var(--amber);
  background: var(--amber-lo);
  transition: all 0.4s;
}

/* ── MINI GAUGES ── */
.gauges-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: var(--border);
  border-top: 1px solid var(--border);
}

.gauge-cell {
  background: var(--bg);
  padding: 16px;
}

.gc-label {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 2px;
  color: var(--muted);
  margin-bottom: 8px;
}

.gc-value {
  font-family: var(--display);
  font-size: 28px;
  color: var(--text);
  line-height: 1;
  margin-bottom: 8px;
}

.gc-bar {
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}

.gc-fill {
  height: 100%;
  background: var(--amber);
  transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* ── FAULT CARD ── */
.fault-block {
  padding: 20px 24px;
  border-bottom: 1px solid var(--border);
}

.fault-name {
  font-family: var(--display);
  font-size: 22px;
  letter-spacing: 2px;
  color: var(--text);
  margin-bottom: 4px;
}

.fault-conf {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
}

.fault-conf span { color: var(--amber); }

/* ── MODE SWITCHER ── */
.mode-tabs {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: var(--border);
  border-bottom: 1px solid var(--border);
}

.mode-tab {
  padding: 12px 8px;
  background: var(--bg);
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 1px;
  color: var(--muted);
  border: none;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
  text-transform: uppercase;
}

.mode-tab:hover { background: var(--surface); color: var(--text); }
.mode-tab.active { background: var(--amber-lo); color: var(--amber); }

/* ── FORM FIELDS ── */
.input-panel {
  padding: 20px;
  border-bottom: 1px solid var(--border);
  display: none;
}

.input-panel.active { display: block; }

.field-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 10px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.field label {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 2px;
  color: var(--muted);
  text-transform: uppercase;
}

.field input {
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  padding: 8px 10px;
  outline: none;
  transition: border-color 0.2s;
  width: 100%;
}

.field input:focus { border-color: var(--amber); }

.scan-btn {
  width: 100%;
  padding: 12px;
  background: var(--amber);
  border: none;
  color: #0c0b08;
  font-family: var(--display);
  font-size: 18px;
  letter-spacing: 3px;
  cursor: pointer;
  margin-top: 8px;
  transition: background 0.2s, transform 0.1s;
}

.scan-btn:hover  { background: var(--amber2); }
.scan-btn:active { transform: scale(0.99); }

/* ── LIVE AUTO PANEL ── */
.live-panel {
  padding: 20px;
  border-bottom: 1px solid var(--border);
  display: none;
}

.live-panel.active { display: block; }

.live-row {
  display: flex;
  justify-content: space-between;
  font-family: var(--mono);
  font-size: 10px;
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
  color: var(--muted);
}

.live-row span:last-child { color: var(--text); }

.live-ping {
  margin-top: 12px;
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  color: var(--muted);
}

.live-ping span {
  display: inline-block;
  animation: blink-txt 1s step-start infinite;
  color: var(--amber);
}

@keyframes blink-txt { 50% { opacity: 0; } }

/* ── RIGHT PANEL ── */
.right-panel {
  display: flex;
  flex-direction: column;
}

/* ── RECOMMENDATION STRIP ── */
.rec-strip {
  border-bottom: 1px solid var(--border);
  padding: 20px 32px;
  background: var(--surface);
}

.rec-title {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 3px;
  color: var(--muted);
  margin-bottom: 14px;
}

.rec-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.rec-list li {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  font-size: 12px;
  line-height: 1.5;
  color: var(--text);
}

.rec-num {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--amber);
  min-width: 20px;
  padding-top: 1px;
}

/* ── CHARTS ROW ── */
.charts-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1px;
  background: var(--border);
  flex: 1;
}

.chart-block {
  background: var(--bg);
  padding: 20px;
  min-height: 180px;
  display: flex;
  flex-direction: column;
}

.chart-title {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 3px;
  color: var(--muted);
  margin-bottom: 16px;
  text-transform: uppercase;
}

.spark-area {
  flex: 1;
  display: flex;
  align-items: flex-end;
  gap: 3px;
}

.spark-col {
  flex: 1;
  min-width: 4px;
  border-radius: 1px 1px 0 0;
  transition: height 0.4s ease;
}

/* ── DEVICE TABS ── */
.dev-tabs {
  display: flex;
  gap: 1px;
  background: var(--border);
  padding: 0;
  border-bottom: 1px solid var(--border);
}

.dev-tab {
  padding: 8px 20px;
  background: var(--bg);
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  color: var(--muted);
  border: none;
  cursor: pointer;
  transition: all 0.15s;
}

.dev-tab:hover  { background: var(--surface); color: var(--text); }
.dev-tab.active { background: var(--surface); color: var(--amber); border-bottom: 2px solid var(--amber); }

/* ── METADATA ROW ── */
.meta-row {
  display: flex;
  border-top: 1px solid var(--border);
  background: var(--surface);
}

.meta-cell {
  flex: 1;
  padding: 12px 24px;
  border-right: 1px solid var(--border);
}

.meta-cell:last-child { border-right: none; }

.meta-k {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 2px;
  color: var(--muted);
  margin-bottom: 4px;
}

.meta-v {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--text);
}

/* ── STATUS STATES ── */
.nominal  .risk-num { color: var(--green); }
.nominal  .risk-badge { border-color: var(--green); color: var(--green); background: var(--green-lo); }

.critical .risk-num { color: var(--red); }
.critical .risk-badge { border-color: var(--red); color: var(--red); background: var(--red-lo); animation: flash-badge 0.8s ease-in-out infinite; }

@keyframes flash-badge { 50% { opacity: 0.5; } }

/* ── EMPTY STATE ── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  padding: 60px 32px;
  gap: 16px;
  color: var(--muted);
}

.empty-state .big-label {
  font-family: var(--display);
  font-size: 64px;
  color: var(--border);
  letter-spacing: 6px;
}

.empty-state p {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 2px;
  text-align: center;
  line-height: 2;
}

/* ── RESPONSIVE ── */
@media (max-width: 900px) {
  .shell { grid-template-columns: 1fr; }
  .right-panel { border-top: 2px solid var(--border); }
  .charts-row { grid-template-columns: 1fr; }
  .meta-row { flex-wrap: wrap; }
}
</style>
</head>
<body>

<header>
  <div class="header-brand">
    <span class="brand-label">System</span>
    <span class="brand-name">NEXUS</span>
  </div>
  <div class="header-status">
    <div class="status-pill">
      <span class="dot" id="conn-dot"></span>
      <span id="conn-label">CONNECTING</span>
    </div>
    <div class="status-pill" style="gap:6px">
      <span style="color:var(--muted)">DEVICES —</span>
      <span id="dev-count" style="color:var(--amber)">0</span>
    </div>
    <div class="header-time">
      <div id="clock">--:--:--</div>
      <div id="date-line" style="opacity:0.4"></div>
    </div>
  </div>
</header>

<div class="shell">

  <!-- LEFT PANEL -->
  <div class="left-panel">

    <div class="sec-hdr">Anomaly Score</div>
    <div class="risk-block" id="risk-state">
      <div class="risk-label">ISOLATION FOREST · PREDICTIVE INDEX</div>
      <div class="risk-arc-wrap">
        <svg viewBox="0 0 200 110" fill="none" xmlns="http://www.w3.org/2000/svg">
          <!-- Track -->
          <path d="M 20 100 A 80 80 0 0 1 180 100" stroke="#2a2720" stroke-width="8" stroke-linecap="round"/>
          <!-- Fill -->
          <path id="arc-fill" d="M 20 100 A 80 80 0 0 1 180 100" stroke="#e8a020"
                stroke-width="8" stroke-linecap="round"
                stroke-dasharray="251.2" stroke-dashoffset="251.2"
                style="transition: stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1), stroke 0.5s"/>
          <!-- Ticks -->
          <g stroke="#2a2720" stroke-width="1">
            <line x1="20"  y1="100" x2="10"  y2="100"/>
            <line x1="180" y1="100" x2="190" y2="100"/>
            <line x1="100" y1="20"  x2="100" y2="8"/>
            <line x1="57"  y1="43"  x2="50"  y2="36"/>
            <line x1="143" y1="43"  x2="150" y2="36"/>
          </g>
          <text x="16"  y="118" fill="#6b6048" font-family="'Share Tech Mono'" font-size="9">0</text>
          <text x="176" y="118" fill="#6b6048" font-family="'Share Tech Mono'" font-size="9">100</text>
        </svg>
        <div class="risk-score-big">
          <div class="risk-num" id="risk-num">--</div>
          <div class="risk-unit">/ 100</div>
        </div>
      </div>
      <span class="risk-badge" id="risk-badge">AWAITING DATA</span>
    </div>

    <!-- mini gauges -->
    <div class="gauges-grid">
      <div class="gauge-cell">
        <div class="gc-label">BATTERY</div>
        <div class="gc-value" id="g-bat">--%</div>
        <div class="gc-bar"><div class="gc-fill" id="b-bat" style="width:0%"></div></div>
      </div>
      <div class="gauge-cell">
        <div class="gc-label">CPU LOAD</div>
        <div class="gc-value" id="g-cpu">--%</div>
        <div class="gc-bar"><div class="gc-fill" id="b-cpu" style="width:0%"></div></div>
      </div>
      <div class="gauge-cell">
        <div class="gc-label">MEMORY</div>
        <div class="gc-value" id="g-ram">--%</div>
        <div class="gc-bar"><div class="gc-fill" id="b-ram" style="width:0%"></div></div>
      </div>
      <div class="gauge-cell">
        <div class="gc-label">VIBRATION</div>
        <div class="gc-value" id="g-vib">-.---</div>
        <div class="gc-bar"><div class="gc-fill" id="b-vib" style="width:0%"></div></div>
      </div>
    </div>

    <!-- fault -->
    <div class="sec-hdr">Fault Classification</div>
    <div class="fault-block">
      <div class="fault-name" id="fault-name">AWAITING SCAN</div>
      <div class="fault-conf">Random Forest confidence: <span id="fault-conf">--%</span></div>
    </div>

    <!-- mode tabs -->
    <div class="sec-hdr">Input Mode</div>
    <div class="mode-tabs">
      <button class="mode-tab active" onclick="setMode('live')">⬤ Live</button>
      <button class="mode-tab"        onclick="setMode('industrial')">⬡ Industrial</button>
      <button class="mode-tab"        onclick="setMode('custom')">◧ Custom</button>
    </div>

    <!-- live panel -->
    <div class="live-panel active" id="panel-live">
      <div class="live-row"><span>DEVICE</span>     <span id="lv-plat">detecting…</span></div>
      <div class="live-row"><span>BATTERY</span>    <span id="lv-bat">detecting…</span></div>
      <div class="live-row"><span>VIBRATION</span>  <span id="lv-vib">0.000 m/s²</span></div>
      <div class="live-row"><span>INTERVAL</span>   <span>4 000 ms</span></div>
      <div id="ios-btn-wrap" style="display:none;margin-top:12px">
        <button class="scan-btn" style="font-size:13px;letter-spacing:2px;height:auto;padding:10px" onclick="requestIOSMotion()">
          ▶ GRANT MOTION ACCESS (iOS)
        </button>
        <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:6px;line-height:1.6">
          iOS blocks accelerometer until<br>user grants permission via button.
        </div>
      </div>
      <div class="live-ping">STATUS <span>▌</span> SCANNING</div>
    </div>

    <!-- industrial panel -->
    <div class="input-panel" id="panel-industrial">
      <div class="field-row">
        <div class="field"><label>Vibration m/s²</label><input id="i-vib"  type="number" step="0.01" value="0.15" placeholder="0.15"></div>
        <div class="field"><label>Temperature °C</label><input id="i-temp" type="number" step="1"    value="45"   placeholder="45"></div>
      </div>
      <div class="field-row">
        <div class="field"><label>Battery / Power %</label><input id="i-bat"  type="number" step="1" value="85"  placeholder="85"></div>
        <div class="field"><label>CPU / Node Load %</label><input id="i-cpu"  type="number" step="1" value="30"  placeholder="30"></div>
      </div>
      <div class="field-row">
        <div class="field"><label>Memory / RAM %</label><input id="i-ram"  type="number" step="1" value="50"  placeholder="50"></div>
        <div class="field"><label>Disk / Storage %</label><input id="i-disk" type="number" step="1" value="40"  placeholder="40"></div>
      </div>
      <button class="scan-btn" onclick="sendIndustrial()">RUN ANALYSIS</button>
    </div>

    <!-- custom panel -->
    <div class="input-panel" id="panel-custom">
      <div class="field-row">
        <div class="field"><label>Vibration</label><input id="c-vib"  type="number" step="0.01" value="0.5"></div>
        <div class="field"><label>Temperature</label><input id="c-temp" type="number" step="1"    value="70"></div>
      </div>
      <div class="field-row">
        <div class="field"><label>Battery %</label><input id="c-bat"  type="number" step="1" value="20"></div>
        <div class="field"><label>CPU %</label><input id="c-cpu"  type="number" step="1" value="90"></div>
      </div>
      <div class="field-row">
        <div class="field"><label>Memory %</label><input id="c-ram"  type="number" step="1" value="88"></div>
        <div class="field"><label>Disk %</label><input id="c-disk" type="number" step="1" value="95"></div>
      </div>
      <button class="scan-btn" onclick="sendCustom()">RUN ANALYSIS</button>
    </div>

  </div><!-- /left-panel -->

  <!-- RIGHT PANEL -->
  <div class="right-panel">

    <div class="dev-tabs" id="dev-tabs"></div>

    <div id="main-content">
      <div class="empty-state" id="empty">
        <div class="big-label">IDLE</div>
        <p>NO PROBE DATA RECEIVED<br>SELECT AN INPUT MODE AND RUN A SCAN</p>
      </div>
    </div>

  </div>

</div><!-- /shell -->

<script>
// ── STATE ──────────────────────────────────────────────────────
let selectedDev = null;
let currentMode = 'live';
let liveData    = {};
let liveInterval;
let motionEnabled = false;
let lastVib = 0;

// ── CLOCK ─────────────────────────────────────────────────────
function tickClock() {
  const n = new Date();
  document.getElementById('clock').textContent = n.toLocaleTimeString('en-GB');
  document.getElementById('date-line').textContent =
    n.toLocaleDateString('en-GB', {day:'2-digit', month:'short', year:'numeric'}).toUpperCase();
}
setInterval(tickClock, 1000);
tickClock();

// ── MODE ──────────────────────────────────────────────────────
function setMode(m) {
  currentMode = m;
  document.querySelectorAll('.mode-tab').forEach((t,i) => {
    t.classList.toggle('active', ['live','industrial','custom'][i] === m);
  });
  ['live','industrial','custom'].forEach(id => {
    const lp = document.getElementById('panel-' + id);
    const ip = document.getElementById('panel-' + id);
    if (lp) lp.classList.toggle('active', id === m);
  });
  // Re-show correct panel
  document.getElementById('panel-live').classList.toggle('active', m === 'live');
  document.getElementById('panel-industrial').classList.toggle('active', m === 'industrial');
  document.getElementById('panel-custom').classList.toggle('active', m === 'custom');

  if (m === 'live') startLiveSensors();
  else clearInterval(liveInterval);
}

// ── DEVICE DETECTION ─────────────────────────────────────────
function detectDevice() {
  const ua = navigator.userAgent;
  if (/iPhone/i.test(ua))               return 'iPhone';
  if (/iPad/i.test(ua))                 return 'iPad';
  if (/Android/i.test(ua))              return 'Android';
  if (/Macintosh|Mac OS X/i.test(ua))   return 'macOS';
  if (/Windows/i.test(ua))              return 'Windows';
  if (/Linux/i.test(ua))                return 'Linux';
  return 'Unknown';
}

function isIOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

// ── SENSORS ───────────────────────────────────────────────────
let batteryLevel = null;   // null = not yet known
let batteryAvailable = false;

async function initBattery() {
  if (isIOS()) {
    // iOS Safari blocks Battery API entirely — mark as unavailable
    batteryAvailable = false;
    batteryLevel = null;
    return;
  }
  try {
    if (navigator.getBattery) {
      const b = await navigator.getBattery();
      batteryLevel = Math.round(b.level * 100);
      batteryAvailable = true;
      // Keep updating if it changes
      b.addEventListener('levelchange', () => {
        batteryLevel = Math.round(b.level * 100);
      });
    }
  } catch(e) {
    batteryAvailable = false;
    batteryLevel = null;
  }
}

function enableMotion() {
  if (isIOS()) {
    // iOS needs explicit user gesture — handled by the Grant Access button
    return;
  }
  if (typeof DeviceMotionEvent !== 'undefined' && DeviceMotionEvent.requestPermission) {
    DeviceMotionEvent.requestPermission().then(s => {
      if (s === 'granted') listenMotion();
    }).catch(() => {});
  } else {
    listenMotion();
  }
}

function listenMotion() {
  motionEnabled = true;
  window.addEventListener('devicemotion', e => {
    const a = e.acceleration || e.accelerationIncludingGravity;
    if (a && (a.x !== null)) {
      lastVib = Math.round(((Math.abs(a.x||0) + Math.abs(a.y||0) + Math.abs(a.z||0)) / 30) * 1000) / 1000;
    }
  });
}

function startLiveSensors() {
  enableMotion();
  clearInterval(liveInterval);
  doLiveScan();
  liveInterval = setInterval(doLiveScan, 4000);
}

async function doLiveScan() {
  const devName = detectDevice();

  // Battery display
  let batDisplay = batteryAvailable && batteryLevel !== null ? batteryLevel + '%' : (isIOS() ? 'N/A (iOS)' : 'detecting…');
  // Use 85 as neutral fallback for ML only when unavailable
  let batForML   = (batteryAvailable && batteryLevel !== null) ? batteryLevel : 85;

  document.getElementById('lv-bat').textContent  = batDisplay;
  document.getElementById('lv-vib').textContent  = lastVib.toFixed(3) + ' m/s²';
  document.getElementById('lv-plat').textContent = devName;

  // Dynamic memory estimate via performance API
  let memEst = 50;
  if (navigator.deviceMemory) {
    // deviceMemory is in GB (1,2,4,8), estimate % as rough heuristic
    memEst = Math.max(20, Math.min(85, Math.round(100 - navigator.deviceMemory * 8)));
  }

  await sendToServer({
    device_type: devName,
    battery:     batForML,
    vibration:   lastVib,
    // CPU load can't be read from browser — use a stable estimate
    cpu_load:    Math.round(20 + lastVib * 40),
    memory:      memEst,
    temperature: Math.round(32 + (100 - batForML) * 0.25 + lastVib * 8),
    disk_usage:  50,
    battery_available: batteryAvailable,
  });
}

async function sendIndustrial() {
  await sendToServer({
    device_type:  'Industrial',
    vibration:    parseFloat(document.getElementById('i-vib').value)  || 0.15,
    temperature:  parseFloat(document.getElementById('i-temp').value) || 45,
    battery:      parseFloat(document.getElementById('i-bat').value)  || 85,
    cpu_load:     parseFloat(document.getElementById('i-cpu').value)  || 30,
    memory:       parseFloat(document.getElementById('i-ram').value)  || 50,
    disk_usage:   parseFloat(document.getElementById('i-disk').value) || 40,
  });
}

async function sendCustom() {
  await sendToServer({
    device_type:  'Custom',
    vibration:    parseFloat(document.getElementById('c-vib').value)  || 0.5,
    temperature:  parseFloat(document.getElementById('c-temp').value) || 70,
    battery:      parseFloat(document.getElementById('c-bat').value)  || 20,
    cpu_load:     parseFloat(document.getElementById('c-cpu').value)  || 90,
    memory:       parseFloat(document.getElementById('c-ram').value)  || 88,
    disk_usage:   parseFloat(document.getElementById('c-disk').value) || 95,
  });
}

async function sendToServer(payload) {
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    updateDashboard(payload.device_type, d);
    refreshData();
  } catch(e) {
    document.getElementById('conn-label').textContent = 'ERROR';
  }
}

// ── RENDER ────────────────────────────────────────────────────
function updateDashboard(devKey, d) {
  // Arc gauge
  const pct   = Math.min(100, d.anomaly_score || 0);
  const total = 251.2;
  const fill  = total - (pct / 100) * total;
  const arc   = document.getElementById('arc-fill');
  arc.style.strokeDashoffset = fill;
  arc.style.stroke = pct >= 70 ? '#d63c2a' : pct >= 40 ? '#c47a10' : '#4a8c5c';

  // Number
  document.getElementById('risk-num').textContent = Math.round(pct);

  // Badge + state class
  const state = pct >= 70 ? 'critical' : pct >= 40 ? 'warning' : 'nominal';
  const badge = pct >= 70 ? 'CRITICAL' : pct >= 40 ? 'WARNING' : 'NOMINAL';
  const rs    = document.getElementById('risk-state');
  rs.className = state;
  document.getElementById('risk-badge').textContent = badge;

  // Mini gauges
  setGauge('bat',  d.battery,    '%');
  setGauge('cpu',  d.cpu_load,   '%');
  setGauge('ram',  d.memory,     '%');
  setVibGauge(d.vibration);

  // Fault
  document.getElementById('fault-name').textContent = (d.fault || 'NORMAL').replace(/_/g,' ');
  document.getElementById('fault-conf').textContent = (d.fault_confidence || 0).toFixed(1) + '%';
}

function setGauge(id, val, unit) {
  // Battery special handling — on iOS it's unavailable
  if (id === 'bat' && isIOS() && !batteryAvailable) {
    document.getElementById('g-bat').textContent = 'N/A';
    document.getElementById('b-bat').style.width = '0%';
    return;
  }
  document.getElementById('g-' + id).textContent = Math.round(val) + unit;
  document.getElementById('b-' + id).style.width  = Math.min(100, val) + '%';
  const fill = document.getElementById('b-' + id);
  fill.style.background = val > 90 ? '#d63c2a' : val > 70 ? '#c47a10' : '#4a8c5c';
  if (id === 'bat') fill.style.background = val < 20 ? '#d63c2a' : val < 40 ? '#c47a10' : '#4a8c5c';
}

function setVibGauge(v) {
  document.getElementById('g-vib').textContent = (v || 0).toFixed(3);
  const pct = Math.min(100, (v / 2) * 100);
  document.getElementById('b-vib').style.width      = pct + '%';
  document.getElementById('b-vib').style.background = v > 0.8 ? '#d63c2a' : v > 0.5 ? '#c47a10' : '#4a8c5c';
}

async function refreshData() {
  try {
    const r = await fetch('/data');
    const d = await r.json();
    const devs = d.all_devices || {};
    const names = Object.keys(devs);

    document.getElementById('dev-count').textContent = names.length;
    document.getElementById('conn-label').textContent = 'ONLINE';

    // Device tabs
    const tabs = names.map(n =>
      `<button class="dev-tab ${n===selectedDev?'active':''}" onclick="selectDev('${n}')">${n.toUpperCase()}</button>`
    ).join('');
    document.getElementById('dev-tabs').innerHTML = tabs;

    if (!names.length) { document.getElementById('empty').style.display = 'flex'; return; }
    document.getElementById('empty').style.display = 'none';

    // If selectedDev not in data yet (page just loaded), pick it or fall back to last
    if (!devs[selectedDev]) {
      selectedDev = names.includes(selectedDev) ? selectedDev : names[names.length-1];
    }
    const current = devs[selectedDev] || {};

    renderRightPanel(current, d.history || []);
  } catch(e) {}
}

function selectDev(name) {
  selectedDev = name;
  refreshData();
}

function renderRightPanel(d, hist) {
  const recs = (d.recommendations || []).map((r, i) =>
    `<li><span class="rec-num">${String(i+1).padStart(2,'0')}.</span>${r}</li>`
  ).join('');

  const sparkBars = hist.map(v => {
    const h   = Math.max(4, v);
    const col = v >= 70 ? '#d63c2a' : v >= 40 ? '#c47a10' : '#4a8c5c';
    return `<div class="spark-col" style="height:${h}%;background:${col}"></div>`;
  }).join('');

  // CPU mini chart (simulated from history)
  const cpuBars = hist.map((v, i) => {
    const c = Math.max(4, Math.min(100, 10 + v * 0.5 + (i % 3) * 5));
    return `<div class="spark-col" style="height:${c}%;background:#e8a020"></div>`;
  }).join('');

  // Battery trend
  const batBars = hist.map((v, i) => {
    const b = Math.max(4, Math.min(100, (d.battery || 80) - i * 0.5));
    const col = b < 20 ? '#d63c2a' : b < 40 ? '#c47a10' : '#4a8c5c';
    return `<div class="spark-col" style="height:${b}%;background:${col}"></div>`;
  }).join('');

  const mc = document.getElementById('main-content');
  mc.innerHTML = `
    <div class="rec-strip">
      <div class="rec-title">ML RECOMMENDATIONS — ${(d.status || 'NO STATUS')}</div>
      <ul class="rec-list">${recs}</ul>
    </div>
    <div class="charts-row">
      <div class="chart-block">
        <div class="chart-title">Anomaly Score History</div>
        <div class="spark-area">${sparkBars || '<span style="color:var(--muted);font-size:10px">NO HISTORY</span>'}</div>
      </div>
      <div class="chart-block">
        <div class="chart-title">CPU Load Correlation</div>
        <div class="spark-area">${cpuBars || '<span style="color:var(--muted);font-size:10px">NO HISTORY</span>'}</div>
      </div>
      <div class="chart-block">
        <div class="chart-title">Battery Life Projection</div>
        <div class="spark-area">${batBars || '<span style="color:var(--muted);font-size:10px">NO HISTORY</span>'}</div>
      </div>
    </div>
    <div class="meta-row">
      <div class="meta-cell"><div class="meta-k">Z-SCORE</div><div class="meta-v">${d.z_score ?? '--'}</div></div>
      <div class="meta-cell"><div class="meta-k">TREND</div><div class="meta-v">${d.trend ?? '--'}</div></div>
      <div class="meta-cell"><div class="meta-k">RUL ESTIMATE</div><div class="meta-v">${d.rul_hours === 999 ? '∞' : (d.rul_hours ?? '--')}h</div></div>
      <div class="meta-cell"><div class="meta-k">TEMPERATURE</div><div class="meta-v">${d.temperature ?? '--'}°C</div></div>
      <div class="meta-cell"><div class="meta-k">DISK USAGE</div><div class="meta-v">${d.disk_usage ?? '--'}%</div></div>
    </div>`;
}

// ── iOS MOTION REQUEST ────────────────────────────────────────
function requestIOSMotion() {
  if (typeof DeviceMotionEvent !== 'undefined' && DeviceMotionEvent.requestPermission) {
    DeviceMotionEvent.requestPermission().then(s => {
      if (s === 'granted') {
        listenMotion();
        document.getElementById('ios-btn-wrap').style.display = 'none';
      }
    }).catch(e => console.warn('Motion permission denied', e));
  }
}

// ── BOOT ──────────────────────────────────────────────────────
// Default selectedDev to the device currently opening the page
selectedDev = detectDevice();

// Show iOS grant button if needed
if (isIOS()) {
  document.getElementById('ios-btn-wrap').style.display = 'block';
}

// Init battery API before first scan
initBattery().then(() => {
  startLiveSensors();
});
setInterval(refreshData, 4000);
refreshData();
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
