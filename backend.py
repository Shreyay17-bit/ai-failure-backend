"""
Nexus PM - Predictive Maintenance Backend v3
Handles: Windows probe, Android phone, iPhone, Industrial, Custom inputs
ML: Isolation Forest + Random Forest + Linear Regression + Z-score + FFT
"""
import os, time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
 
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
 
# ── ML MODELS ──────────────────────────────────────────────────────────────────
 
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
rng = np.random.default_rng(42)
baseline = np.column_stack([
    rng.normal(0.02, 0.01, 600),   # vibration  — very low for healthy PC
    rng.normal(45,   20,   600),   # cpu %
    rng.normal(55,   15,   600),   # memory %
    rng.normal(80,   15,   600),   # battery %
    rng.normal(45,    8,   600),   # temp °C
    rng.normal(50,   20,   600),   # disk %
])
iso.fit(baseline)
 
rf = RandomForestClassifier(n_estimators=150, random_state=42)
_faults = ["NORMAL","BEARING_FAULT","MISALIGNMENT","IMBALANCE","OVERHEATING","WEAR"]
_means = {
    "NORMAL":        [0.02, 45, 55, 80, 42, 50],
    "BEARING_FAULT": [0.08, 55, 60, 75, 48, 55],
    "MISALIGNMENT":  [0.15, 70, 65, 65, 52, 60],
    "IMBALANCE":     [0.25, 80, 72, 55, 56, 65],
    "OVERHEATING":   [0.05, 92, 85, 45, 75, 70],
    "WEAR":          [0.12, 65, 68, 60, 58, 80],
}
_Xf, _yf = [], []
for i, f in enumerate(_faults):
    m = _means[f]
    for _ in range(100):
        _Xf.append([rng.normal(m[j], abs(m[j])*0.12 + 0.01) for j in range(6)])
        _yf.append(i)
rf.fit(_Xf, _yf)
 
trend_lr = LinearRegression()
 
# ── HISTORY ────────────────────────────────────────────────────────────────────
hist = {k: [] for k in ["vib","cpu","mem","bat","temp","disk","score","t"]}
MAX = 60
 
def hpush(k, v):
    hist[k].append(v)
    if len(hist[k]) > MAX: hist[k].pop(0)
 
# ── ML ENGINE ──────────────────────────────────────────────────────────────────
def run_ml(vib, cpu, mem, bat, temp, disk, extra):
    t = time.time()
    feat = np.array([[vib, cpu, mem, bat, temp, disk]])
 
    # Isolation Forest → health score
    iso_raw = iso.decision_function(feat)[0]
    health = int(np.clip((iso_raw + 0.3) * 160, 0, 100))
 
    for k, v in zip(["vib","cpu","mem","bat","temp","disk","t"],
                     [vib, cpu, mem, bat, temp, disk, t]):
        hpush(k, v)
 
    # Z-score on vibration
    va = np.array(hist["vib"])
    z = float(abs((vib - va.mean()) / (va.std() + 1e-9))) if len(va) > 2 else 0.0
 
    # Linear Regression trend on vibration
    trend, slope = "STABLE", 0.0
    if len(va) >= 5:
        trend_lr.fit(np.arange(len(va)).reshape(-1,1), va)
        slope = float(trend_lr.coef_[0])
        trend = "RISING" if slope > 0.001 else "FALLING" if slope < -0.001 else "STABLE"
 
    # FFT
    freq = 0.0
    if len(va) >= 10:
        fft = np.abs(np.fft.fft(va))
        freqs = np.fft.fftfreq(len(va))
        freq = float(abs(freqs[np.argmax(fft[1:]) + 1]))
 
    # Random Forest fault
    fi = int(rf.predict(feat)[0])
    fault = _faults[fi]
    fconf = float(rf.predict_proba(feat)[0][fi] * 100)
 
    # Override by rules
    if temp > 75:     fault = "OVERHEATING"
    elif z > 3.5:     fault = "IMBALANCE"
    elif z > 2.5:     fault = "MISALIGNMENT"
    elif disk > 92:   fault = "WEAR"
 
    # RUL
    rul = None
    if len(va) >= 10 and slope > 0:
        rul = round(max(0, (0.8 - vib) / (slope + 1e-9)) * 2 / 60, 1)
 
    # Battery life
    bat_life = "N/A"
    ba = np.array(hist["bat"]); ta = np.array(hist["t"])
    if len(ba) >= 6 and bat >= 0:
        trend_lr.fit(ta.reshape(-1,1), ba)
        dps = float(trend_lr.coef_[0])
        if dps < -1e-6:
            mins = abs(bat / dps) / 60
            bat_life = f"{int(mins)} min" if mins < 120 else f"{mins/60:.1f} hr"
        elif extra.get("is_charging"):
            bat_life = "Charging"
 
    # ── RISK SCORE (properly calibrated) ──────────────────────────────────────
    risk_score = 0
 
    # Isolation Forest component (0-40 pts)
    if_component = max(0, (1 - health/100) * 40)
    risk_score += if_component
 
    # Temperature (0-25 pts)
    if   temp > 85: risk_score += 25
    elif temp > 75: risk_score += 18
    elif temp > 65: risk_score += 10
    elif temp > 55: risk_score += 4
 
    # Memory pressure (0-15 pts)
    if   mem > 95: risk_score += 15
    elif mem > 85: risk_score += 8
    elif mem > 75: risk_score += 3
 
    # CPU overload (0-10 pts)
    if   cpu > 95: risk_score += 10
    elif cpu > 85: risk_score += 5
 
    # Battery (0-15 pts) — only penalise if actually on battery (not charging)
    if not extra.get("is_charging", True) and bat >= 0:
        if   bat < 10: risk_score += 15
        elif bat < 20: risk_score += 10
        elif bat < 30: risk_score += 5
 
    # Vibration / Z-score (0-15 pts)
    if   z > 4:    risk_score += 15
    elif z > 3:    risk_score += 10
    elif z > 2:    risk_score += 5
 
    # Disk (0-10 pts)
    if   disk > 95: risk_score += 10
    elif disk > 88: risk_score += 5
 
    risk_score = int(min(risk_score, 100))
    risk = "CRITICAL" if risk_score >= 60 else "WARNING" if risk_score >= 30 else "NOMINAL"
 
    # Recommendations
    recs = []
    if temp > 75:    recs.append("🔥 Critical temperature — check cooling system")
    elif temp > 60:  recs.append("🌡️ Elevated temp — close heavy applications")
    if mem > 85:     recs.append("💾 High memory — close unused programs")
    if cpu > 85:     recs.append("⚙️ CPU overload — check background processes")
    if not extra.get("is_charging", True) and bat >= 0 and bat < 25:
        recs.append("🔋 Low battery — connect charger soon")
    if disk > 88:    recs.append("💿 Disk nearly full — free up space")
    if z > 2.5:      recs.append("📳 Abnormal vibration signal detected")
    if trend=="RISING": recs.append("📈 Anomaly signal rising — monitor closely")
    if extra.get("proc_count", 0) > 400:
        recs.append(f"🔄 {extra['proc_count']} processes running — consider cleanup")
    if not recs:     recs.append("✅ All systems nominal — no action required")
 
    hpush("score", risk_score)
 
    return {
        "health": health,
        "score": risk_score,
        "risk": risk,
        "fault": fault,
        "fault_confidence": round(fconf, 1),
        "if_component": round(if_component, 1),
        "vibration": round(vib, 4),
        "z_score": round(z, 3),
        "trend": trend,
        "slope": round(slope, 6),
        "frequency": round(freq, 4),
        "rul_hours": rul,
        "cpu": round(cpu, 1),
        "memory": round(mem, 1),
        "battery": round(bat, 1),
        "temp": round(temp, 1),
        "disk": round(disk, 1),
        "battery_life": bat_life,
        "is_charging": extra.get("is_charging", False),
        "cpu_freq": extra.get("cpu_freq", 0),
        "ram_total": extra.get("ram_total", 0),
        "ram_used": extra.get("ram_used", 0),
        "proc_count": extra.get("proc_count", 0),
        "device": extra.get("device", "unknown"),
        "platform": extra.get("platform", "unknown"),
        "recommendations": recs,
        "history": {k: hist[k][-30:] for k in ["score","vib","cpu","mem","temp"]},
        "timestamp": round(t * 1000),
    }
 
# ── ROUTES ─────────────────────────────────────────────────────────────────────
last = {}
 
@app.get("/")
async def root(): return await monitor()
 
@app.post("/predict")
async def predict(data: dict):
    global last
    last = run_ml(
        float(data.get("vibration", 0.02)),
        float(data.get("cpu", data.get("cpu_load", 50))),
        float(data.get("memory", 50)),
        float(data.get("battery", -1)),
        float(data.get("temp", data.get("temperature", 40))),
        float(data.get("disk", 50)),
        data  # pass full dict for extra fields
    )
    return last
 
@app.get("/data")
async def get_data(): return last if last else {"error": "No data yet — run probe.py"}
 
@app.get("/monitor", response_class=HTMLResponse)
async def monitor(): return HTML
 
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NEXUS PM — Predictive Maintenance</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#04080f;--bg2:#080f1a;--bg3:#0d1825;--bg4:#112030;--cyan:#00e5ff;--green:#00ff9d;--yellow:#ffd60a;--red:#ff2d55;--orange:#ff6b00;--purple:#bd93f9;--text:#c8dff0;--text2:#4a6a88;--text3:#1e3448;--r:10px}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,229,255,.008) 3px,rgba(0,229,255,.008) 4px);pointer-events:none;z-index:0}
.wrap{max-width:1440px;margin:0 auto;padding:12px;position:relative;z-index:1}
 
/* TOP BAR */
.topbar{display:flex;align-items:center;gap:10px;padding:10px 16px;background:var(--bg2);border:1px solid #0a2540;border-radius:var(--r);margin-bottom:12px;flex-wrap:wrap}
.logo{font-family:'Share Tech Mono',monospace;font-size:14px;color:var(--cyan);letter-spacing:3px;flex-shrink:0}
.logo b{color:var(--green)}
.tabs{display:flex;gap:6px;flex:1;justify-content:center;flex-wrap:wrap}
.tab{padding:5px 14px;border:1px solid var(--text3);border-radius:5px;cursor:pointer;font-size:12px;font-weight:600;letter-spacing:1px;background:transparent;color:var(--text2);transition:.2s;font-family:'Rajdhani',sans-serif}
.tab.on{border-color:var(--cyan);color:var(--cyan);background:rgba(0,229,255,.07)}
.pdot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pd 2s infinite;flex-shrink:0}
@keyframes pd{0%,100%{box-shadow:0 0 0 0 rgba(0,255,157,.6)}60%{box-shadow:0 0 0 7px rgba(0,255,157,0)}}
#clk{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--text2)}
 
/* RISK BANNER */
.banner{display:flex;align-items:center;gap:14px;padding:12px 18px;border-radius:var(--r);margin-bottom:12px;border:1px solid;transition:all .5s;flex-wrap:wrap}
.banner.NOMINAL{background:rgba(0,255,157,.05);border-color:rgba(0,255,157,.25)}
.banner.WARNING{background:rgba(255,214,10,.05);border-color:rgba(255,214,10,.4);animation:bw 2s infinite}
.banner.CRITICAL{background:rgba(255,45,85,.08);border-color:rgba(255,45,85,.5);animation:bc .9s infinite}
@keyframes bw{0%,100%{border-color:rgba(255,214,10,.35)}50%{border-color:rgba(255,214,10,.9)}}
@keyframes bc{0%,100%{border-color:rgba(255,45,85,.5)}50%{border-color:rgba(255,45,85,1)}}
.banner-icon{font-size:28px;flex-shrink:0}
.banner-text h2{font-size:20px;font-weight:700;letter-spacing:3px}
.banner-text p{font-size:12px;color:var(--text2);margin-top:2px}
.banner-score{margin-left:auto;text-align:right}
.banner-score .n{font-family:'Share Tech Mono',monospace;font-size:44px;line-height:1}
.banner-score .l{font-size:10px;color:var(--text2);letter-spacing:2px}
 
/* GRID LAYOUTS */
.g3{display:grid;grid-template-columns:250px 1fr 260px;gap:12px;margin-bottom:12px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.g3b{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px}
.g4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:12px}
@media(max-width:1100px){.g3{grid-template-columns:1fr 1fr}}
@media(max-width:750px){.g3,.g2,.g3b,.g4{grid-template-columns:1fr}}
 
/* CARDS */
.card{background:var(--bg2);border:1px solid #0a2540;border-radius:var(--r);padding:14px;animation:fi .35s ease both}
@keyframes fi{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
.ct{font-size:9px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:7px}
.ct::before{content:'';width:2px;height:10px;background:var(--cyan);border-radius:2px;flex-shrink:0}
 
/* BIG GAUGE */
.bgw{display:flex;flex-direction:column;align-items:center;padding:4px 0}
.bgn{font-family:'Share Tech Mono',monospace;font-size:44px;font-weight:700;text-align:center;line-height:1;margin-top:-4px}
.bgs{font-size:11px;color:var(--text2);text-align:center;margin-top:4px;letter-spacing:1px}
 
/* MINI GAUGES */
.mg-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.mgw{display:flex;flex-direction:column;align-items:center;gap:3px}
.mg-val{font-family:'Share Tech Mono',monospace;font-size:15px;font-weight:700;text-align:center}
.mg-lbl{font-size:10px;color:var(--text2);letter-spacing:1px;text-align:center}
 
/* METRIC ROWS */
.mr{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--bg4)}
.mr:last-child{border-bottom:none}
.mi{width:26px;height:26px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0}
.mn{font-size:11px;color:var(--text2);font-weight:600;letter-spacing:.3px}
.mv{font-family:'Share Tech Mono',monospace;font-size:15px;font-weight:700;line-height:1.2}
.mb{height:3px;background:var(--bg4);border-radius:2px;margin-top:3px}
.mbf{height:100%;border-radius:2px;transition:width .7s ease}
 
/* STAT BOXES */
.sg{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.sb{background:var(--bg3);border:1px solid var(--bg4);border-radius:7px;padding:9px 11px}
.sl{font-size:9px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:3px}
.sv{font-family:'Share Tech Mono',monospace;font-size:14px;font-weight:700}
 
/* FAULT */
.fb{border-radius:8px;padding:12px 14px;margin-bottom:10px;border:1px solid}
.fn{font-size:16px;font-weight:700;letter-spacing:2px;margin-bottom:3px}
.fc{font-size:11px;color:var(--text2)}
 
/* SPEC ROW — shows extra PC fields */
.spec-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:12px}
.spec-item{background:var(--bg2);border:1px solid #0a2540;border-radius:8px;padding:10px 12px}
.spec-l{font-size:9px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:3px}
.spec-v{font-family:'Share Tech Mono',monospace;font-size:14px;font-weight:700;color:var(--cyan)}
 
/* CHARTS */
.sc{position:relative;height:75px;margin-top:6px}
 
/* RECS */
.ri{display:flex;gap:8px;padding:6px 0;border-bottom:1px solid var(--bg4);font-size:13px;font-weight:500;line-height:1.4}
.ri:last-child{border-bottom:none}
 
/* INPUTS */
.ig{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
.ig3{grid-template-columns:1fr 1fr 1fr}
.igrp{display:flex;flex-direction:column;gap:5px}
.igrp label{font-size:10px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase}
.igrp input{background:var(--bg3);border:1px solid var(--bg4);border-radius:7px;color:var(--text);font-family:'Share Tech Mono',monospace;font-size:13px;padding:8px 11px;outline:none;transition:.2s;width:100%}
.igrp input:focus{border-color:var(--cyan);box-shadow:0 0 0 2px rgba(0,229,255,.1)}
.igrp .hint{font-size:10px;color:var(--text3)}
.sbtn{width:100%;padding:11px;background:rgba(0,229,255,.08);border:1px solid var(--cyan);border-radius:var(--r);color:var(--cyan);font-family:'Rajdhani',sans-serif;font-size:14px;font-weight:700;letter-spacing:2px;cursor:pointer;transition:.2s}
.sbtn:hover{background:rgba(0,229,255,.18)}
.gbtn{padding:13px 28px;background:rgba(0,255,157,.08);border:1px solid var(--green);border-radius:var(--r);color:var(--green);font-family:'Rajdhani',sans-serif;font-size:15px;font-weight:700;letter-spacing:2px;cursor:pointer;transition:.2s}
.gbtn:hover{background:rgba(0,255,157,.18)}
 
.probe-box{background:var(--bg3);border:1px solid var(--bg4);border-radius:8px;padding:14px;margin-bottom:12px;font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--text2)}
.probe-box h4{color:var(--cyan);font-size:12px;margin-bottom:8px;letter-spacing:1px}
.probe-box code{display:block;background:var(--bg);border:1px solid var(--bg4);border-radius:5px;padding:8px 10px;color:var(--green);margin-top:4px;word-break:break-all;line-height:1.6}
</style>
</head>
<body>
<div class="wrap">
 
<div class="topbar">
  <div class="logo">NEXUS<b>/PM</b></div>
  <div class="tabs">
    <button class="tab on" onclick="setMode('live')" id="t-live">⚡ LIVE</button>
    <button class="tab" onclick="setMode('industrial')" id="t-industrial">🏭 INDUSTRIAL</button>
    <button class="tab" onclick="setMode('custom')" id="t-custom">✏️ CUSTOM</button>
  </div>
  <div class="pdot"></div>
  <span id="clk">--:--:--</span>
</div>
 
<!-- RISK BANNER -->
<div class="banner NOMINAL" id="banner">
  <div class="banner-icon" id="b-icon">⏳</div>
  <div class="banner-text">
    <h2 id="b-title" style="color:var(--green)">WAITING</h2>
    <p id="b-desc">Run probe.py on your PC — or open this page on your phone</p>
  </div>
  <div class="banner-score">
    <div class="n" id="b-score" style="color:var(--green)">--</div>
    <div class="l">RISK / 100</div>
  </div>
</div>
 
<!-- LIVE MODE -->
<div id="m-live">
 
  <!-- spec row — real PC values -->
  <div class="spec-row" id="spec-row" style="display:none">
    <div class="spec-item"><div class="spec-l">Device</div><div class="spec-v" id="sp-dev">--</div></div>
    <div class="spec-item"><div class="spec-l">Platform</div><div class="spec-v" id="sp-plat">--</div></div>
    <div class="spec-item"><div class="spec-l">CPU Freq</div><div class="spec-v" id="sp-freq">--</div></div>
    <div class="spec-item"><div class="spec-l">RAM Used</div><div class="spec-v" id="sp-ram">--</div></div>
    <div class="spec-item"><div class="spec-l">Processes</div><div class="spec-v" id="sp-proc">--</div></div>
    <div class="spec-item"><div class="spec-l">Charging</div><div class="spec-v" id="sp-chg">--</div></div>
  </div>
 
  <!-- phone prompt -->
  <div id="ph-prompt" class="card" style="display:none;text-align:center;padding:22px;margin-bottom:12px">
    <div class="ct" style="justify-content:center">PHONE SENSOR ACCESS</div>
    <div style="font-size:44px;margin:8px 0">📱</div>
    <p style="color:var(--text2);font-size:13px;margin-bottom:14px">Tap to enable real accelerometer & battery</p>
    <button class="gbtn" onclick="startPhone()">▶ GRANT ACCESS</button>
  </div>
 
  <!-- probe hint for PC desktop -->
  <div id="probe-hint" class="probe-box">
    <h4>📡 FOR EXACT WINDOWS/PC VALUES — run this on your PC:</h4>
    <code>pip install psutil requests</code>
    <code>python probe.py</code>
    <p style="margin-top:8px;color:var(--text3);font-size:11px">Without probe.py the browser sends estimated values. probe.py sends real Task Manager numbers.</p>
  </div>
 
  <div class="g3">
    <!-- LEFT -->
    <div style="display:flex;flex-direction:column;gap:12px">
      <div class="card">
        <div class="ct">Health Index (Isolation Forest)</div>
        <div class="bgw">
          <svg width="190" height="115" viewBox="0 0 190 115" style="overflow:visible">
            <defs>
              <linearGradient id="ag" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#00ff9d"/>
                <stop offset="45%" stop-color="#ffd60a"/>
                <stop offset="100%" stop-color="#ff2d55"/>
              </linearGradient>
            </defs>
            <path d="M18,105 A77,77 0 0,1 172,105" fill="none" stroke="#0d1825" stroke-width="16" stroke-linecap="round"/>
            <path id="big-arc" d="M18,105 A77,77 0 0,1 172,105" fill="none" stroke="url(#ag)" stroke-width="16"
              stroke-linecap="round" stroke-dasharray="241.6" stroke-dashoffset="241.6"
              style="transition:stroke-dashoffset .9s ease"/>
          </svg>
          <div class="bgn" id="h-num" style="color:var(--green)">--</div>
          <div class="bgs" id="h-sub">AWAITING DATA</div>
        </div>
      </div>
 
      <div class="card">
        <div class="ct">Fault Classification (RF)</div>
        <div class="fb" id="fb" style="background:rgba(0,255,157,.05);border-color:rgba(0,255,157,.25)">
          <div class="fn" id="fn" style="color:var(--green)">NORMAL</div>
          <div class="fc" id="fc">Confidence: -- | Random Forest</div>
        </div>
        <div class="sg">
          <div class="sb"><div class="sl">Z-Score</div><div class="sv" id="z-s" style="color:var(--cyan)">--</div></div>
          <div class="sb"><div class="sl">Frequency</div><div class="sv" id="fr-v" style="color:var(--cyan)">--</div></div>
          <div class="sb"><div class="sl">Trend</div><div class="sv" id="tr-v">--</div></div>
          <div class="sb"><div class="sl">RUL Est.</div><div class="sv" id="rl-v" style="color:var(--yellow)">--</div></div>
        </div>
      </div>
    </div>
 
    <!-- CENTER -->
    <div style="display:flex;flex-direction:column;gap:12px">
      <div class="card">
        <div class="ct">Parameter Gauges</div>
        <div class="mg-grid" id="mg-grid"><div style="color:var(--text3);font-size:12px;grid-column:span 2">Awaiting data...</div></div>
      </div>
      <div class="card">
        <div class="ct">Live Metrics</div>
        <div id="met-list"><div style="color:var(--text3);font-size:12px">Awaiting data...</div></div>
      </div>
    </div>
 
    <!-- RIGHT -->
    <div style="display:flex;flex-direction:column;gap:12px">
      <div class="card">
        <div class="ct">AI Recommendations</div>
        <div id="rec-list"><div style="color:var(--text3);font-size:13px">Awaiting analysis...</div></div>
      </div>
      <div class="card">
        <div class="ct">Battery Intelligence</div>
        <div style="text-align:center;padding:6px 0 10px">
          <div style="font-family:'Share Tech Mono',monospace;font-size:36px" id="bat-n" class="">--%</div>
          <div style="font-size:12px;color:var(--text2);margin-top:3px" id="bat-l">--</div>
        </div>
        <div style="height:5px;background:var(--bg4);border-radius:3px;margin-bottom:10px">
          <div id="bat-b" style="height:100%;border-radius:3px;width:0%;background:var(--green);transition:width .7s"></div>
        </div>
        <div class="sg">
          <div class="sb"><div class="sl">Vibration</div><div class="sv" id="vib-v" style="color:var(--cyan)">--</div></div>
          <div class="sb"><div class="sl">IF Score</div><div class="sv" id="if-v" style="color:var(--purple)">--</div></div>
        </div>
      </div>
    </div>
  </div>
 
  <!-- CHARTS -->
  <div class="g3b">
    <div class="card"><div class="ct">Risk Score</div><div class="sc"><canvas id="ch-s"></canvas></div></div>
    <div class="card"><div class="ct">CPU + Memory</div><div class="sc"><canvas id="ch-c"></canvas></div></div>
    <div class="card"><div class="ct">Temperature</div><div class="sc"><canvas id="ch-t"></canvas></div></div>
  </div>
</div>
 
<!-- INDUSTRIAL MODE -->
<div id="m-industrial" style="display:none">
  <div class="card">
    <div class="ct">Industrial Machine Parameters</div>
    <div class="ig">
      <div class="igrp"><label>Vibration (m/s²)</label><input type="number" id="iv" value="0.08" step="0.01"/><div class="hint">Normal &lt;0.15 | Warn &gt;0.5 | Critical &gt;0.8</div></div>
      <div class="igrp"><label>Temperature (°C)</label><input type="number" id="it" value="42" step="1"/><div class="hint">Normal &lt;55 | Warn 55-75 | Critical &gt;75</div></div>
      <div class="igrp"><label>RPM / Speed Load (%)</label><input type="number" id="ic" value="65" step="1" min="0" max="100"/><div class="hint">Machine speed as % of max rated</div></div>
      <div class="igrp"><label>Pressure / Mech Load (%)</label><input type="number" id="im" value="55" step="1" min="0" max="100"/><div class="hint">Hydraulic or mechanical load</div></div>
      <div class="igrp"><label>Power Supply (%)</label><input type="number" id="ib" value="95" step="1" min="0" max="100"/><div class="hint">Input power health / voltage stability</div></div>
      <div class="igrp"><label>Wear Index (%)</label><input type="number" id="id" value="35" step="1" min="0" max="100"/><div class="hint">Cumulative wear from cycle count</div></div>
    </div>
    <button class="sbtn" onclick="subInd()">⚙️ RUN ML ANALYSIS</button>
  </div>
</div>
 
<!-- CUSTOM MODE -->
<div id="m-custom" style="display:none">
  <div class="card">
    <div class="ct">Custom Parameter Input — any 6 values</div>
    <div class="ig">
      <div class="igrp"><label>Vibration / Anomaly Signal</label><input type="number" id="xv" value="0.05" step="0.001"/></div>
      <div class="igrp"><label>CPU / Primary Load (%)</label><input type="number" id="xc" value="50" step="1" min="0" max="100"/></div>
      <div class="igrp"><label>Memory / Secondary Load (%)</label><input type="number" id="xm" value="50" step="1" min="0" max="100"/></div>
      <div class="igrp"><label>Battery / Power (%)</label><input type="number" id="xb" value="80" step="1" min="0" max="100"/></div>
      <div class="igrp"><label>Temperature (°C)</label><input type="number" id="xt" value="40" step="1"/></div>
      <div class="igrp"><label>Disk / Wear (%)</label><input type="number" id="xd" value="50" step="1" min="0" max="100"/></div>
    </div>
    <button class="sbtn" onclick="subCus()">🧠 ANALYZE WITH ML</button>
  </div>
</div>
 
</div><!-- wrap -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
let charts={}, phoneOn=false, batLvl=-1, mode='live';
 
// Clock
setInterval(()=>{document.getElementById('clk').textContent=new Date().toLocaleTimeString('en',{hour12:false})},1000);
 
// Battery API
if(navigator.getBattery)navigator.getBattery().then(b=>{
  batLvl=Math.round(b.level*100);
  b.addEventListener('levelchange',()=>{batLvl=Math.round(b.level*100);});
}).catch(()=>{});
 
// Mode
function setMode(m){
  mode=m;
  ['live','industrial','custom'].forEach(k=>{
    document.getElementById('m-'+k).style.display=k===m?'block':'none';
    document.getElementById('t-'+k).classList.toggle('on',k===m);
  });
  if(m==='live'&&/Mobi|Android/i.test(navigator.userAgent)&&!phoneOn)
    document.getElementById('ph-prompt').style.display='block';
}
 
// Show phone prompt on mobile
if(/Mobi|Android/i.test(navigator.userAgent))document.getElementById('ph-prompt').style.display='block';
 
// Color helpers
function cFor(v,w,c){return v>=c?'#ff2d55':v>=w?'#ffd60a':'#00ff9d'}
function cForInv(v,w,c){return v<=c?'#ff2d55':v<=w?'#ffd60a':'#00ff9d'}
 
// RENDER
function render(d){
  if(!d||d.error)return;
  const C={NOMINAL:'#00ff9d',WARNING:'#ffd60a',CRITICAL:'#ff2d55'};
  const I={NOMINAL:'✅',WARNING:'⚠️',CRITICAL:'🚨'};
  const D={NOMINAL:'All systems nominal',WARNING:'Elevated risk — schedule maintenance',CRITICAL:'Immediate action required — multiple failures'};
  const col=C[d.risk]||'#00ff9d';
 
  // Banner
  document.getElementById('banner').className='banner '+(d.risk||'NOMINAL');
  document.getElementById('b-icon').textContent=I[d.risk]||'✅';
  document.getElementById('b-title').textContent=d.risk||'--';
  document.getElementById('b-title').style.color=col;
  document.getElementById('b-desc').textContent=D[d.risk]||'';
  document.getElementById('b-score').textContent=d.score;
  document.getElementById('b-score').style.color=col;
 
  // Big gauge
  document.getElementById('big-arc').style.strokeDashoffset=241.6*(1-(d.health||0)/100);
  document.getElementById('h-num').textContent=(d.health||0)+'%';
  document.getElementById('h-num').style.color=col;
  document.getElementById('h-sub').textContent=d.risk+' — '+(d.fault||'').replace(/_/g,' ');
 
  // Fault
  const FC={NORMAL:['rgba(0,255,157,.05)','rgba(0,255,157,.25)','#00ff9d'],
    BEARING_FAULT:['rgba(255,107,0,.08)','rgba(255,107,0,.4)','#ff6b00'],
    MISALIGNMENT:['rgba(255,214,10,.06)','rgba(255,214,10,.35)','#ffd60a'],
    IMBALANCE:['rgba(255,45,85,.08)','rgba(255,45,85,.4)','#ff2d55'],
    OVERHEATING:['rgba(255,45,85,.12)','rgba(255,45,85,.6)','#ff2d55'],
    WEAR:['rgba(189,147,249,.06)','rgba(189,147,249,.3)','#bd93f9']};
  const fc=FC[d.fault]||FC.NORMAL;
  const fb=document.getElementById('fb');
  fb.style.background=fc[0];fb.style.borderColor=fc[1];
  document.getElementById('fn').textContent=(d.fault||'NORMAL').replace(/_/g,' ');
  document.getElementById('fn').style.color=fc[2];
  document.getElementById('fc').textContent=`Confidence: ${d.fault_confidence}% | Random Forest`;
 
  // Stats
  const ze=document.getElementById('z-s');
  ze.textContent=d.z_score;ze.style.color=d.z_score>3?'#ff2d55':d.z_score>2?'#ffd60a':'#00e5ff';
  document.getElementById('fr-v').textContent=d.frequency+' Hz';
  const te=document.getElementById('tr-v');
  te.textContent=d.trend;te.style.color=d.trend==='RISING'?'#ff2d55':d.trend==='FALLING'?'#ffd60a':'#00ff9d';
  document.getElementById('rl-v').textContent=d.rul_hours!=null?d.rul_hours+'h':'N/A';
 
  // Battery
  const bc=d.battery<20?'#ff2d55':d.battery<35?'#ffd60a':'#00ff9d';
  document.getElementById('bat-n').textContent=d.battery>=0?d.battery+'%':'N/A';
  document.getElementById('bat-n').style.color=bc;
  document.getElementById('bat-l').textContent=(d.is_charging?'⚡ Charging — ':'🔋 ')+(d.battery_life||'--');
  document.getElementById('bat-b').style.width=Math.max(0,d.battery)+'%';
  document.getElementById('bat-b').style.background=bc;
  document.getElementById('vib-v').textContent=d.vibration;
  document.getElementById('if-v').textContent=d.if_component?.toFixed(1)||'--';
 
  // Recs
  document.getElementById('rec-list').innerHTML=
    (d.recommendations||[]).map(r=>`<div class="ri">${r}</div>`).join('')||
    '<div style="color:var(--text3);font-size:12px">No issues</div>';
 
  // Mini gauges
  const gs=[
    {l:'CPU',v:d.cpu,u:'%',p:d.cpu,c:cFor(d.cpu,70,90)},
    {l:'MEMORY',v:d.memory,u:'%',p:d.memory,c:cFor(d.memory,75,88)},
    {l:'TEMP',v:d.temp,u:'°C',p:Math.min(d.temp/90*100,100),c:cFor(d.temp,55,72)},
    {l:'DISK',v:d.disk,u:'%',p:d.disk,c:cFor(d.disk,80,92)},
  ];
  document.getElementById('mg-grid').innerHTML=gs.map(g=>`
    <div class="mgw">
      <svg width="105" height="62" viewBox="0 0 105 62" style="overflow:visible">
        <path d="M9,57 A43,43 0 0,1 96,57" fill="none" stroke="#0d1825" stroke-width="9" stroke-linecap="round"/>
        <path d="M9,57 A43,43 0 0,1 96,57" fill="none" stroke="${g.c}" stroke-width="9" stroke-linecap="round"
          stroke-dasharray="135" stroke-dashoffset="${135*(1-g.p/100)}" style="transition:stroke-dashoffset .6s"/>
      </svg>
      <div class="mg-val" style="color:${g.c}">${g.v.toFixed(1)}${g.u}</div>
      <div class="mg-lbl">${g.l}</div>
    </div>`).join('');
 
  // Metrics
  const ms=[
    {i:'⚙️',n:'CPU Load',v:d.cpu+'%',b:d.cpu,c:cFor(d.cpu,70,90)},
    {i:'💾',n:'Memory',v:d.memory+'%',b:d.memory,c:cFor(d.memory,75,88)},
    {i:'🌡️',n:'Temperature',v:d.temp+'°C',b:Math.min(d.temp/90*100,100),c:cFor(d.temp,55,72)},
    {i:'💿',n:'Disk',v:d.disk+'%',b:d.disk,c:cFor(d.disk,80,92)},
  ];
  document.getElementById('met-list').innerHTML=ms.map(m=>`
    <div class="mr">
      <div class="mi" style="background:${m.c}1a">${m.i}</div>
      <div style="flex:1;min-width:0">
        <div class="mn">${m.n}</div>
        <div class="mv" style="color:${m.c}">${m.v}</div>
        <div class="mb"><div class="mbf" style="width:${m.b}%;background:${m.c}"></div></div>
      </div>
    </div>`).join('');
 
  // Spec row (probe.py extra fields)
  if(d.device&&d.device!=='unknown'&&d.device!=='Desktop'&&d.device!=='Phone'){
    document.getElementById('spec-row').style.display='grid';
    document.getElementById('probe-hint').style.display='none';
    document.getElementById('sp-dev').textContent=d.device.substring(0,18);
    document.getElementById('sp-plat').textContent=d.platform||'--';
    document.getElementById('sp-freq').textContent=d.cpu_freq?d.cpu_freq+' MHz':'--';
    document.getElementById('sp-ram').textContent=d.ram_used&&d.ram_total?`${d.ram_used}/${d.ram_total}GB`:'--';
    document.getElementById('sp-proc').textContent=d.proc_count||'--';
    document.getElementById('sp-chg').textContent=d.is_charging?'⚡ Yes':'🔋 No';
  }
 
  // Charts
  if(d.history){
    const h=d.history,lb=h.score.map((_,i)=>i);
    function up(ch,ds){ch.data.labels=lb;ds.forEach((v,i)=>{ch.data.datasets[i].data=v;});ch.update('none');}
    up(charts.s,[h.score]);
    up(charts.c,[h.cpu,h.mem]);
    up(charts.t,[h.temp]);
  }
}
 
// Charts init
function initCharts(){
  const base={responsive:true,maintainAspectRatio:false,animation:{duration:400},plugins:{legend:{display:false}},
    scales:{x:{display:false},y:{display:true,min:0,max:100,grid:{color:'#0d1825'},ticks:{color:'#1e3448',font:{size:8},maxTicksLimit:3}}}};
  charts.s=new Chart(document.getElementById('ch-s'),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#ff2d55',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'#ff2d5514',tension:.4}]},options:base});
  charts.c=new Chart(document.getElementById('ch-c'),{type:'line',data:{labels:[],datasets:[
    {label:'CPU',data:[],borderColor:'#00e5ff',borderWidth:1.5,pointRadius:0,fill:false,tension:.4},
    {label:'Mem',data:[],borderColor:'#bd93f9',borderWidth:1.5,pointRadius:0,fill:false,tension:.4}
  ]},options:{...base,plugins:{legend:{display:true,labels:{color:'#4a6a88',font:{size:9},boxWidth:8}}}}});
  charts.t=new Chart(document.getElementById('ch-t'),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#ff6b00',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'#ff6b0012',tension:.4}]},options:{...base,scales:{...base.scales,y:{...base.scales.y,min:20,max:100}}}});
}
 
// Poll
setInterval(async()=>{
  try{const r=await fetch('/data');const d=await r.json();if(d&&!d.error)render(d);}catch(e){}
},2000);
 
// Browser auto-send (fallback when probe.py not running)
let autoT=setInterval(async()=>{
  if(mode!=='live')return;
  const mem=performance.memory?Math.round(performance.memory.usedJSHeapSize/performance.memory.jsHeapSizeLimit*100):50;
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({vibration:0.02,cpu:50,memory:mem,battery:batLvl,temp:40,disk:50,device:'Desktop',platform:'Browser'})
    });
    render(await r.json());
  }catch(e){}
},5000);
 
// Phone sensors
function startPhone(){
  function go(){
    phoneOn=true;document.getElementById('ph-prompt').style.display='none';
    let st=null;
    window.addEventListener('devicemotion',e=>{
      const a=e.accelerationIncludingGravity;if(!a)return;
      const vib=Math.max(0,(Math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z)-9.81));
      clearTimeout(st);st=setTimeout(async()=>{
        try{
          const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({vibration:parseFloat(vib.toFixed(4)),battery:batLvl,cpu:50,memory:60,temp:36,disk:50,device:'Phone',platform:'Android'})
          });render(await r.json());
        }catch(e){}
      },400);
    });
  }
  if(typeof DeviceMotionEvent!=='undefined'&&typeof DeviceMotionEvent.requestPermission==='function')
    DeviceMotionEvent.requestPermission().then(p=>{if(p==='granted')go();}).catch(()=>{});
  else go();
}
 
// Submit handlers
async function subInd(){
  const d={vibration:+document.getElementById('iv').value,cpu:+document.getElementById('ic').value,
    memory:+document.getElementById('im').value,battery:+document.getElementById('ib').value,
    temp:+document.getElementById('it').value,disk:+document.getElementById('id').value,device:'Industrial',platform:'Industrial'};
  try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});render(await r.json());setMode('live');}catch(e){}
}
async function subCus(){
  const d={vibration:+document.getElementById('xv').value,cpu:+document.getElementById('xc').value,
    memory:+document.getElementById('xm').value,battery:+document.getElementById('xb').value,
    temp:+document.getElementById('xt').value,disk:+document.getElementById('xd').value,device:'Custom',platform:'Custom'};
  try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});render(await r.json());setMode('live');}catch(e){}
}
 
initCharts();
</script>
</body>
</html>"""
 
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)