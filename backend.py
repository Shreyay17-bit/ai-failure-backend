import os
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.ensemble import IsolationForest

app = FastAPI()

rng = np.random.default_rng(42)
N   = 1200

X_healthy = np.column_stack([
    rng.uniform(0.00, 0.40, N),   # vibration 
    rng.uniform(5,    80,   N),   # cpu 
    rng.uniform(10,   80,   N),   # ram 
    rng.uniform(25,   100,  N),   # battery
    rng.uniform(25,   68,   N),   # temperature 
    rng.uniform(5,    80,   N),   # disk
])

iforest = IsolationForest(contamination=0.02, n_estimators=200, random_state=42)
iforest.fit(X_healthy)


_s_h = iforest.score_samples(X_healthy)
GOOD_THRESH = float(np.percentile(_s_h, 95))

X_fault = np.column_stack([
    rng.uniform(1.5, 3.0, 300), rng.uniform(88, 100, 300),
    rng.uniform(88, 100, 300),  rng.uniform(1,   10, 300),
    rng.uniform(85, 100, 300),  rng.uniform(88, 100, 300),
])
_s_f = iforest.score_samples(X_fault)
BAD_THRESH = float(np.percentile(_s_f, 5))

def if_score(vec: list) -> float:
    """Raw IF output mapped to 0–100 (0=healthy, 100=extreme anomaly)."""
    raw = float(iforest.score_samples([vec])[0])
    return float(np.clip(
        (GOOD_THRESH - raw) / (GOOD_THRESH - BAD_THRESH) * 100,
        0, 100
    ))

def rule_penalty(vib, cpu, ram, bat, temp, disk) -> int:
    """Rule-based additive penalty covering IF blind-spots (battery, temp)."""
    p = 0
    if   bat  <  5:  p += 50
    elif bat  < 15:  p += 35
    elif bat  < 25:  p += 18
    if   vib  > 1.8: p += 35
    elif vib  > 1.0: p += 20
    elif vib  > 0.6: p += 8
    if   temp > 85:  p += 30
    elif temp > 75:  p += 15
    elif temp > 70:  p += 6
    if   cpu  > 95:  p += 18
    elif cpu  > 88:  p += 8
    if   ram  > 95:  p += 12
    elif ram  > 90:  p += 5
    if   disk > 95:  p += 10
    elif disk > 90:  p += 4
    return min(p, 60)

def classify_fault(vib, cpu, ram, bat, temp, disk, score):
    if score < 30:
        return "NORMAL"
    candidates = {
        "BEARING FAULT":   vib  * 30,
        "OVERHEATING":     (temp - 40) * 1.5 + cpu * 0.2,
        "MEMORY OVERLOAD": ram  * 0.8  + cpu * 0.3,
        "LOW POWER":       max(0, (30 - bat) * 2.5),
        "DISK PRESSURE":   disk * 0.7,
    }
    return max(candidates, key=candidates.get)

all_devices:    dict = {}
device_history: dict = {}


@app.post("/predict")
async def predict(data: dict):
    vib  = float(data.get("vibration",   0.1))
    cpu  = float(data.get("cpu_load",    30))
    ram  = float(data.get("memory",      45))
    bat  = float(data.get("battery",     85))
    temp = float(data.get("temperature", 42))
    disk = float(data.get("disk_usage",  50))
    dev  = str(data.get("device_type",   "Unknown"))

    
    if_comp = if_score([vib, cpu, ram, bat, temp, disk])

    
    penalty = rule_penalty(vib, cpu, ram, bat, temp, disk)

    
    raw = if_comp * 0.60 + (penalty / 60 * 100) * 0.40
    health_score = int(np.clip(raw, 0, 100))

    
    all_fine = (vib < 0.45 and cpu < 85 and ram < 85
                and bat > 22 and temp < 72 and disk < 88)
    if all_fine:
        health_score = min(health_score, 28)

    fault = classify_fault(vib, cpu, ram, bat, temp, disk, health_score)

    
    dev_hist = device_history.get(dev, [])
    if len(dev_hist) >= 3:
        bats = [h["bat"] for h in dev_hist[-6:]]
        drain_per_scan = (bats[0] - bats[-1]) / len(bats)
        drain_per_hour = drain_per_scan * (3600 / 4)
        rul_h     = int(bat / drain_per_hour) if drain_per_hour > 0.1 else 999
        bat_trend = ("RISING"  if drain_per_scan < -0.2 else
                     "DRAINING" if drain_per_scan >  0.2 else "STABLE")
    else:
        rul_h, bat_trend = 999, "STABLE"

    
    vib_health = ("SMOOTH" if vib < 0.15 else "NORMAL" if vib < 0.40 else
                  "ELEVATED" if vib < 0.70 else "HIGH" if vib < 1.20 else "CRITICAL")
    thermal    = ("COOL" if temp < 40 else "NORMAL" if temp < 60 else
                  "WARM" if temp < 72 else "HOT" if temp < 82 else "OVERHEATING")

    if health_score >= 65:
        risk, status = "CRITICAL", "CRITICAL — IMMEDIATE ACTION"
        recs = [
            f"HALT OPERATIONS — primary fault detected: {fault}",
            f"Health score {health_score}/100 exceeds critical threshold (65)",
            "Inspect mechanical and thermal subsystems before restart",
            f"Battery at {bat:.0f}% — {'replace immediately' if bat < 15 else 'monitor closely'}",
        ]
    elif health_score >= 35:
        risk, status = "WARNING", "ADVISORY — SCHEDULE MAINTENANCE"
        recs = [
            f"Fault signature detected: {fault}",
            f"Health score {health_score}/100 — above normal range (35–64)",
            f"Battery life estimate: {'stable' if rul_h == 999 else str(rul_h) + 'h remaining'}",
            f"Vibration: {vib_health} — {'inspect bearings' if vib > 0.5 else 'continue monitoring'}",
        ]
    else:
        risk, status = "NOMINAL", "NOMINAL — ALL SYSTEMS HEALTHY"
        recs = [
            f"All parameters within healthy envelope — score {health_score}/100",
            f"Vibration: {vib_health}  ·  Thermal: {thermal}  ·  Battery: {bat:.0f}%",
            f"Battery life: {'stable / charging' if rul_h == 999 else str(rul_h) + 'h at current drain rate'}",
            "Continue standard monitoring intervals",
        ]

    result = dict(
        vibration=round(vib,3), cpu_load=round(cpu,1),
        memory=round(ram,1),    battery=round(bat,1),
        temperature=round(temp,1), disk_usage=round(disk,1),
        device=dev,
        health_score=health_score,
        if_component=round(if_comp,1),
        penalty=penalty,
        fault=fault, risk=risk, status=status,
        rul_hours=rul_h, bat_trend=bat_trend,
        vib_health=vib_health, thermal=thermal,
        recommendations=recs,
    )

    all_devices[dev] = result
    device_history.setdefault(dev, []).append(
        {"bat": bat, "score": health_score, "cpu": cpu, "vib": vib}
    )
    if len(device_history[dev]) > 30:
        device_history[dev].pop(0)

    return result


@app.get("/data")
async def data():
    latest = list(all_devices.values())[-1] if all_devices else {}
    per_dev_hist = {k: [h["score"] for h in v] for k, v in device_history.items()}
    return {**latest, "per_dev_hist": per_dev_hist, "all_devices": all_devices}


@app.get("/")
async def root():
    return {"status": "online", "version": "3.0.0", "model": "IsolationForest"}


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
  --bg:       #0c0b08;
  --surface:  #141310;
  --surface2: #1c1a16;
  --border:   #2e2b24;
  --border2:  #3d3a30;
  --amber:    #f0a830;
  --amber2:   #d48a18;
  --amber-lo: #3a2800;
  --amber-hi: #ffe0a0;
  --red:      #e04030;
  --red-lo:   #3a0e08;
  --red-hi:   #ff9080;
  --green:    #5ab870;
  --green-lo: #0d2a18;
  --green-hi: #a0f0b8;
  --yellow:   #e8c040;
  --text:     #e8dfc8;
  --text2:    #a89880;
  --muted:    #6a5e48;
  --mono:    'Share Tech Mono', monospace;
  --display: 'Bebas Neue', cursive;
  --body:    'DM Sans', sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:14px}
body{background:var(--bg);color:var(--text);font-family:var(--body);min-height:100vh;overflow-x:hidden}
body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.04) 3px,rgba(0,0,0,0.04) 4px);pointer-events:none;z-index:9999}

header{display:flex;align-items:stretch;border-bottom:2px solid var(--border);background:var(--surface)}
.hdr-brand{padding:16px 28px;border-right:2px solid var(--border);display:flex;flex-direction:column;gap:1px}
.hdr-label{font-family:var(--mono);font-size:8px;letter-spacing:4px;color:var(--muted);text-transform:uppercase}
.hdr-name{font-family:var(--display);font-size:36px;color:var(--amber);line-height:1;letter-spacing:3px}
.hdr-right{flex:1;display:flex;align-items:center;padding:0 28px;gap:28px;flex-wrap:wrap}
.hdr-pill{display:flex;align-items:center;gap:9px;font-family:var(--mono);font-size:10px;letter-spacing:2px;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pdot 2.4s ease-in-out infinite}
@keyframes pdot{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(90,184,112,.4)}50%{opacity:.7;box-shadow:0 0 0 5px rgba(90,184,112,0)}}
.hdr-time{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--muted);text-align:right;line-height:2}

.shell{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 75px)}
.left-col{border-right:2px solid var(--border);display:flex;flex-direction:column}
.right-col{display:flex;flex-direction:column}

.sec{padding:10px 18px;border-bottom:1px solid var(--border);font-family:var(--mono);font-size:8px;letter-spacing:4px;color:var(--muted);background:var(--surface);display:flex;align-items:center;gap:8px;text-transform:uppercase}
.sec::before{content:'';display:block;width:3px;height:9px;background:var(--amber);border-radius:1px}

/* gauge */
.gauge-wrap{padding:22px 20px 18px;border-bottom:1px solid var(--border)}
.gauge-sub{font-family:var(--mono);font-size:8px;letter-spacing:3px;color:var(--muted);margin-bottom:14px}
.arc-outer{position:relative;width:190px;height:110px;margin:0 auto 14px}
.arc-outer svg{width:100%;height:100%;overflow:visible}
.score-center{position:absolute;bottom:4px;left:50%;transform:translateX(-50%);text-align:center}
.score-num{font-family:var(--display);font-size:52px;line-height:1;color:var(--amber);transition:color .5s}
.score-denom{font-family:var(--mono);font-size:9px;color:var(--muted)}
.badge{display:block;margin:12px auto 0;width:fit-content;padding:5px 22px;font-family:var(--mono);font-size:10px;letter-spacing:3px;border:1.5px solid var(--amber);color:var(--amber);background:var(--amber-lo);transition:all .4s;text-align:center}
.badge.critical{border-color:var(--red);color:var(--red-hi);background:var(--red-lo);animation:flash .9s ease-in-out infinite}
.badge.nominal{border-color:var(--green);color:var(--green-hi);background:var(--green-lo)}
@keyframes flash{50%{opacity:.35}}

/* metric grid */
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border)}
.metric-cell{background:var(--bg);padding:14px 16px}
.mc-label{font-family:var(--mono);font-size:8px;letter-spacing:2px;color:var(--muted);margin-bottom:6px}
.mc-value{font-family:var(--display);font-size:26px;color:var(--text);line-height:1;margin-bottom:5px}
.mc-value.warn{color:var(--yellow)} .mc-value.bad{color:var(--red-hi)} .mc-value.good{color:var(--green-hi)}
.mc-tag{font-family:var(--mono);font-size:8px;letter-spacing:1px;color:var(--muted)}
.bar{height:3px;background:var(--border2);border-radius:2px;overflow:hidden;margin-top:6px}
.bar-fill{height:100%;background:var(--green);border-radius:2px;transition:width .8s cubic-bezier(.4,0,.2,1)}

/* fault */
.fault-block{padding:16px 20px;border-bottom:1px solid var(--border)}
.fault-name{font-family:var(--display);font-size:20px;letter-spacing:2px;color:var(--text);margin-bottom:4px}
.fault-sub{font-family:var(--mono);font-size:9px;color:var(--muted);line-height:1.8}

/* mode tabs */
.mode-tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border)}
.mode-tab{padding:11px 6px;background:var(--bg);font-family:var(--mono);font-size:8px;letter-spacing:1px;color:var(--muted);border:none;cursor:pointer;transition:all .2s;text-align:center;text-transform:uppercase}
.mode-tab:hover{background:var(--surface2);color:var(--text2)}
.mode-tab.on{background:var(--amber-lo);color:var(--amber)}

/* input panels */
.ipanel{padding:16px;border-bottom:1px solid var(--border);display:none}
.ipanel.on{display:block}
.frow{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
.field{display:flex;flex-direction:column;gap:3px}
.field label{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--muted);text-transform:uppercase}
.field input{background:var(--surface);border:1px solid var(--border2);color:var(--text);font-family:var(--mono);font-size:13px;padding:8px 10px;outline:none;transition:border-color .2s;width:100%}
.field input:focus{border-color:var(--amber)}
.go-btn{width:100%;padding:11px;background:var(--amber);border:none;color:#0c0b08;font-family:var(--display);font-size:17px;letter-spacing:3px;cursor:pointer;margin-top:6px;transition:background .15s,transform .1s}
.go-btn:hover{background:var(--amber2)}
.go-btn:active{transform:scale(.99)}

/* live panel */
.live-panel{padding:16px;border-bottom:1px solid var(--border);display:none}
.live-panel.on{display:block}
.lrow{display:flex;justify-content:space-between;align-items:center;font-family:var(--mono);font-size:9px;padding:7px 0;border-bottom:1px solid var(--border)}
.lrow:last-of-type{border:none}
.lrow .lk{color:var(--muted)} .lrow .lv{color:var(--text)}
.live-ping{font-family:var(--mono);font-size:8px;color:var(--muted);margin-top:10px}
.blink{animation:flash 1s step-start infinite;color:var(--amber)}
#ios-wrap{display:none;margin-top:10px}
.ios-btn{width:100%;padding:9px;background:transparent;border:1px solid var(--border2);color:var(--text2);font-family:var(--mono);font-size:9px;letter-spacing:2px;cursor:pointer;transition:all .2s}
.ios-btn:hover{border-color:var(--amber);color:var(--amber)}
.ios-note{font-family:var(--mono);font-size:8px;color:var(--muted);margin-top:6px;line-height:1.7}

/* right column */
.dev-tabs{display:flex;gap:1px;background:var(--border);border-bottom:1px solid var(--border)}
.dev-tab{padding:8px 18px;background:var(--bg);font-family:var(--mono);font-size:8px;letter-spacing:2px;color:var(--muted);border:none;cursor:pointer;transition:all .15s;border-bottom:2px solid transparent}
.dev-tab:hover{background:var(--surface);color:var(--text2)}
.dev-tab.on{background:var(--surface);color:var(--amber);border-bottom-color:var(--amber)}

.recs-block{padding:20px 28px;border-bottom:1px solid var(--border);background:var(--surface)}
.recs-title{font-family:var(--mono);font-size:8px;letter-spacing:3px;color:var(--muted);margin-bottom:12px;text-transform:uppercase}
.recs-list{list-style:none;display:flex;flex-direction:column;gap:9px}
.recs-list li{display:flex;align-items:flex-start;gap:12px;font-size:12.5px;line-height:1.55;color:var(--text)}
.rnum{font-family:var(--mono);font-size:10px;color:var(--amber);min-width:24px;padding-top:1px}

.charts-row{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);flex:1}
.chart-block{background:var(--bg);padding:18px;min-height:160px;display:flex;flex-direction:column}
.chart-title{font-family:var(--mono);font-size:8px;letter-spacing:3px;color:var(--muted);margin-bottom:4px;text-transform:uppercase}
.chart-sub{font-family:var(--mono);font-size:9px;color:var(--text2);margin-bottom:14px}
.spark{flex:1;display:flex;align-items:flex-end;gap:3px}
.spark-bar{flex:1;min-width:4px;border-radius:1px 1px 0 0;transition:height .4s ease}

.meta-row{display:flex;border-top:1px solid var(--border);background:var(--surface)}
.mcell{flex:1;padding:12px 20px;border-right:1px solid var(--border)}
.mcell:last-child{border:none}
.mk{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--muted);margin-bottom:5px;text-transform:uppercase}
.mv{font-family:var(--mono);font-size:12px;color:var(--text);display:flex;align-items:center;gap:6px}
.tag{display:inline-block;padding:2px 8px;font-family:var(--mono);font-size:8px;letter-spacing:1px;border-radius:2px}
.tag-n{background:var(--green-lo);color:var(--green-hi)}
.tag-w{background:var(--amber-lo);color:var(--amber)}
.tag-c{background:var(--red-lo);color:var(--red-hi)}

/* model explainer */
.explainer{padding:12px 20px;border-bottom:1px solid var(--border);background:var(--surface2)}
.exp-title{font-family:var(--mono);font-size:8px;letter-spacing:3px;color:var(--muted);margin-bottom:8px}
.exp-row{display:flex;gap:10px;align-items:center;margin-bottom:6px}
.exp-lbl{font-family:var(--mono);font-size:9px;color:var(--muted);min-width:110px}
.exp-bar-wrap{flex:1;height:6px;background:var(--border2);border-radius:3px;overflow:hidden}
.exp-bar{height:100%;border-radius:3px;transition:width .8s}
.exp-val{font-family:var(--mono);font-size:9px;color:var(--text2);min-width:40px;text-align:right}

.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;padding:60px 32px;gap:14px}
.empty .big{font-family:var(--display);font-size:60px;color:var(--border2);letter-spacing:6px}
.empty p{font-family:var(--mono);font-size:9px;letter-spacing:2px;text-align:center;color:var(--muted);line-height:2.2}

@media(max-width:860px){.shell{grid-template-columns:1fr}.right-col{border-top:2px solid var(--border)}.charts-row{grid-template-columns:1fr}.meta-row{flex-wrap:wrap}}
</style>
</head>
<body>

<header>
  <div class="hdr-brand">
    <span class="hdr-label">Predictive Maintenance</span>
    <span class="hdr-name">NEXUS</span>
  </div>
  <div class="hdr-right">
    <div class="hdr-pill"><span class="dot"></span><span id="conn-lbl">CONNECTING</span></div>
    <div class="hdr-pill">MODEL <span style="color:var(--amber)">ISOLATION FOREST</span></div>
    <div class="hdr-pill">DEVICES <span id="dev-count" style="color:var(--amber)">0</span></div>
    <div class="hdr-time"><div id="clock">--:--:--</div><div id="dline" style="opacity:.4"></div></div>
  </div>
</header>

<div class="shell">

  <div class="left-col">

    <div class="sec">Health Score</div>
    <div class="gauge-wrap">
      <div class="gauge-sub">ISOLATION FOREST · 0 = HEALTHY · 100 = CRITICAL</div>
      <div class="arc-outer">
        <svg viewBox="0 0 190 108" fill="none">
          <path d="M 15 100 A 80 80 0 0 1 175 100" stroke="#2e2b24" stroke-width="9" stroke-linecap="round"/>
          <path id="arc" d="M 15 100 A 80 80 0 0 1 175 100"
                stroke="#f0a830" stroke-width="9" stroke-linecap="round"
                stroke-dasharray="251.2" stroke-dashoffset="251.2"
                style="transition:stroke-dashoffset 1.1s cubic-bezier(.4,0,.2,1),stroke .5s"/>
          <g stroke="#3d3a30" stroke-width="1.2">
            <line x1="15" y1="100" x2="5" y2="100"/>
            <line x1="175" y1="100" x2="185" y2="100"/>
            <line x1="95" y1="20" x2="95" y2="8"/>
            <line x1="52" y1="42" x2="44" y2="34"/>
            <line x1="138" y1="42" x2="146" y2="34"/>
          </g>
          <text x="8" y="118" fill="#6a5e48" font-family="'Share Tech Mono'" font-size="8">HEALTHY</text>
          <text x="148" y="118" fill="#6a5e48" font-family="'Share Tech Mono'" font-size="8">FAULT</text>
        </svg>
        <div class="score-center">
          <div class="score-num" id="score-num">--</div>
          <div class="score-denom">out of 100</div>
        </div>
      </div>
      <span class="badge" id="badge">AWAITING DATA</span>
    </div>

    <div class="metric-grid">
      <div class="metric-cell">
        <div class="mc-label">Battery</div>
        <div class="mc-value" id="mv-bat">--%</div>
        <div class="mc-tag" id="mt-bat">--</div>
        <div class="bar"><div class="bar-fill" id="bf-bat" style="width:0%"></div></div>
      </div>
      <div class="metric-cell">
        <div class="mc-label">CPU Load</div>
        <div class="mc-value" id="mv-cpu">--%</div>
        <div class="mc-tag" id="mt-cpu">--</div>
        <div class="bar"><div class="bar-fill" id="bf-cpu" style="width:0%"></div></div>
      </div>
      <div class="metric-cell">
        <div class="mc-label">Memory</div>
        <div class="mc-value" id="mv-ram">--%</div>
        <div class="mc-tag" id="mt-ram">--</div>
        <div class="bar"><div class="bar-fill" id="bf-ram" style="width:0%"></div></div>
      </div>
      <div class="metric-cell">
        <div class="mc-label">Vibration</div>
        <div class="mc-value" id="mv-vib">-.---</div>
        <div class="mc-tag" id="mt-vib">--</div>
        <div class="bar"><div class="bar-fill" id="bf-vib" style="width:0%"></div></div>
      </div>
    </div>

    <div class="sec">Detected Fault</div>
    <div class="fault-block">
      <div class="fault-name" id="fault-name">AWAITING SCAN</div>
      <div class="fault-sub" id="fault-sub">Run a scan to detect fault type</div>
    </div>

    <div class="sec">Input Mode</div>
    <div class="mode-tabs">
      <button class="mode-tab on" onclick="setMode('live')">⬤ Live</button>
      <button class="mode-tab"    onclick="setMode('ind')">⬡ Industrial</button>
      <button class="mode-tab"    onclick="setMode('custom')">◧ Custom</button>
    </div>

    <div class="live-panel on" id="p-live">
      <div class="lrow"><span class="lk">DEVICE</span>     <span class="lv" id="lv-dev">detecting…</span></div>
      <div class="lrow"><span class="lk">BATTERY</span>    <span class="lv" id="lv-bat">detecting…</span></div>
      <div class="lrow"><span class="lk">VIBRATION</span>  <span class="lv" id="lv-vib">0.000 m/s²</span></div>
      <div class="lrow"><span class="lk">SCAN EVERY</span> <span class="lv">4 seconds</span></div>
      <div class="live-ping">STATUS <span class="blink">▌</span> AUTO-SCANNING</div>
      <div id="ios-wrap">
        <button class="ios-btn" onclick="reqIOSMotion()">▶ GRANT MOTION ACCESS (iPhone)</button>
        <div class="ios-note">iPhone blocks accelerometer until<br>you tap the button above.</div>
      </div>
    </div>

    <div class="ipanel" id="p-ind">
      <div class="frow">
        <div class="field"><label>Vibration m/s²</label><input id="i-vib" type="number" step="0.01" value="0.15"></div>
        <div class="field"><label>Temperature °C</label><input id="i-tmp" type="number" step="1"    value="45"></div>
      </div>
      <div class="frow">
        <div class="field"><label>Power / Battery %</label><input id="i-bat" type="number" step="1" value="85"></div>
        <div class="field"><label>CPU / Node Load %</label><input id="i-cpu" type="number" step="1" value="30"></div>
      </div>
      <div class="frow">
        <div class="field"><label>Memory %</label>       <input id="i-ram" type="number" step="1" value="50"></div>
        <div class="field"><label>Disk / Storage %</label><input id="i-dsk" type="number" step="1" value="40"></div>
      </div>
      <button class="go-btn" onclick="sendInd()">RUN ANALYSIS</button>
    </div>

    <div class="ipanel" id="p-custom">
      <div class="frow">
        <div class="field"><label>Vibration m/s²</label><input id="c-vib" type="number" step="0.01" value="0.5"></div>
        <div class="field"><label>Temperature °C</label><input id="c-tmp" type="number" step="1"    value="70"></div>
      </div>
      <div class="frow">
        <div class="field"><label>Battery %</label><input id="c-bat" type="number" step="1" value="20"></div>
        <div class="field"><label>CPU %</label>    <input id="c-cpu" type="number" step="1" value="90"></div>
      </div>
      <div class="frow">
        <div class="field"><label>Memory %</label> <input id="c-ram" type="number" step="1" value="88"></div>
        <div class="field"><label>Disk %</label>   <input id="c-dsk" type="number" step="1" value="95"></div>
      </div>
      <button class="go-btn" onclick="sendCustom()">RUN ANALYSIS</button>
    </div>

  </div><!-- /left-col -->

  <div class="right-col">
    <div class="dev-tabs" id="dev-tabs"></div>
    <div id="main-area">
      <div class="empty" id="empty-state">
        <div class="big">IDLE</div>
        <p>NO PROBE DATA RECEIVED<br>SELECT A MODE AND RUN A SCAN</p>
      </div>
    </div>
  </div>

</div>

<script>
let selDev  = null, curMode = 'live', liveInt;
let lastVib = 0, batLevel = null, batOK = false;

function tick() {
  const n = new Date();
  document.getElementById('clock').textContent = n.toLocaleTimeString('en-GB');
  document.getElementById('dline').textContent = n.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}).toUpperCase();
}
setInterval(tick,1000); tick();

function devName() {
  const ua = navigator.userAgent;
  if (/iPhone/i.test(ua))  return 'iPhone';
  if (/iPad/i.test(ua))    return 'iPad';
  if (/Android/i.test(ua)) return 'Android';
  if (/Mac OS X/i.test(ua) && !/iPhone|iPad/i.test(ua)) return 'macOS';
  if (/Windows/i.test(ua)) return 'Windows';
  if (/Linux/i.test(ua))   return 'Linux';
  return 'Unknown';
}
function isIOS() { return /iPhone|iPad|iPod/i.test(navigator.userAgent); }

async function initBat() {
  if (isIOS()) { batOK = false; return; }
  try {
    if (navigator.getBattery) {
      const b = await navigator.getBattery();
      batLevel = Math.round(b.level * 100); batOK = true;
      b.addEventListener('levelchange', () => { batLevel = Math.round(b.level * 100); });
    }
  } catch(e) { batOK = false; }
}

function initMotion() {
  if (isIOS()) return;
  if (typeof DeviceMotionEvent !== 'undefined' && DeviceMotionEvent.requestPermission) {
    DeviceMotionEvent.requestPermission().then(s => { if (s==='granted') listenMotion(); }).catch(()=>{});
  } else { listenMotion(); }
}
function listenMotion() {
  window.addEventListener('devicemotion', e => {
    const a = e.acceleration || e.accelerationIncludingGravity;
    if (a && a.x !== null)
      lastVib = Math.round(((Math.abs(a.x||0)+Math.abs(a.y||0)+Math.abs(a.z||0))/30)*1000)/1000;
  });
}
function reqIOSMotion() {
  if (DeviceMotionEvent && DeviceMotionEvent.requestPermission) {
    DeviceMotionEvent.requestPermission().then(s => {
      if (s==='granted') { listenMotion(); document.getElementById('ios-wrap').style.display='none'; }
    });
  }
}

function startLive() { initMotion(); clearInterval(liveInt); doScan(); liveInt = setInterval(doScan, 4000); }

async function doScan() {
  const dn  = devName();
  const bat = (batOK && batLevel !== null) ? batLevel : 85;
  const batD= (batOK && batLevel !== null) ? bat+'%' : (isIOS()?'N/A (iOS)':'detecting…');
  let memEst = 50;
  if (navigator.deviceMemory) memEst = Math.max(20, Math.min(85, Math.round(100 - navigator.deviceMemory * 8)));

  document.getElementById('lv-dev').textContent = dn;
  document.getElementById('lv-bat').textContent = batD;
  document.getElementById('lv-vib').textContent = lastVib.toFixed(3) + ' m/s²';

  await send({
    device_type: dn, battery: bat, vibration: lastVib,
    cpu_load: Math.round(18 + lastVib * 35), memory: memEst,
    temperature: Math.round(30 + (100-bat)*0.22 + lastVib*7), disk_usage: 50,
  });
}

async function sendInd() {
  await send({ device_type:'Industrial',
    vibration: +document.getElementById('i-vib').value||0.15,
    temperature:+document.getElementById('i-tmp').value||45,
    battery:   +document.getElementById('i-bat').value||85,
    cpu_load:  +document.getElementById('i-cpu').value||30,
    memory:    +document.getElementById('i-ram').value||50,
    disk_usage:+document.getElementById('i-dsk').value||40,
  });
}
async function sendCustom() {
  await send({ device_type:'Custom',
    vibration: +document.getElementById('c-vib').value||0.5,
    temperature:+document.getElementById('c-tmp').value||70,
    battery:   +document.getElementById('c-bat').value||20,
    cpu_load:  +document.getElementById('c-cpu').value||90,
    memory:    +document.getElementById('c-ram').value||88,
    disk_usage:+document.getElementById('c-dsk').value||95,
  });
}

async function send(payload) {
  try {
    const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d = await r.json();
    applyGauges(d);
    await refresh();
  } catch(e) { document.getElementById('conn-lbl').textContent='ERROR'; }
}

function applyGauges(d) {
  const sc = d.health_score || 0;

  // arc
  document.getElementById('arc').style.strokeDashoffset = 251.2 - (sc/100)*251.2;
  document.getElementById('arc').style.stroke = sc>=65?'#e04030':sc>=35?'#f0a830':'#5ab870';
  const sn = document.getElementById('score-num');
  sn.textContent = sc;
  sn.style.color = sc>=65?'#ff9080':sc>=35?'#f0a830':'#a0f0b8';

  const b = document.getElementById('badge');
  b.textContent = sc>=65?'CRITICAL — HALT':sc>=35?'WARNING':'NOMINAL';
  b.className   = 'badge '+(sc>=65?'critical':sc>=35?'':'nominal');

  // tiles
  tile('bat', d.battery+'%',
    d.battery<20?'bad':d.battery<40?'warn':'good',
    d.battery<20?'CRITICAL — charge now':d.battery<40?'LOW — drain risk':'HEALTHY',
    d.battery, d.battery<20?'#e04030':d.battery<40?'#e8c040':'#5ab870');

  tile('cpu', Math.round(d.cpu_load)+'%',
    d.cpu_load>90?'bad':d.cpu_load>75?'warn':'good',
    d.cpu_load>90?'SATURATED':d.cpu_load>75?'ELEVATED':'NORMAL',
    d.cpu_load, d.cpu_load>90?'#e04030':d.cpu_load>75?'#e8c040':'#5ab870');

  tile('ram', Math.round(d.memory)+'%',
    d.memory>90?'bad':d.memory>75?'warn':'good',
    d.memory>90?'NEARLY FULL':d.memory>75?'HIGH USAGE':'NORMAL',
    d.memory, d.memory>90?'#e04030':d.memory>75?'#e8c040':'#5ab870');

  const vibPct = Math.min(100,(d.vibration/2)*100);
  tile('vib', d.vibration.toFixed(3),
    d.vibration>0.8?'bad':d.vibration>0.45?'warn':'good',
    d.vib_health||'NORMAL', vibPct,
    d.vibration>0.8?'#e04030':d.vibration>0.45?'#e8c040':'#5ab870');

  document.getElementById('fault-name').textContent = d.fault||'NORMAL';
  document.getElementById('fault-sub').textContent  =
    `IF anomaly component: ${d.if_component??'--'} / 100  ·  Rule penalty: +${d.penalty??0}`;
}

function tile(id, val, cls, tag, pct, color) {
  document.getElementById('mv-'+id).textContent = val;
  document.getElementById('mv-'+id).className   = 'mc-value '+cls;
  document.getElementById('mt-'+id).textContent = tag;
  const bf = document.getElementById('bf-'+id);
  bf.style.width = Math.min(100,pct)+'%';
  bf.style.background = color;
}

async function refresh() {
  try {
    const r = await fetch('/data');
    const d = await r.json();
    const devs  = d.all_devices||{};
    const names = Object.keys(devs);

    document.getElementById('dev-count').textContent = names.length;
    document.getElementById('conn-lbl').textContent  = 'ONLINE';

    document.getElementById('dev-tabs').innerHTML = names.map(n =>
      `<button class="dev-tab ${n===selDev?'on':''}" onclick="selClick('${n}')">${n.toUpperCase()}</button>`
    ).join('');

    if (!names.length) { document.getElementById('empty-state').style.display='flex'; return; }
    document.getElementById('empty-state').style.display='none';

    if (!devs[selDev]) selDev = names.includes(devName())?devName():names[names.length-1];
    const cur  = devs[selDev]||{};
    const hist = ((d.per_dev_hist||{})[selDev])||[];

    applyGauges(cur);
    renderRight(cur, hist);
  } catch(e) {}
}

function selClick(n) { selDev=n; refresh(); }

function renderRight(d, hist) {
  const recs = (d.recommendations||[]).map((r,i) =>
    `<li><span class="rnum">${String(i+1).padStart(2,'0')}.</span>${r}</li>`
  ).join('');

  const mkSpark = (vals, cfn) => vals.length
    ? vals.map(v => `<div class="spark-bar" style="height:${Math.max(4,Math.min(100,v))}%;background:${cfn(v)}"></div>`).join('')
    : '<span style="color:var(--muted);font-family:var(--mono);font-size:9px;padding:4px">COLLECTING DATA…</span>';

  const scoreS = mkSpark(hist, v => v>=65?'#e04030':v>=35?'#f0a830':'#5ab870');
  const cpuS   = mkSpark(hist.map(v=>Math.min(100,10+v*0.45)), ()=>'#f0a830');
  const batVals = hist.map((_,i)=>Math.max(4,Math.min(100,(d.battery||80)-i*0.5)));
  const batS   = mkSpark(batVals, v=>v<20?'#e04030':v<40?'#e8c040':'#5ab870');

  const tv = cl => `tag ${cl}`;
  const vt = d.vibration>0.8?'tag-c':d.vibration>0.45?'tag-w':'tag-n';
  const tt = d.thermal==='OVERHEATING'?'tag-c':(d.thermal==='HOT'||d.thermal==='WARM')?'tag-w':'tag-n';
  const bt = d.bat_trend==='DRAINING'?'tag-w':d.bat_trend==='RISING'?'tag-n':'tag-n';

  // score breakdown bar
  const ifPct   = Math.round((d.if_component||0) * 0.6);
  const rulePct = Math.round(((d.penalty||0)/60*100) * 0.4);

  document.getElementById('main-area').innerHTML = `
    <div class="explainer">
      <div class="exp-title">How This Score Was Calculated</div>
      <div class="exp-row">
        <span class="exp-lbl">IF Anomaly (60%)</span>
        <div class="exp-bar-wrap"><div class="exp-bar" style="width:${Math.min(100,d.if_component||0)}%;background:var(--amber)"></div></div>
        <span class="exp-val">${d.if_component??'--'}/100</span>
      </div>
      <div class="exp-row">
        <span class="exp-lbl">Rule Penalty (40%)</span>
        <div class="exp-bar-wrap"><div class="exp-bar" style="width:${Math.min(100,(d.penalty||0)/60*100)}%;background:var(--red)"></div></div>
        <span class="exp-val">+${d.penalty??0}</span>
      </div>
      <div class="exp-row">
        <span class="exp-lbl">Final Score</span>
        <div class="exp-bar-wrap"><div class="exp-bar" style="width:${d.health_score||0}%;background:${(d.health_score||0)>=65?'#e04030':(d.health_score||0)>=35?'#f0a830':'#5ab870'}"></div></div>
        <span class="exp-val">${d.health_score??'--'}/100</span>
      </div>
    </div>
    <div class="recs-block">
      <div class="recs-title">AI Recommendations — ${d.status||''}</div>
      <ul class="recs-list">${recs}</ul>
    </div>
    <div class="charts-row">
      <div class="chart-block">
        <div class="chart-title">Health Score History</div>
        <div class="chart-sub">Last ${hist.length} scans · ${selDev||''}</div>
        <div class="spark">${scoreS}</div>
      </div>
      <div class="chart-block">
        <div class="chart-title">CPU Load Trend</div>
        <div class="chart-sub">Correlated with anomaly score</div>
        <div class="spark">${cpuS}</div>
      </div>
      <div class="chart-block">
        <div class="chart-title">Battery Projection</div>
        <div class="chart-sub">${d.rul_hours===999?'Stable / Charging':d.rul_hours+'h est. remaining'}</div>
        <div class="spark">${batS}</div>
      </div>
    </div>
    <div class="meta-row">
      <div class="mcell"><div class="mk">Vibration State</div><div class="mv"><span class="tag ${vt}">${d.vib_health||'--'}</span></div></div>
      <div class="mcell"><div class="mk">Thermal State</div><div class="mv"><span class="tag ${tt}">${d.thermal||'--'}</span></div></div>
      <div class="mcell"><div class="mk">Battery Trend</div><div class="mv"><span class="tag ${bt}">${d.bat_trend||'STABLE'}</span></div></div>
      <div class="mcell"><div class="mk">Battery Life Est.</div><div class="mv">${d.rul_hours===999?'∞ / Charging':d.rul_hours+'h'}</div></div>
      <div class="mcell"><div class="mk">Temperature</div><div class="mv">${d.temperature??'--'}°C</div></div>
    </div>`;
}

function setMode(m) {
  curMode = m;
  ['live','ind','custom'].forEach((id,i)=>{
    document.querySelectorAll('.mode-tab')[i].classList.toggle('on',id===m);
    const lp = document.getElementById('p-'+id);
    if (lp) lp.classList.toggle('on',id===m);
  });
  ['ipanel'].forEach(()=>{});
  document.getElementById('p-live').classList.toggle('on',m==='live');
  document.getElementById('p-ind').classList.toggle('on',m==='ind');
  document.getElementById('p-custom').classList.toggle('on',m==='custom');
  if (m==='live') startLive(); else clearInterval(liveInt);
}

// boot
selDev = devName();
if (isIOS()) document.getElementById('ios-wrap').style.display='block';
initBat().then(()=>startLive());
setInterval(refresh, 4000);
refresh();
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
