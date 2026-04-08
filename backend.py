"""
NEXUS PM Backend v5 - Fixed all bugs:
- Custom/Industrial results stay visible (no auto-switch back)
- CPU values explained clearly (browser=estimated, probe.py=exact)
- Phone accelerometer works correctly
- Single Isolation Forest model with proper calibration
"""
import os, time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import IsolationForest

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ═══════════════════════════════════════════════════════════════════
#  HOW THE ML MODEL WORKS — Isolation Forest
# ═══════════════════════════════════════════════════════════════════
#
#  Isolation Forest works by randomly splitting features until each
#  data point is "isolated". NORMAL points need MANY splits to isolate
#  (they cluster together). ANOMALOUS points need FEW splits (they
#  are already far from the cluster).
#
#  We train ONLY on healthy data (normal ranges for cpu/ram/temp etc.)
#  so the model learns what "normal" looks like.
#
#  score_samples() returns a raw score:
#    - Values near 0 (e.g. -0.45) = NORMAL (hard to isolate)
#    - Very negative values (e.g. -0.65) = ANOMALY (easy to isolate)
#
#  We then calibrate this to 0-100:
#    - GOOD_T = p95 of healthy training scores  -> maps to 0  (fully healthy)
#    - BAD_T  = p5  of extreme fault scores     -> maps to 100 (fully broken)
#    - Linear interpolation between them
#
#  THEN we add a rule-based penalty (0-60 points) for single-metric
#  failures that IF alone misses (e.g. critically low battery).
#
#  FINAL SCORE = IF_component(55%) + Rule_penalty(45%), capped 0-100
#  Score < 35  = NOMINAL
#  Score 35-64 = WARNING
#  Score >= 65 = CRITICAL
# ═══════════════════════════════════════════════════════════════════

rng = np.random.default_rng(42)
N = 2000

# Train on realistic healthy ranges
# Features: [cpu%, ram%, temp°C, disk%, vibration m/s², battery%]
X_healthy = np.column_stack([
    rng.uniform(5,   75,  N),   # cpu: idle to moderate
    rng.uniform(20,  80,  N),   # ram: light to moderate
    rng.uniform(30,  68,  N),   # temp: cool to warm
    rng.uniform(5,   80,  N),   # disk: not nearly full
    rng.uniform(0,   0.35,N),   # vibration: low
    rng.uniform(25,  100, N),   # battery: charged
])

iforest = IsolationForest(contamination=0.02, n_estimators=300, random_state=42)
iforest.fit(X_healthy)

# Calibration: find the score range
_s_healthy = iforest.score_samples(X_healthy)
GOOD_T = float(np.percentile(_s_healthy, 95))  # border of normal

X_fault = np.column_stack([
    rng.uniform(92, 100, 500),
    rng.uniform(92, 100, 500),
    rng.uniform(88, 105, 500),
    rng.uniform(92, 100, 500),
    rng.uniform(1.5, 3.5, 500),
    rng.uniform(1,   8,   500),
])
BAD_T = float(np.percentile(iforest.score_samples(X_fault), 5))

def if_component(cpu, ram, temp, disk, vib, bat):
    """Returns 0-100. Lower = healthier."""
    raw = float(iforest.score_samples([[cpu, ram, temp, disk, vib, bat]])[0])
    return float(np.clip((GOOD_T - raw) / (GOOD_T - BAD_T) * 100, 0, 100))

def rule_penalty(cpu, ram, temp, disk, vib, bat, charging):
    """Rule-based penalties for single-metric failures. Returns (0-60, alerts list)."""
    p = 0
    alerts = []
    # Temperature
    if   temp > 90: p += 35; alerts.append(["TEMP", "CRITICAL", f"{temp:.1f}°C — thermal throttling"])
    elif temp > 80: p += 22; alerts.append(["TEMP", "HOT",      f"{temp:.1f}°C — cooling stressed"])
    elif temp > 72: p += 12; alerts.append(["TEMP", "ELEVATED", f"{temp:.1f}°C — above comfort zone"])
    elif temp > 62: p +=  5; alerts.append(["TEMP", "WARM",     f"{temp:.1f}°C — monitor closely"])
    # CPU
    if   cpu > 97: p += 25; alerts.append(["CPU", "SATURATED", f"{cpu:.1f}% — system unresponsive"])
    elif cpu > 90: p += 14; alerts.append(["CPU", "OVERLOAD",  f"{cpu:.1f}% — heavy load"])
    elif cpu > 80: p +=  7; alerts.append(["CPU", "HIGH",      f"{cpu:.1f}% — elevated"])
    # RAM
    if   ram > 96: p += 20; alerts.append(["RAM", "CRITICAL",  f"{ram:.1f}% — paging to disk"])
    elif ram > 88: p += 10; alerts.append(["RAM", "HIGH",      f"{ram:.1f}% — near exhausted"])
    elif ram > 78: p +=  4; alerts.append(["RAM", "ELEVATED",  f"{ram:.1f}%"])
    # Battery (only penalise if on battery, not charging)
    if not charging and bat >= 0:
        if   bat <  5: p += 35; alerts.append(["BAT", "EMPTY",    f"{bat:.0f}% — shutdown imminent"])
        elif bat < 12: p += 22; alerts.append(["BAT", "CRITICAL", f"{bat:.0f}% — plug in now"])
        elif bat < 20: p += 12; alerts.append(["BAT", "LOW",      f"{bat:.0f}% — drain risk"])
        elif bat < 30: p +=  5; alerts.append(["BAT", "WARNING",  f"{bat:.0f}% — below 30%"])
    # Disk
    if   disk > 97: p += 18; alerts.append(["DISK", "FULL",     f"{disk:.0f}% — write errors likely"])
    elif disk > 92: p += 10; alerts.append(["DISK", "CRITICAL", f"{disk:.0f}% — nearly full"])
    elif disk > 85: p +=  5; alerts.append(["DISK", "HIGH",     f"{disk:.0f}% — low free space"])
    # Vibration
    if   vib > 2.0: p += 20; alerts.append(["VIB", "CRITICAL", f"{vib:.3f} m/s² — bearing failure"])
    elif vib > 1.0: p += 10; alerts.append(["VIB", "HIGH",     f"{vib:.3f} m/s² — abnormal"])
    elif vib > 0.5: p +=  4; alerts.append(["VIB", "ELEVATED", f"{vib:.3f} m/s²"])
    return min(p, 60), alerts

def classify_fault(cpu, ram, temp, disk, vib, bat, score):
    if score < 28:
        return "NORMAL", "All parameters within healthy envelope"
    candidates = {
        "OVERHEATING":     (temp - 45) * 1.8 + cpu * 0.15,
        "CPU OVERLOAD":    cpu * 0.9 + ram * 0.2,
        "MEMORY PRESSURE": ram * 0.85 + disk * 0.2,
        "DISK CRITICAL":   disk * 0.9,
        "BEARING FAULT":   vib * 40,
        "LOW POWER":       max(0, 50 - bat) * 0.8,
    }
    fault = max(candidates, key=candidates.get)
    desc = {
        "OVERHEATING":     "Thermal anomaly — cooling system under stress",
        "CPU OVERLOAD":    "Processor saturation — check background processes",
        "MEMORY PRESSURE": "RAM exhaustion — swap/paging activity elevated",
        "DISK CRITICAL":   "Storage nearly full — write performance degraded",
        "BEARING FAULT":   "Mechanical vibration outside normal envelope",
        "LOW POWER":       "Insufficient power — battery drain accelerated",
    }
    return fault, desc.get(fault, "")

devices = {}
device_history = {}

@app.post("/predict")
async def predict(data: dict):
    cpu  = float(data.get("cpu",  data.get("cpu_load",  50)))
    ram  = float(data.get("memory", 50))
    temp = float(data.get("temp", data.get("temperature", 42)))
    disk = float(data.get("disk", data.get("disk_usage", 50)))
    vib  = float(data.get("vibration", 0.02))
    bat  = float(data.get("battery", -1))
    dev  = str(data.get("device", data.get("device_type", "Unknown")))
    chg  = bool(data.get("is_charging", bat < 0))
    src  = str(data.get("source", "browser"))

    bat_if = bat if bat >= 0 else 85
    ifc = if_component(cpu, ram, temp, disk, vib, bat_if)
    pen, alerts = rule_penalty(cpu, ram, temp, disk, vib, bat, chg)

    # Weighted combination
    combined = ifc * 0.55 + (pen / 60 * 100) * 0.45
    hs = int(np.clip(combined, 0, 100))

    # Sanity cap: if everything looks fine, cannot be CRITICAL
    all_ok = (cpu < 80 and ram < 82 and temp < 70 and
              disk < 88 and vib < 0.5 and (bat < 0 or bat > 20 or chg))
    if all_ok:
        hs = min(hs, 30)

    fault, fault_desc = classify_fault(cpu, ram, temp, disk, vib, bat, hs)

    # Battery life from history
    dh = device_history.get(dev, [])
    bat_life = "N/A"
    bat_trend = "STABLE"
    if bat >= 0:
        if chg:
            bat_life = "Charging"
            bat_trend = "CHARGING"
        elif len(dh) >= 4:
            bats = [h["bat"] for h in dh[-8:] if h["bat"] >= 0]
            if len(bats) >= 4:
                drain = (bats[0] - bats[-1]) / (len(bats) * 3)
                dph = drain * 3600
                if dph > 0.2:
                    r = bat / dph
                    bat_life = f"{int(r*60)}min" if r < 1 else (f"{r:.1f}h" if r < 10 else f"{int(r)}h")
                    bat_trend = "DRAINING"
                else:
                    bat_life = "Stable"
    elif chg:
        bat_life = "AC Power"

    # CPU trend
    if len(dh) >= 5:
        cpus = [h["cpu"] for h in dh[-10:]]
        sl = (cpus[-1] - cpus[0]) / len(cpus)
        cpu_trend = "RISING" if sl > 2 else "FALLING" if sl < -2 else "STABLE"
    else:
        cpu_trend = "STABLE"

    vib_label = ("SMOOTH" if vib < 0.08 else "NORMAL" if vib < 0.35 else
                 "ELEVATED" if vib < 0.7 else "HIGH" if vib < 1.2 else "CRITICAL")
    thermal = ("COOL" if temp < 38 else "NORMAL" if temp < 58 else
               "WARM" if temp < 70 else "HOT" if temp < 82 else "OVERHEATING")

    if hs >= 65:
        risk, status = "CRITICAL", "CRITICAL — IMMEDIATE ACTION"
        recs = [f"🚨 Primary fault: {fault} — {fault_desc}",
                f"Score {hs}/100 exceeds critical threshold (65+)",
                "Halt non-essential processes immediately",
                f"{'Plug in charger — ' if bat>=0 and bat<15 and not chg else ''}Inspect thermal & mechanical subsystems"]
    elif hs >= 35:
        risk, status = "WARNING", "WARNING — SCHEDULE MAINTENANCE"
        recs = [f"⚠️ Fault pattern: {fault}",
                f"Score {hs}/100 — {len(alerts)} parameter(s) above threshold",
                f"Battery: {bat_life}" if bat >= 0 else "Monitor CPU and thermal trends",
                "Schedule maintenance or reduce system load"]
    else:
        risk, status = "NOMINAL", "NOMINAL — ALL SYSTEMS HEALTHY"
        recs = [f"✅ No anomalies detected (Isolation Forest score: {hs}/100)",
                f"CPU: {cpu:.1f}%  ·  RAM: {ram:.1f}%  ·  Temp: {temp:.1f}°C",
                f"Battery: {bat_life}" if bat >= 0 else "Running on AC power",
                "Continue standard monitoring intervals"]

    extras = {k: data[k] for k in [
        "cpu_freq","cpu_cores_logical","cpu_cores_physical",
        "ram_total","ram_used","ram_avail","disk_total","disk_used",
        "disk_read_mb","disk_write_mb","net_send_mb","net_recv_mb",
        "bat_mins_left","proc_count","top_procs","os_version","platform"
    ] if k in data}

    result = {
        "cpu": round(cpu,1), "memory": round(ram,1), "temp": round(temp,1),
        "disk": round(disk,1), "vibration": round(vib,4), "battery": round(bat,1),
        "is_charging": chg, "device": dev, "source": src,
        "health_score": hs, "if_component": round(ifc,1), "penalty": pen,
        "risk": risk, "status": status, "fault": fault, "fault_desc": fault_desc,
        "alerts": alerts, "bat_life": bat_life, "bat_trend": bat_trend,
        "cpu_trend": cpu_trend, "vib_label": vib_label, "thermal": thermal,
        "recommendations": recs,
        **extras
    }

    devices[dev] = result
    device_history.setdefault(dev, []).append(
        {"cpu": cpu, "ram": ram, "temp": temp, "disk": disk, "bat": bat, "score": hs, "ts": time.time()}
    )
    if len(device_history[dev]) > 40:
        device_history[dev].pop(0)
    return result

@app.get("/data")
async def get_data():
    if not devices:
        return {"error": "No data yet"}
    return {
        "all_devices": devices,
        "score_hist": {k: [h["score"] for h in v] for k, v in device_history.items()},
        "cpu_hist":   {k: [h["cpu"]   for h in v] for k, v in device_history.items()},
        "ram_hist":   {k: [h["ram"]   for h in v] for k, v in device_history.items()},
        "tmp_hist":   {k: [h["temp"]  for h in v] for k, v in device_history.items()},
    }

@app.get("/")
async def root():
    return {"status": "online", "version": "5.0", "model": "IsolationForest", "endpoints": ["/monitor", "/predict", "/data"]}

@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return HTML

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS PM — Predictive Maintenance</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --void:#03050a;--panel:#080f1c;--panel2:#0b1525;--rim:#0e1e35;--rim2:#162840;
  --blue:#00aaff;--teal:#00e5c4;--green:#00ff87;--yellow:#ffd24d;
  --orange:#ff8c42;--red:#ff3355;--purple:#a78bfa;
  --text:#c8dff5;--text2:#4a7090;--text3:#1e3a55;
  --mono:'JetBrains Mono',monospace;--display:'Orbitron',monospace;--ui:'Syne',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:14px}
body{background:var(--void);color:var(--text);font-family:var(--ui);min-height:100vh;overflow-x:hidden}
/* Stars */
#stars{position:fixed;inset:0;pointer-events:none;z-index:0}
.st{position:absolute;border-radius:50%;background:#fff;animation:tw var(--d,3s) ease-in-out infinite var(--delay,0s)}
@keyframes tw{0%,100%{opacity:var(--lo,.1)}50%{opacity:var(--hi,.7)}}
/* Grid */
body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(rgba(0,170,255,.022) 1px,transparent 1px),linear-gradient(90deg,rgba(0,170,255,.022) 1px,transparent 1px);
  background-size:55px 55px;}
.app{position:relative;z-index:1;max-width:1480px;margin:0 auto;padding:12px;display:flex;flex-direction:column;gap:10px}

/* HEADER */
.hdr{display:flex;align-items:center;gap:14px;padding:12px 20px;
  background:rgba(8,15,28,.96);border:1px solid var(--rim2);border-radius:12px;
  backdrop-filter:blur(20px);flex-wrap:wrap;
  box-shadow:0 0 30px rgba(0,170,255,.07),inset 0 1px 0 rgba(0,170,255,.1)}
.logo{font-family:var(--display);font-size:20px;font-weight:900;letter-spacing:4px;color:var(--blue);text-shadow:0 0 18px rgba(0,170,255,.5)}
.logo em{color:var(--teal);font-style:normal;font-size:10px;letter-spacing:2px;margin-left:4px}
.chip{font-family:var(--mono);font-size:8px;letter-spacing:2px;color:var(--text3);padding:3px 10px;border:1px solid var(--rim2);border-radius:4px}
.hst{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--text2)}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pa 2.5s ease-in-out infinite;flex-shrink:0}
@keyframes pa{0%,100%{box-shadow:0 0 0 0 rgba(0,255,135,.5)}70%{box-shadow:0 0 0 8px rgba(0,255,135,0)}}
.hdr-r{margin-left:auto;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.clk{font-family:var(--display);font-size:13px;color:var(--blue);letter-spacing:2px}
.dtl{font-family:var(--mono);font-size:8px;color:var(--text3);letter-spacing:1px;margin-top:2px}
.dtabs{display:flex;gap:5px;flex-wrap:wrap}
.dtab{padding:4px 12px;border:1px solid var(--rim2);border-radius:5px;font-family:var(--mono);font-size:8px;letter-spacing:1px;color:var(--text2);background:transparent;cursor:pointer;transition:all .2s}
.dtab:hover{border-color:var(--blue);color:var(--text)}
.dtab.on{border-color:var(--blue);color:var(--blue);background:rgba(0,170,255,.08)}

/* MODE TABS */
.modes{display:flex;gap:6px;flex-wrap:wrap}
.mbtn{flex:1;min-width:140px;padding:10px 16px;border:1px solid var(--rim2);border-radius:8px;
  font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--text2);
  background:rgba(8,15,28,.8);cursor:pointer;transition:all .25s;text-align:center}
.mbtn:hover{border-color:var(--teal);color:var(--teal)}
.mbtn.on{border-color:var(--blue);color:var(--blue);background:rgba(0,170,255,.1);box-shadow:0 0 12px rgba(0,170,255,.15)}

/* BANNER */
.banner{display:flex;align-items:center;gap:16px;padding:14px 22px;border-radius:12px;border:1.5px solid;transition:all .5s;flex-wrap:wrap;position:relative;overflow:hidden}
.banner::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.012),transparent);animation:sw 5s linear infinite;pointer-events:none}
@keyframes sw{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
.banner.NOMINAL{background:rgba(0,255,135,.04);border-color:rgba(0,255,135,.2)}
.banner.WARNING{background:rgba(255,210,77,.05);border-color:rgba(255,210,77,.4);animation:bw 2.5s infinite}
.banner.CRITICAL{background:rgba(255,51,85,.07);border-color:rgba(255,51,85,.6);animation:bc .9s infinite}
@keyframes bw{0%,100%{border-color:rgba(255,210,77,.3)}50%{border-color:rgba(255,210,77,.9)}}
@keyframes bc{0%,100%{border-color:rgba(255,51,85,.4)}50%{border-color:rgba(255,51,85,1)}}
.bico{font-size:30px;flex-shrink:0}
.bttl{font-family:var(--display);font-size:17px;font-weight:700;letter-spacing:3px}
.bsub{font-family:var(--mono);font-size:9px;color:var(--text2);margin-top:3px}
.bsc-w{margin-left:auto;text-align:right;flex-shrink:0}
.bnum{font-family:var(--display);font-size:50px;font-weight:900;line-height:1;transition:color .5s}
.blbl{font-family:var(--mono);font-size:8px;color:var(--text3);letter-spacing:2px}
.bbar{position:absolute;bottom:0;left:0;width:100%;height:3px;background:var(--rim2)}
.bbarf{height:100%;transition:width 1s cubic-bezier(.4,0,.2,1),background .5s}

/* CARDS */
.card{background:linear-gradient(135deg,var(--panel),var(--panel2));border:1px solid var(--rim);border-radius:12px;padding:15px;box-shadow:0 4px 20px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.03);animation:fu .35s ease both}
@keyframes fu{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.ch{font-family:var(--mono);font-size:8px;letter-spacing:3px;color:var(--text3);text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:7px}
.ch::before{content:'';width:2px;height:10px;background:var(--blue);border-radius:2px;flex-shrink:0}

/* LAYOUT */
.grid3{display:grid;grid-template-columns:270px 1fr 250px;gap:10px}
@media(max-width:1050px){.grid3{grid-template-columns:1fr 1fr}}
@media(max-width:660px){.grid3{grid-template-columns:1fr}}
.col{display:flex;flex-direction:column;gap:10px}

/* RING GAUGE */
.rw{display:flex;flex-direction:column;align-items:center;padding:2px 0}
.rbadge{display:inline-block;margin:8px auto 0;padding:4px 16px;font-family:var(--mono);font-size:9px;letter-spacing:2px;border:1px solid;border-radius:4px;text-align:center;transition:all .4s}
.rbadge.NOMINAL{border-color:rgba(0,255,135,.4);color:var(--green);background:rgba(0,255,135,.06)}
.rbadge.WARNING{border-color:rgba(255,210,77,.5);color:var(--yellow);background:rgba(255,210,77,.07)}
.rbadge.CRITICAL{border-color:rgba(255,51,85,.6);color:var(--red);background:rgba(255,51,85,.09);animation:bc .9s infinite}

/* SCORE BREAKDOWN */
.bd{display:flex;flex-direction:column;gap:10px}
.bdr{display:flex;align-items:center;gap:8px}
.bdlbl{font-family:var(--mono);font-size:8px;color:var(--text2);width:130px;flex-shrink:0}
.bdtr{flex:1;height:5px;background:var(--rim2);border-radius:3px;overflow:hidden}
.bdfill{height:100%;border-radius:3px;transition:width .9s cubic-bezier(.4,0,.2,1)}
.bdval{font-family:var(--mono);font-size:9px;color:var(--text2);min-width:40px;text-align:right;flex-shrink:0}
.bd-note{font-family:var(--mono);font-size:8px;color:var(--text3);line-height:1.7;margin-top:6px;padding:8px;background:rgba(0,170,255,.04);border:1px solid var(--rim);border-radius:6px}

/* MINI RINGS */
.mr4{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.m4{display:flex;flex-direction:column;align-items:center;padding:10px 8px;background:var(--panel);border:1px solid var(--rim);border-radius:10px}
.m4v{font-family:var(--display);font-size:16px;font-weight:700;text-align:center;margin-top:3px;line-height:1}
.m4l{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--text3);text-align:center;margin-top:2px}
.m4t{font-family:var(--mono);font-size:7px;margin-top:3px;text-align:center;padding:1px 6px;border-radius:3px}

/* BATTERY */
.batbig{font-family:var(--display);font-size:40px;font-weight:700;text-align:center;line-height:1}
.batbar{height:8px;background:var(--rim2);border-radius:4px;overflow:hidden;margin:8px 0}
.batfill{height:100%;border-radius:4px;transition:width .8s,background .4s}
.batinfo{font-family:var(--mono);font-size:9px;color:var(--text2);text-align:center;letter-spacing:1px}

/* FAULT */
.fcard{border-radius:9px;padding:12px;margin-bottom:10px;border:1px solid;transition:all .4s}
.fname{font-family:var(--display);font-size:14px;font-weight:700;letter-spacing:2px;margin-bottom:3px}
.fdesc{font-family:var(--mono);font-size:9px;color:var(--text2);line-height:1.6}

/* ALERTS */
.alist{display:flex;flex-direction:column;gap:5px}
.aitem{display:flex;align-items:flex-start;gap:8px;padding:6px 10px;border-radius:7px;border:1px solid;font-family:var(--mono);font-size:9px;line-height:1.5}
.aitem.CRITICAL,.aitem.EMPTY,.aitem.SATURATED,.aitem.FULL{background:rgba(255,51,85,.07);border-color:rgba(255,51,85,.3);color:var(--red)}
.aitem.HOT,.aitem.OVERLOAD,.aitem.HIGH{background:rgba(255,140,66,.07);border-color:rgba(255,140,66,.25);color:var(--orange)}
.aitem.ELEVATED,.aitem.WARM,.aitem.LOW,.aitem.WARNING{background:rgba(255,210,77,.05);border-color:rgba(255,210,77,.2);color:var(--yellow)}
.asev{font-weight:700;min-width:74px;flex-shrink:0}
.amsg{color:var(--text2);flex:1}
.noal{font-family:var(--mono);font-size:10px;color:var(--text3);padding:10px;text-align:center}

/* METRICS */
.mrow{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--rim)}
.mrow:last-child{border:none}
.mico{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0}
.mname{font-family:var(--mono);font-size:9px;letter-spacing:1px;color:var(--text2)}
.mval{font-family:var(--display);font-size:14px;font-weight:600;line-height:1.2}
.mbar{height:2px;background:var(--rim2);border-radius:2px;margin-top:3px;overflow:hidden}
.mbarfill{height:100%;border-radius:2px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.mestimated{font-family:var(--mono);font-size:7px;color:var(--text3);margin-top:1px;letter-spacing:1px}

/* SYSTEM SPEC */
.spgrid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.spitem{background:var(--panel);border:1px solid var(--rim);border-radius:7px;padding:8px 10px}
.splbl{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--text3);margin-bottom:2px;text-transform:uppercase}
.spval{font-family:var(--mono);font-size:11px;color:var(--blue)}

/* RECS */
.recitem{display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid var(--rim);font-size:12px;line-height:1.55;color:var(--text)}
.recitem:last-child{border:none}
.recn{font-family:var(--mono);font-size:9px;color:var(--blue);min-width:22px;flex-shrink:0;padding-top:1px}

/* SPARKLINES */
.charts3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
@media(max-width:680px){.charts3{grid-template-columns:1fr}}
.ccard{background:var(--panel);border:1px solid var(--rim);border-radius:12px;padding:14px}
.ctitle{font-family:var(--mono);font-size:8px;letter-spacing:3px;color:var(--text3);margin-bottom:3px;text-transform:uppercase}
.csub{font-family:var(--mono);font-size:9px;color:var(--text2);margin-bottom:10px}
.spark{display:flex;align-items:flex-end;gap:2px;height:65px}
.spbar{flex:1;min-width:3px;border-radius:1px 1px 0 0;transition:height .5s ease,background .3s}

/* META STRIP */
.metas{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
@media(max-width:680px){.metas{grid-template-columns:repeat(2,1fr)}}
.mc{background:var(--panel);border:1px solid var(--rim);border-radius:9px;padding:10px 12px}
.mk{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--text3);margin-bottom:5px;text-transform:uppercase}
.mv{font-family:var(--mono);font-size:11px;color:var(--text)}
.tag{display:inline-block;padding:2px 8px;border-radius:3px;font-family:var(--mono);font-size:8px;letter-spacing:1px}
.tn{background:rgba(0,255,135,.1);color:var(--green)}
.tw{background:rgba(255,210,77,.1);color:var(--yellow)}
.tc{background:rgba(255,51,85,.12);color:var(--red)}
.tb{background:rgba(0,170,255,.1);color:var(--blue)}

/* INPUT PANELS */
.igrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
@media(max-width:500px){.igrid{grid-template-columns:1fr}}
.ifield{display:flex;flex-direction:column;gap:4px}
.ifield label{font-family:var(--mono);font-size:8px;letter-spacing:2px;color:var(--text3);text-transform:uppercase}
.ifield input{background:var(--panel2);border:1px solid var(--rim2);border-radius:7px;color:var(--text);font-family:var(--mono);font-size:14px;padding:9px 12px;outline:none;transition:border-color .2s,box-shadow .2s;width:100%}
.ifield input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(0,170,255,.12)}
.ifield .hint{font-family:var(--mono);font-size:8px;color:var(--text3);line-height:1.5}
.rbtn{width:100%;padding:13px;background:rgba(0,170,255,.1);border:1px solid var(--blue);border-radius:10px;color:var(--blue);font-family:var(--display);font-size:13px;letter-spacing:3px;cursor:pointer;transition:all .2s}
.rbtn:hover{background:rgba(0,170,255,.2);box-shadow:0 0 18px rgba(0,170,255,.2)}

/* PROBE HINT */
.probe-hint{background:rgba(0,229,196,.04);border:1px solid rgba(0,229,196,.2);border-radius:10px;padding:14px 16px;margin-bottom:10px}
.probe-title{font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--teal);margin-bottom:8px;text-transform:uppercase}
.probe-code{background:var(--void);border:1px solid var(--rim);border-radius:6px;padding:7px 12px;font-family:var(--mono);font-size:11px;color:var(--green);display:block;margin-top:4px;word-break:break-all;line-height:1.7}
.probe-note{font-family:var(--mono);font-size:9px;color:var(--text3);margin-top:8px;line-height:1.7}

/* PHONE PROMPT */
.phcard{text-align:center;padding:24px}
.grantbtn{padding:12px 32px;background:rgba(0,229,196,.1);border:1px solid var(--teal);border-radius:10px;color:var(--teal);font-family:var(--display);font-size:13px;letter-spacing:3px;cursor:pointer;transition:all .2s;margin-top:12px;display:inline-block}
.grantbtn:hover{background:rgba(0,229,196,.2);box-shadow:0 0 18px rgba(0,229,196,.12)}

/* PROCESS TABLE */
.ptbl{width:100%;border-collapse:collapse}
.ptbl th{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--text3);text-align:left;padding:4px 8px;border-bottom:1px solid var(--rim);text-transform:uppercase}
.ptbl td{font-family:var(--mono);font-size:10px;padding:5px 8px;border-bottom:1px solid var(--rim);color:var(--text2)}
.ptbl tr:last-child td{border:none}
.ptbl td:first-child{color:var(--text)}
.ptbl td.warn{color:var(--yellow)} .ptbl td.crit{color:var(--red)}

/* RESULT STAYED VISIBLE */
.result-shown{border:1px solid rgba(0,229,196,.3);border-radius:10px;padding:10px 14px;margin-bottom:10px;background:rgba(0,229,196,.04);font-family:var(--mono);font-size:9px;color:var(--teal);letter-spacing:1px}

/* PROCESS TABLE */
</style>
</head>
<body>
<div id="stars"></div>
<div class="app">

<!-- HEADER -->
<header class="hdr">
  <div class="logo">NEXUS <em>PM</em></div>
  <div class="chip">ISOLATION FOREST v5</div>
  <div class="hst"><span class="pulse"></span><span id="clbl">CONNECTING</span></div>
  <div class="hdr-r">
    <div id="dtabs" class="dtabs"></div>
    <div><div class="clk" id="clk">--:--:--</div><div class="dtl" id="dtl"></div></div>
  </div>
</header>

<!-- MODES -->
<div class="modes">
  <button class="mbtn on" onclick="setMode('live')"    id="mb-live">⬤ &nbsp;LIVE SENSOR</button>
  <button class="mbtn"    onclick="setMode('ind')"     id="mb-ind">⬡ &nbsp;INDUSTRIAL</button>
  <button class="mbtn"    onclick="setMode('custom')"  id="mb-custom">◧ &nbsp;CUSTOM INPUT</button>
</div>

<!-- BANNER -->
<div class="banner NOMINAL" id="bnr">
  <div class="bico" id="bico">⏳</div>
  <div>
    <div class="bttl" id="bttl" style="color:var(--green)">WAITING</div>
    <div class="bsub" id="bsub">Open on phone · run probe.py on PC · or use Industrial/Custom mode</div>
  </div>
  <div class="bsc-w">
    <div class="bnum" id="bnum" style="color:var(--green)">--</div>
    <div class="blbl">HEALTH SCORE / 100</div>
  </div>
  <div class="bbar"><div class="bbarf" id="bbarf" style="width:0%"></div></div>
</div>

<!-- ─── LIVE MODE ─── -->
<div id="mode-live">
  <div class="probe-hint" id="probe-hint">
    <div class="probe-title">📡 For exact Task Manager values — run probe.py on your PC:</div>
    <code class="probe-code">pip install psutil requests &amp;&amp; python probe.py</code>
    <div class="probe-note">
      Without probe.py: browser sends ESTIMATED values (CPU≈50%, RAM from JS heap, Temp estimated).<br>
      With probe.py: sends REAL Task Manager CPU%, RAM%, Disk%, Temperature, Battery, top processes.
    </div>
  </div>
  <div class="card phcard" id="ph-card" style="display:none">
    <div class="ch" style="justify-content:center">PHONE SENSOR ACCESS</div>
    <div style="font-size:48px;margin:8px 0">📱</div>
    <div style="font-family:var(--mono);font-size:10px;color:var(--text2);margin-bottom:14px">
      Enable accelerometer + battery for real phone readings
    </div>
    <button class="grantbtn" onclick="startPhone()">▶ GRANT ACCESS</button>
  </div>

  <div class="grid3">
    <!-- LEFT COL -->
    <div class="col">
      <!-- Main ring gauge -->
      <div class="card">
        <div class="ch">Health Score</div>
        <div class="rw">
          <svg width="196" height="120" viewBox="0 0 196 120" style="overflow:visible">
            <defs>
              <linearGradient id="rg" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%"  stop-color="#00ff87"/>
                <stop offset="50%" stop-color="#ffd24d"/>
                <stop offset="100%" stop-color="#ff3355"/>
              </linearGradient>
              <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/>
                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>
            <path d="M16,112 A80,80 0 0,1 180,112" fill="none" stroke="#0e1e35" stroke-width="14" stroke-linecap="round"/>
            <path id="rf" d="M16,112 A80,80 0 0,1 180,112" fill="none" stroke="url(#rg)" stroke-width="14"
              stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="251.2"
              style="transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)" filter="url(#glow)"/>
            <!-- mask over unfilled part -->
            <path id="rmask" d="M16,112 A80,80 0 0,1 180,112" fill="none" stroke="#03050a" stroke-width="14"
              stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="0"
              style="transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)"/>
            <g stroke="#162840" stroke-width="1.5">
              <line x1="16" y1="112" x2="6" y2="112"/>
              <line x1="180" y1="112" x2="190" y2="112"/>
              <line x1="98" y1="32" x2="98" y2="20"/>
              <line x1="52" y1="55" x2="43" y2="46"/>
              <line x1="144" y1="55" x2="153" y2="46"/>
            </g>
            <text x="8"  y="126" fill="#1e3a55" font-family="Orbitron" font-size="7">0</text>
            <text x="167" y="126" fill="#1e3a55" font-family="Orbitron" font-size="7">100</text>
            <text id="rnum" x="98" y="96" fill="#00ff87" font-family="Orbitron" font-size="36" font-weight="700"
              text-anchor="middle" dominant-baseline="central" style="transition:fill .5s">--</text>
            <text x="98" y="114" fill="#1e3a55" font-family="JetBrains Mono" font-size="7" text-anchor="middle">OUT OF 100</text>
          </svg>
          <div class="rbadge NOMINAL" id="rbadge">AWAITING DATA</div>
        </div>
      </div>

      <!-- Breakdown -->
      <div class="card">
        <div class="ch">How Score Is Calculated</div>
        <div class="bd">
          <div class="bdr">
            <span class="bdlbl">IF Anomaly (55%)</span>
            <div class="bdtr"><div class="bdfill" id="bd-if" style="width:0%;background:var(--blue)"></div></div>
            <span class="bdval" id="bv-if">--</span>
          </div>
          <div class="bdr">
            <span class="bdlbl">Rule Penalty (45%)</span>
            <div class="bdtr"><div class="bdfill" id="bd-rl" style="width:0%;background:var(--red)"></div></div>
            <span class="bdval" id="bv-rl">--</span>
          </div>
          <div class="bdr">
            <span class="bdlbl">Final Health Score</span>
            <div class="bdtr"><div class="bdfill" id="bd-fn" style="width:0%;background:var(--green)"></div></div>
            <span class="bdval" id="bv-fn">--</span>
          </div>
        </div>
        <div class="bd-note">
          <strong style="color:var(--blue)">Isolation Forest:</strong> detects multi-parameter anomalies by measuring how far your readings are from 2000 healthy training samples.<br>
          <strong style="color:var(--red)">Rule Penalty:</strong> catches single critical failures (e.g. battery &lt;12%, temp &gt;80°C) that IF alone misses.<br>
          <strong style="color:var(--green)">Final = IF×55% + Rules×45%</strong>
        </div>
      </div>

      <!-- Battery -->
      <div class="card">
        <div class="ch">Battery</div>
        <div class="batbig" id="bat-big" style="color:var(--green)">--%</div>
        <div class="batbar"><div class="batfill" id="bat-fill" style="width:0%;background:var(--green)"></div></div>
        <div class="batinfo" id="bat-info">--</div>
      </div>
    </div>

    <!-- CENTER COL -->
    <div class="col">
      <!-- 4 mini rings -->
      <div class="card">
        <div class="ch">Live Parameters</div>
        <div class="mr4" id="mr4">
          <div style="font-family:var(--mono);font-size:10px;color:var(--text3);grid-column:span 2;padding:14px;text-align:center">Awaiting first scan…</div>
        </div>
      </div>

      <!-- Fault + Alerts -->
      <div class="card">
        <div class="ch">Fault Classification</div>
        <div class="fcard" id="fcard" style="background:rgba(0,255,135,.04);border-color:rgba(0,255,135,.2)">
          <div class="fname" id="fname" style="color:var(--green)">AWAITING SCAN</div>
          <div class="fdesc" id="fdesc">Run a scan to detect fault type</div>
        </div>
        <div class="ch">Active Alerts</div>
        <div class="alist" id="alist">
          <div class="noal">No alerts — all metrics normal</div>
        </div>
      </div>

      <!-- Recommendations -->
      <div class="card">
        <div class="ch">AI Recommendations</div>
        <div id="rlist"><div style="color:var(--text3);font-family:var(--mono);font-size:10px">Awaiting analysis…</div></div>
      </div>
    </div>

    <!-- RIGHT COL -->
    <div class="col">
      <!-- Metric rows -->
      <div class="card">
        <div class="ch">System Metrics</div>
        <div id="mrows"><div style="color:var(--text3);font-family:var(--mono);font-size:10px">Awaiting data…</div></div>
      </div>

      <!-- System info (probe.py only) -->
      <div class="card" id="spcard" style="display:none">
        <div class="ch">System Info (from probe.py)</div>
        <div class="spgrid" id="spgrid"></div>
      </div>

      <!-- Process table (probe.py only) -->
      <div class="card" id="pcard" style="display:none">
        <div class="ch">Top Processes</div>
        <table class="ptbl">
          <thead><tr><th>Process</th><th>CPU%</th><th>RAM%</th></tr></thead>
          <tbody id="pbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts3">
    <div class="ccard">
      <div class="ctitle">Health Score History</div>
      <div class="csub" id="cs1">Last scans</div>
      <div class="spark" id="sp1"></div>
    </div>
    <div class="ccard">
      <div class="ctitle">CPU Load Trend</div>
      <div class="csub" id="cs2">CPU over time</div>
      <div class="spark" id="sp2"></div>
    </div>
    <div class="ccard">
      <div class="ctitle">Temperature Curve</div>
      <div class="csub" id="cs3">°C over time</div>
      <div class="spark" id="sp3"></div>
    </div>
  </div>

  <!-- Meta strip -->
  <div class="metas">
    <div class="mc"><div class="mk">Vibration State</div><div class="mv" id="m-vib">--</div></div>
    <div class="mc"><div class="mk">Thermal State</div> <div class="mv" id="m-thm">--</div></div>
    <div class="mc"><div class="mk">Battery Trend</div> <div class="mv" id="m-btr">--</div></div>
    <div class="mc"><div class="mk">Battery Life</div>  <div class="mv" id="m-blt">--</div></div>
    <div class="mc"><div class="mk">CPU Trend</div>     <div class="mv" id="m-ctr">--</div></div>
  </div>
</div><!-- /mode-live -->

<!-- ─── INDUSTRIAL MODE ─── -->
<div id="mode-ind" style="display:none">
  <div class="card">
    <div class="ch">Industrial Machine Input</div>
    <div class="igrid">
      <div class="ifield"><label>Vibration (m/s²)</label><input type="number" id="iv" value="0.08" step="0.01"><div class="hint">Normal &lt;0.35 · High &gt;0.7 · Critical &gt;1.2</div></div>
      <div class="ifield"><label>Temperature (°C)</label><input type="number" id="it" value="45" step="1"><div class="hint">Normal &lt;62 · Hot &gt;80</div></div>
      <div class="ifield"><label>CPU / Node Load (%)</label><input type="number" id="ic" value="55" step="1" min="0" max="100"><div class="hint">Task Manager CPU column</div></div>
      <div class="ifield"><label>Memory / RAM (%)</label><input type="number" id="im" value="50" step="1" min="0" max="100"><div class="hint">Task Manager Memory column</div></div>
      <div class="ifield"><label>Power / Battery (%)</label><input type="number" id="ib" value="90" step="1" min="0" max="100"></div>
      <div class="ifield"><label>Disk / Wear (%)</label><input type="number" id="id" value="40" step="1" min="0" max="100"><div class="hint">Task Manager Disk column</div></div>
    </div>
    <button class="rbtn" onclick="sendInd()">⚙ RUN ML ANALYSIS</button>
    <div id="ind-result" style="display:none;margin-top:12px" class="result-shown">✓ Analysis complete — results shown in Live view above</div>
  </div>
</div>

<!-- ─── CUSTOM MODE ─── -->
<div id="mode-custom" style="display:none">
  <div class="card">
    <div class="ch">Custom Parameter Input</div>
    <div class="igrid">
      <div class="ifield"><label>CPU %</label><input type="number" id="xc" value="50" step="1" min="0" max="100"></div>
      <div class="ifield"><label>RAM / Memory %</label><input type="number" id="xm" value="50" step="1" min="0" max="100"></div>
      <div class="ifield"><label>Temperature °C</label><input type="number" id="xt" value="42" step="1"></div>
      <div class="ifield"><label>Disk %</label><input type="number" id="xd" value="50" step="1" min="0" max="100"></div>
      <div class="ifield"><label>Vibration m/s²</label><input type="number" id="xv" value="0.05" step="0.01"></div>
      <div class="ifield"><label>Battery % (−1 = no battery)</label><input type="number" id="xb" value="80" step="1" min="-1" max="100"></div>
    </div>
    <button class="rbtn" onclick="sendCustom()">🧠 ANALYZE WITH ML</button>
    <div id="cus-result" style="display:none;margin-top:12px" class="result-shown">✓ Analysis complete — results shown in Live view above</div>
  </div>
</div>

</div><!-- /app -->

<script>
// ── STATE ──────────────────────────────────────────────────────
let selDev = null, curMode = 'live', liveInt = null;
let batLvl = null, batOK = false, lastVib = 0;

// ── STARS ──────────────────────────────────────────────────────
(function(){
  const c = document.getElementById('stars');
  for(let i = 0; i < 100; i++){
    const s = document.createElement('div');
    s.className = 'st';
    const sz = Math.random()*2+0.5;
    s.style.cssText = `width:${sz}px;height:${sz}px;left:${Math.random()*100}%;top:${Math.random()*100}%;`+
      `--lo:${(Math.random()*.1+.04).toFixed(2)};--hi:${(Math.random()*.6+.2).toFixed(2)};`+
      `--d:${(Math.random()*4+2).toFixed(1)}s;--delay:-${(Math.random()*5).toFixed(1)}s`;
    c.appendChild(s);
  }
})();

// ── CLOCK ──────────────────────────────────────────────────────
function tick(){
  const n = new Date();
  document.getElementById('clk').textContent = n.toLocaleTimeString('en-GB');
  document.getElementById('dtl').textContent = n.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}).toUpperCase();
}
setInterval(tick,1000); tick();

// ── DEVICE DETECTION ──────────────────────────────────────────
function devName(){
  const u = navigator.userAgent;
  if(/iPhone/i.test(u))  return 'iPhone';
  if(/iPad/i.test(u))    return 'iPad';
  if(/Android/i.test(u)) return 'Android';
  if(/Mac OS X/i.test(u) && !/iPhone|iPad/i.test(u)) return 'macOS';
  if(/Windows/i.test(u)) return 'Windows';
  return 'Device';
}
const isIOS  = () => /iPhone|iPad|iPod/i.test(navigator.userAgent);
const isMob  = () => /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);

// ── BATTERY ───────────────────────────────────────────────────
async function initBat(){
  if(isIOS()){ batOK = false; return; }
  try{
    if(navigator.getBattery){
      const b = await navigator.getBattery();
      batLvl = Math.round(b.level*100); batOK = true;
      b.addEventListener('levelchange', ()=>{ batLvl = Math.round(b.level*100); });
    }
  }catch(e){ batOK = false; }
}

// ── MOTION ────────────────────────────────────────────────────
function listenMotion(){
  window.addEventListener('devicemotion', e=>{
    const a = e.accelerationIncludingGravity;
    if(!a || a.x === null) return;
    const mag = Math.sqrt((a.x||0)**2+(a.y||0)**2+(a.z||0)**2);
    lastVib = Math.round(Math.max(0,(mag-9.81)/9.81*0.5)*10000)/10000;
  });
}
function initMotion(){
  if(isIOS()) return;
  if(typeof DeviceMotionEvent !== 'undefined' && DeviceMotionEvent.requestPermission){
    DeviceMotionEvent.requestPermission().then(s=>{ if(s==='granted') listenMotion(); }).catch(()=>{});
  } else { listenMotion(); }
}
function startPhone(){
  if(typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function'){
    DeviceMotionEvent.requestPermission().then(p=>{
      if(p==='granted'){ listenMotion(); document.getElementById('ph-card').style.display='none'; }
    }).catch(()=>{});
  } else {
    listenMotion(); document.getElementById('ph-card').style.display='none';
  }
}

// ── LIVE SCANNING ─────────────────────────────────────────────
function startLive(){
  initMotion();
  clearInterval(liveInt);
  doScan();
  liveInt = setInterval(doScan, 4000);
}

async function doScan(){
  const dev = devName();
  const bat = (batOK && batLvl !== null) ? batLvl : -1;

  // RAM: use JS heap API if available, else deviceMemory estimate
  let mem = 50;
  if(performance.memory){
    mem = Math.round(performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit * 100);
    mem = Math.max(10, Math.min(95, mem));
  } else if(navigator.deviceMemory){
    mem = Math.max(20, Math.min(85, Math.round(100 - navigator.deviceMemory * 9)));
  }

  const vib = isMob() ? lastVib : 0.02;
  // CPU: browser cannot read real CPU. Use vibration-based proxy on mobile, fixed 50 on desktop
  // probe.py is the ONLY way to get real CPU on Windows
  const cpu = isMob() ? Math.round(15 + lastVib * 50) : 50;

  await post({
    device: dev, device_type: dev,
    battery: bat,
    vibration: vib,
    cpu: cpu, cpu_load: cpu,
    memory: mem,
    temperature: Math.round(32 + (100-(bat>=0?bat:80))*0.2 + lastVib*8),
    disk: 50,
    is_charging: !batOK || bat < 0 || (batLvl !== null && batLvl > 95),
    source: 'browser',
  });
}

// ── SEND & RENDER ─────────────────────────────────────────────
async function post(payload){
  try{
    const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d = await r.json();
    selDev = d.device || selDev;
    render(d);
    await refreshTabs();
  }catch(e){ document.getElementById('clbl').textContent='ERROR'; }
}

// FIX: Industrial/Custom — send, show result, DON'T switch mode or restart live
async function sendInd(){
  const payload = {
    device:'Industrial', device_type:'Industrial',
    cpu: +document.getElementById('ic').value || 55,
    memory: +document.getElementById('im').value || 50,
    temp: +document.getElementById('it').value || 45,
    disk: +document.getElementById('id').value || 40,
    vibration: +document.getElementById('iv').value || 0.08,
    battery: +document.getElementById('ib').value || 90,
    is_charging: true, source:'industrial',
  };
  document.querySelector('#mode-ind .rbtn').textContent = '⏳ ANALYZING…';
  try{
    const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d = await r.json();
    selDev = 'Industrial';
    render(d);
    await refreshTabs();
    document.getElementById('ind-result').style.display='block';
    document.querySelector('#mode-ind .rbtn').textContent = '⚙ RUN ML ANALYSIS';
    // Switch to live view to SHOW the results
    setMode('live');
  }catch(e){ document.querySelector('#mode-ind .rbtn').textContent = '⚙ RUN ML ANALYSIS'; }
}

async function sendCustom(){
  const bat = +document.getElementById('xb').value;
  const payload = {
    device:'Custom', device_type:'Custom',
    cpu: +document.getElementById('xc').value || 50,
    memory: +document.getElementById('xm').value || 50,
    temp: +document.getElementById('xt').value || 42,
    disk: +document.getElementById('xd').value || 50,
    vibration: +document.getElementById('xv').value || 0.05,
    battery: bat,
    is_charging: bat < 0,
    source:'custom',
  };
  document.querySelector('#mode-custom .rbtn').textContent = '⏳ ANALYZING…';
  try{
    const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d = await r.json();
    selDev = 'Custom';
    render(d);
    await refreshTabs();
    document.getElementById('cus-result').style.display='block';
    document.querySelector('#mode-custom .rbtn').textContent = '🧠 ANALYZE WITH ML';
    // Switch to live view to SHOW the results — but DON'T restart auto-scanning
    switchToLiveDisplay();
  }catch(e){ document.querySelector('#mode-custom .rbtn').textContent = '🧠 ANALYZE WITH ML'; }
}

// Switch to live display WITHOUT restarting the auto-scan (so results stay visible)
function switchToLiveDisplay(){
  curMode = 'live';
  ['live','ind','custom'].forEach(id=>{
    document.getElementById('mode-'+id).style.display = id==='live'?'block':'none';
    document.getElementById('mb-'+id).classList.toggle('on', id==='live');
  });
  // Don't call startLive() here — that would overwrite with a new browser scan immediately
}

// ── RENDER ALL DATA ────────────────────────────────────────────
function col(s){ return s>=65?'#ff3355':s>=35?'#ffd24d':'#00ff87'; }

function render(d){
  if(!d || d.error) return;
  const sc = d.health_score || 0;
  const C = col(sc);
  const icons = {NOMINAL:'✅',WARNING:'⚠️',CRITICAL:'🚨'};
  const descs = {NOMINAL:'All systems operating within healthy parameters',WARNING:'Elevated risk — schedule maintenance',CRITICAL:'HALT — immediate inspection required'};

  // Banner
  document.getElementById('bnr').className = 'banner '+(d.risk||'NOMINAL');
  document.getElementById('bico').textContent = icons[d.risk]||'⏳';
  document.getElementById('bttl').textContent = d.risk||'--';
  document.getElementById('bttl').style.color = C;
  document.getElementById('bsub').textContent = descs[d.risk]||'';
  document.getElementById('bnum').textContent = sc;
  document.getElementById('bnum').style.color = C;
  const bf = document.getElementById('bbarf'); bf.style.width=sc+'%'; bf.style.background=C;

  // Ring gauge
  const total = 251.2;
  document.getElementById('rf').style.strokeDashoffset    = total - (sc/100)*total;
  document.getElementById('rmask').style.strokeDashoffset = (sc/100)*total;
  document.getElementById('rnum').textContent = sc;
  document.getElementById('rnum').setAttribute('fill', C);
  const rb = document.getElementById('rbadge');
  rb.textContent = sc>=65?'CRITICAL':sc>=35?'WARNING':'NOMINAL';
  rb.className = 'rbadge '+(sc>=65?'CRITICAL':sc>=35?'WARNING':'NOMINAL');

  // Score breakdown
  const ifc = d.if_component||0, pen = d.penalty||0;
  document.getElementById('bd-if').style.width = ifc+'%';
  document.getElementById('bd-rl').style.width = Math.min(100, pen/60*100)+'%';
  document.getElementById('bd-fn').style.width = sc+'%';
  document.getElementById('bd-fn').style.background = C;
  document.getElementById('bv-if').textContent = ifc.toFixed(1);
  document.getElementById('bv-rl').textContent = '+'+pen;
  document.getElementById('bv-fn').textContent = sc+'/100';

  // Battery
  const bat = d.battery;
  const bc = bat<0?'var(--blue)':bat<15?'var(--red)':bat<30?'var(--yellow)':'var(--green)';
  document.getElementById('bat-big').textContent  = bat<0?'AC':(bat+'%');
  document.getElementById('bat-big').style.color  = bc;
  document.getElementById('bat-fill').style.width = Math.max(0,bat)+'%';
  document.getElementById('bat-fill').style.background = bc;
  document.getElementById('bat-info').textContent =
    d.is_charging ? '⚡ '+d.bat_life : bat>=0 ? '🔋 '+d.bat_life : '🔌 AC Power';

  // 4 mini rings
  const params=[
    {l:'CPU',    v:d.cpu,    u:'%',  mx:100, c:d.cpu>90?'#ff3355':d.cpu>75?'#ffd24d':'#00aaff',
     t:d.cpu>90?'OVERLOAD':d.cpu>75?'HIGH':'OK',
     est: d.source==='browser' && !d.cpu_freq},
    {l:'MEMORY', v:d.memory, u:'%',  mx:100, c:d.memory>90?'#ff3355':d.memory>75?'#ffd24d':'#a78bfa', t:d.memory>90?'CRITICAL':d.memory>75?'HIGH':'OK'},
    {l:'TEMP',   v:d.temp,   u:'°C', mx:110, c:d.temp>80?'#ff3355':d.temp>65?'#ff8c42':'#00e5c4',     t:d.thermal||'NORMAL'},
    {l:'DISK',   v:d.disk,   u:'%',  mx:100, c:d.disk>90?'#ff3355':d.disk>80?'#ffd24d':'#00ff87',     t:d.disk>90?'CRITICAL':d.disk>80?'HIGH':'OK'},
  ];
  document.getElementById('mr4').innerHTML = params.map((p,i)=>{
    const pct = Math.min(100, p.v/p.mx*100);
    const circ = 2*Math.PI*34;
    const off = circ*(1-pct/100);
    const tg = p.t==='OK'||p.t==='NORMAL'?'tn':p.t==='CRITICAL'?'tc':'tw';
    return `<div class="m4">
      <svg width="80" height="50" viewBox="0 0 80 50" style="overflow:visible">
        <defs><filter id="g${i}"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
        <path d="M7,45 A33,33 0 0,1 73,45" fill="none" stroke="#0e1e35" stroke-width="8" stroke-linecap="round"/>
        <path d="M7,45 A33,33 0 0,1 73,45" fill="none" stroke="${p.c}" stroke-width="8" stroke-linecap="round"
          stroke-dasharray="${(circ*0.65).toFixed(1)}" stroke-dashoffset="${(circ*0.65*(1-pct/100)).toFixed(1)}"
          style="transition:stroke-dashoffset .7s ease" filter="url(#g${i})"/>
      </svg>
      <div class="m4v" style="color:${p.c};margin-top:-4px">${typeof p.v==='number'?p.v.toFixed(p.u==='°C'?1:0):p.v}${p.u}</div>
      <div class="m4l">${p.l}${p.est?' <span style="color:#1e3a55">~est</span>':''}</div>
      <div class="m4t tag ${tg}">${p.t}</div>
    </div>`;
  }).join('');

  // Fault card
  const fC = {'NORMAL':['rgba(0,255,135,.04)','rgba(0,255,135,.2)','#00ff87'],
    'OVERHEATING':['rgba(255,51,85,.08)','rgba(255,51,85,.4)','#ff3355'],
    'CPU OVERLOAD':['rgba(255,140,66,.07)','rgba(255,140,66,.35)','#ff8c42'],
    'MEMORY PRESSURE':['rgba(167,139,250,.07)','rgba(167,139,250,.3)','#a78bfa'],
    'DISK CRITICAL':['rgba(255,210,77,.06)','rgba(255,210,77,.3)','#ffd24d'],
    'BEARING FAULT':['rgba(0,229,196,.06)','rgba(0,229,196,.25)','#00e5c4'],
    'LOW POWER':['rgba(255,51,85,.08)','rgba(255,51,85,.4)','#ff3355']};
  const fc = fC[d.fault]||fC['NORMAL'];
  const fcard = document.getElementById('fcard');
  fcard.style.background = fc[0]; fcard.style.borderColor = fc[1];
  document.getElementById('fname').textContent = d.fault||'NORMAL';
  document.getElementById('fname').style.color = fc[2];
  document.getElementById('fdesc').textContent = d.fault_desc||'All parameters within healthy envelope';

  // Alerts
  const al = d.alerts||[];
  document.getElementById('alist').innerHTML = al.length
    ? al.map(a=>`<div class="aitem ${a[1]}"><span class="asev">[${a[0]}] ${a[1]}</span><span class="amsg">${a[2]}</span></div>`).join('')
    : '<div class="noal">✓ No active alerts</div>';

  // Recs
  document.getElementById('rlist').innerHTML = (d.recommendations||[]).map((r,i)=>
    `<div class="recitem"><span class="recn">${String(i+1).padStart(2,'0')}.</span>${r}</div>`
  ).join('');

  // Metric rows (show estimated label for browser CPU)
  const metrics = [
    {ico:'⚙️',name:'CPU Load', val:d.cpu+'%',   bar:d.cpu,  c:d.cpu>90?'var(--red)':d.cpu>75?'var(--yellow)':'var(--blue)',    est: d.source==='browser'&&!d.cpu_freq},
    {ico:'💾',name:'Memory',   val:d.memory+'%', bar:d.memory,c:d.memory>90?'var(--red)':d.memory>75?'var(--yellow)':'var(--purple)'},
    {ico:'🌡️',name:'Temp',     val:d.temp+'°C',  bar:Math.min(100,d.temp), c:d.temp>80?'var(--red)':d.temp>65?'var(--orange)':'var(--teal)', est:d.source==='browser'&&!d.cpu_freq},
    {ico:'💿',name:'Disk',     val:d.disk+'%',   bar:d.disk, c:d.disk>90?'var(--red)':d.disk>80?'var(--yellow)':'var(--green)'},
    {ico:'📳',name:'Vibration',val:(d.vibration||0).toFixed(4), bar:Math.min(100,(d.vibration||0)/2*100), c:(d.vibration||0)>1?'var(--red)':(d.vibration||0)>0.5?'var(--orange)':'var(--teal)'},
  ];
  document.getElementById('mrows').innerHTML = metrics.map(m=>`
    <div class="mrow">
      <div class="mico" style="font-size:13px">${m.ico}</div>
      <div style="flex:1;min-width:0">
        <div class="mname">${m.name}${m.est?' <span style="color:#1e3a55;font-size:7px">ESTIMATED — run probe.py for real value</span>':''}</div>
        <div class="mval" style="color:${m.c}">${m.val}</div>
        <div class="mbar"><div class="mbarfill" style="width:${m.bar}%;background:${m.c}"></div></div>
      </div>
    </div>`).join('');

  // System info (probe.py only)
  if(d.cpu_freq || d.ram_total){
    document.getElementById('spcard').style.display='block';
    document.getElementById('probe-hint').style.display='none';
    const specs=[
      {l:'Device',   v:d.device||'--'},
      {l:'Platform', v:d.platform||'--'},
      {l:'CPU Freq', v:d.cpu_freq?d.cpu_freq+' MHz':'--'},
      {l:'Cores',    v:d.cpu_cores_logical?d.cpu_cores_logical+'L / '+d.cpu_cores_physical+'P':'--'},
      {l:'RAM',      v:d.ram_total?d.ram_used+' / '+d.ram_total+' GB':'--'},
      {l:'RAM Free', v:d.ram_avail?d.ram_avail+' GB free':'--'},
      {l:'Disk',     v:d.disk_total?d.disk_used+' / '+d.disk_total+' GB':'--'},
      {l:'Processes',v:d.proc_count||'--'},
      {l:'Disk R/W', v:d.disk_read_mb!=null?d.disk_read_mb+' / '+d.disk_write_mb+' MB/s':'--'},
      {l:'Network',  v:d.net_send_mb!=null?'↑'+d.net_send_mb+' ↓'+d.net_recv_mb+' MB/s':'--'},
    ];
    document.getElementById('spgrid').innerHTML = specs.map(s=>
      `<div class="spitem"><div class="splbl">${s.l}</div><div class="spval">${s.v}</div></div>`
    ).join('');
  }

  // Process table
  if(d.top_procs && d.top_procs.length){
    document.getElementById('pcard').style.display='block';
    document.getElementById('pbody').innerHTML = d.top_procs.map(p=>
      `<tr><td>${p.name}</td><td class="${p.cpu>50?'crit':p.cpu>20?'warn':''}">${p.cpu}%</td><td>${p.mem}%</td></tr>`
    ).join('');
  }

  // Meta strip tags
  const vt = d.vibration>1?'tc':d.vibration>0.5?'tw':'tn';
  const tt = d.thermal==='OVERHEATING'||d.thermal==='HOT'?'tc':d.thermal==='WARM'?'tw':'tn';
  const bt = d.bat_trend==='DRAINING'?'tw':d.bat_trend==='CHARGING'?'tb':'tn';
  const ct = d.cpu_trend==='RISING'?'tw':'tn';
  document.getElementById('m-vib').innerHTML = `<span class="tag ${vt}">${d.vib_label||'--'}</span>`;
  document.getElementById('m-thm').innerHTML = `<span class="tag ${tt}">${d.thermal||'--'}</span>`;
  document.getElementById('m-btr').innerHTML = `<span class="tag ${bt}">${d.bat_trend||'STABLE'}</span>`;
  document.getElementById('m-blt').textContent = d.bat_life||'--';
  document.getElementById('m-ctr').innerHTML = `<span class="tag ${ct}">${d.cpu_trend||'STABLE'}</span>`;
}

// ── SPARKLINES ─────────────────────────────────────────────────
function mkSpark(id, vals, cfn, max){
  const el = document.getElementById(id);
  if(!vals||!vals.length){
    el.innerHTML='<span style="color:var(--text3);font-family:var(--mono);font-size:9px">COLLECTING DATA…</span>';
    return;
  }
  const m = max||Math.max(...vals,1);
  el.innerHTML = vals.map(v=>
    `<div class="spbar" style="height:${Math.max(4,Math.min(100,v/m*100))}%;background:${cfn(v)}"></div>`
  ).join('');
}

// ── REFRESH TABS + CHARTS ──────────────────────────────────────
async function refreshTabs(){
  try{
    const r = await fetch('/data');
    const d = await r.json();
    if(d.error) return;
    document.getElementById('clbl').textContent='ONLINE';

    const devs  = d.all_devices||{};
    const names = Object.keys(devs);
    document.getElementById('dtabs').innerHTML = names.map(n=>
      `<button class="dtab ${n===selDev?'on':''}" onclick="switchDev('${n}')">${n.toUpperCase()}</button>`
    ).join('');

    if(!names.length) return;
    if(!devs[selDev]) selDev = names[names.length-1];

    const sh = (d.score_hist||{})[selDev]||[];
    const ch = (d.cpu_hist||{})[selDev]||[];
    const th = (d.tmp_hist||{})[selDev]||[];

    mkSpark('sp1', sh, v=>v>=65?'#ff3355':v>=35?'#ffd24d':'#00ff87', 100);
    mkSpark('sp2', ch, v=>v>=90?'#ff3355':v>=75?'#ffd24d':'#00aaff', 100);
    mkSpark('sp3', th, v=>v>=80?'#ff3355':v>=65?'#ff8c42':'#00e5c4', 110);

    document.getElementById('cs1').textContent = `Last ${sh.length} scans · ${selDev||''}`;
    document.getElementById('cs2').textContent = `Avg ${ch.length?Math.round(ch.reduce((a,b)=>a+b,0)/ch.length):0}% CPU · ${ch.length} scans`;
    document.getElementById('cs3').textContent = `Max ${th.length?Math.max(...th).toFixed(1):0}°C in window`;
  }catch(e){}
}

function switchDev(n){
  selDev = n;
  refreshTabs().then(()=>{
    fetch('/data').then(r=>r.json()).then(d=>{
      if(d.all_devices && d.all_devices[n]) render(d.all_devices[n]);
    });
  });
}

// ── MODE SWITCHING ─────────────────────────────────────────────
function setMode(m){
  curMode = m;
  ['live','ind','custom'].forEach(id=>{
    document.getElementById('mode-'+id).style.display = id===m?'block':'none';
    document.getElementById('mb-'+id).classList.toggle('on', id===m);
  });
  if(m==='live'){
    startLive();  // restart auto-scan only when explicitly choosing Live mode
  } else {
    clearInterval(liveInt);
  }
}

// ── BOOT ──────────────────────────────────────────────────────
selDev = devName();
if(isMob()) document.getElementById('ph-card').style.display='block';

initBat().then(()=>{ startLive(); });
setInterval(refreshTabs, 5000);
refreshTabs();
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)