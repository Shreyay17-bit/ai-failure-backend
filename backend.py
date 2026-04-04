import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

last_data = {}
history = []

@app.get("/")
async def root():
    return {"status": "AI System Online", "message": "Neural Monitor is Live on Render", "version": "1.0.2"}

@app.post("/predict")
async def predict(data: dict):
    global last_data, history

    vibration = data.get("vibration", 0)
    battery   = data.get("battery", 100)
    cpu_load  = data.get("cpu_load", 0)
    memory    = data.get("memory", 0)
    device    = data.get("device_type", "Unknown")

    risk_score = 0
    if vibration > 0.8:   risk_score += 50
    elif vibration > 0.5: risk_score += 25
    if battery < 20:      risk_score += 30
    elif battery < 40:    risk_score += 15
    if cpu_load > 90:     risk_score += 15
    if memory > 90:       risk_score += 10

    if risk_score >= 60:
        status   = "CRITICAL FAILURE"
        analysis = "Multiple failure indicators detected. Immediate action required."
        risk     = "CRITICAL"
    elif risk_score >= 30:
        status   = "WARNING: MAINTENANCE REQUIRED"
        analysis = "Elevated risk detected. Schedule maintenance soon."
        risk     = "WARNING"
    else:
        status   = "NOMINAL"
        analysis = "All systems operating within normal parameters."
        risk     = "NOMINAL"

    last_data = {
        "vibration": round(vibration, 3),
        "battery":   battery,
        "cpu_load":  cpu_load,
        "memory":    memory,
        "device":    device,
        "status":    status,
        "risk":      risk,
        "analysis":  analysis,
        "score":     risk_score,
    }
    history.append(risk_score)
    if len(history) > 20:
        history.pop(0)

    return {"status": status, "analysis": analysis, "risk": risk, "score": risk_score}


@app.get("/data")
async def data():
    return {**last_data, "history": history}


@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return """<!DOCTYPE html>
<html>
<head>
<title>NEURAL MONITOR | LIVE</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #050505; color: #00ffcc; font-family: 'Courier New', monospace; min-height: 100vh; padding: 20px; }
  h1 { text-align: center; letter-spacing: 8px; border-bottom: 1px solid #00ffcc; padding-bottom: 12px; margin-bottom: 24px; font-size: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { border: 1px solid #00ffcc33; background: rgba(0,255,204,0.04); border-radius: 8px; padding: 16px; }
  .card-label { font-size: 10px; letter-spacing: 2px; color: #00ffcc88; margin-bottom: 8px; }
  .card-value { font-size: 28px; font-weight: bold; }
  .nominal  { color: #00ffcc; }
  .warning  { color: #ffd166; }
  .critical { color: #ff4d4d; animation: blink 0.8s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .status-box { border: 2px solid; border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 24px; font-size: 18px; letter-spacing: 4px; }
  .analysis { color: #00ffcc99; font-size: 13px; margin-top: 8px; }
  .bar-wrap { background: #0a1a14; border-radius: 4px; height: 8px; margin-top: 6px; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width .5s; }
  .history { border: 1px solid #00ffcc22; border-radius: 8px; padding: 16px; }
  .hist-label { font-size: 10px; letter-spacing: 2px; color: #00ffcc88; margin-bottom: 10px; }
  .sparkline { display: flex; align-items: flex-end; gap: 4px; height: 50px; }
  .spark-bar { flex: 1; border-radius: 2px 2px 0 0; min-width: 8px; }
  .footer { text-align: center; font-size: 10px; color: #333; margin-top: 20px; }
  .ping { font-size: 10px; color: #00ffcc44; text-align: right; margin-bottom: 8px; }
</style>
</head>
<body>
<h1>NEURAL MONITOR</h1>
<div id="root">Loading data...</div>
<div class="footer">AUTO-REFRESHES EVERY 3 SECONDS | WAITING FOR PROBE DATA</div>
<script>
async function load() {
  try {
    const r = await fetch('/data');
    const d = await r.json();
    if (!d.device) {
      document.getElementById('root').innerHTML = '<p style="text-align:center;color:#333;margin-top:60px">No probe data yet.<br>Start a scan from your phone or laptop.</p>';
      return;
    }
    const riskClass = d.risk === 'CRITICAL' ? 'critical' : d.risk === 'WARNING' ? 'warning' : 'nominal';
    const borderColor = d.risk === 'CRITICAL' ? '#ff4d4d' : d.risk === 'WARNING' ? '#ffd166' : '#00ffcc';
    const sparkBars = (d.history || []).map(v =>
      `<div class="spark-bar" style="height:${Math.max(4,v)}%;background:${v>=60?'#ff4d4d':v>=30?'#ffd166':'#00ffcc'}"></div>`
    ).join('');
    document.getElementById('root').innerHTML = `
      <div class="ping">LAST PING FROM: ${d.device} | SCORE: ${d.score}/100</div>
      <div class="status-box ${riskClass}" style="border-color:${borderColor}">
        ${d.status}
        <div class="analysis">${d.analysis}</div>
      </div>
      <div class="grid">
        <div class="card">
          <div class="card-label">VIBRATION</div>
          <div class="card-value ${d.vibration>0.8?'critical':d.vibration>0.5?'warning':'nominal'}">${d.vibration}</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:${Math.min(d.vibration*100,100)}%;background:${d.vibration>0.8?'#ff4d4d':d.vibration>0.5?'#ffd166':'#00ffcc'}"></div></div>
        </div>
        <div class="card">
          <div class="card-label">BATTERY %</div>
          <div class="card-value ${d.battery<20?'critical':d.battery<40?'warning':'nominal'}">${d.battery}%</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:${d.battery}%;background:${d.battery<20?'#ff4d4d':d.battery<40?'#ffd166':'#00ffcc'}"></div></div>
        </div>
        <div class="card">
          <div class="card-label">CPU LOAD</div>
          <div class="card-value ${d.cpu_load>90?'critical':'nominal'}">${d.cpu_load}%</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:${d.cpu_load}%;background:${d.cpu_load>90?'#ff4d4d':'#00ffcc'}"></div></div>
        </div>
        <div class="card">
          <div class="card-label">MEMORY</div>
          <div class="card-value ${d.memory>90?'critical':'nominal'}">${d.memory}%</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:${d.memory}%;background:${d.memory>90?'#ff4d4d':'#00ffcc'}"></div></div>
        </div>
      </div>
      <div class="history">
        <div class="hist-label">RISK SCORE HISTORY (last ${(d.history||[]).length} scans)</div>
        <div class="sparkline">${sparkBars || '<span style="color:#333;font-size:11px">No history yet</span>'}</div>
      </div>`;
  } catch(e) {
    document.getElementById('root').innerHTML = '<p style="text-align:center;color:#ff4d4d;margin-top:60px">Error: ' + e.message + '</p>';
  }
}
load();
setInterval(load, 3000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)