from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

class Telemetry(BaseModel):
    device_type: str
    battery: float
    cpu_load: float
    memory: float

@app.get("/")
async def home():
    return {"status": "AI System Online", "version": "1.0.3"}

@app.post("/predict")
async def predict(data: Telemetry):
    risk_score = (data.cpu_load * 0.6) + (data.memory * 0.4)
    analysis = "NOMINAL"
    if risk_score > 80: analysis = "CRITICAL"
    elif risk_score > 50: analysis = "WARNING"

    return {
        "risk": f"{min(risk_score, 99):.1f}%",
        "analysis": analysis,
        "raw": {
            "CPU": f"{int(data.cpu_load)}%",
            "RAM": f"{int(data.memory)}%",
            "BATT": f"{int(data.battery)}%",
            "DEVICE": data.device_type
        }
    }

@app.get("/monitor", response_class=HTMLResponse)
async def monitor_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Monitor Pro</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background: #05050a; color: #00ffc8; font-family: 'Courier New', monospace; text-align: center; padding: 20px; }
            .container { border: 2px solid #00ffc8; border-radius: 20px; padding: 30px; max-width: 500px; margin: auto; box-shadow: 0 0 30px #00ffc833; }
            .gauge { font-size: 5em; font-weight: bold; margin: 10px 0; text-shadow: 0 0 10px #00ffc8; }
            .grid { display: grid; grid-template-cols: 1fr 1fr 1fr; gap: 15px; margin-top: 30px; border-top: 1px solid #222; padding-top: 20px; }
            .stat-box { background: #111; padding: 10px; border-radius: 8px; border: 1px solid #333; }
            .label { color: #888; font-size: 0.7em; text-transform: uppercase; }
            .value { font-size: 1.2em; color: #fff; margin-top: 5px; }
            .device-tag { background: #00ffc8; color: #000; padding: 5px 15px; border-radius: 20px; display: inline-block; font-weight: bold; margin-bottom: 10px; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <div class="container">
            <div id="device-name" class="device-tag">DETECTING...</div>
            <div style="color: #888; font-size: 0.9em;">FAILURE PROBABILITY</div>
            <div id="risk-val" class="gauge">0%</div>
            <div id="status-text" style="letter-spacing: 3px; font-weight: bold;">INITIALIZING AI...</div>

            <div class="grid">
                <div class="stat-box"><div class="label">CPU</div><div id="cpu-val" class="value">--</div></div>
                <div class="stat-box"><div class="label">RAM</div><div id="ram-val" class="value">--</div></div>
                <div class="stat-box"><div class="label">BATT</div><div id="batt-val" class="value">--</div></div>
            </div>
        </div>

        <script>
            async function refresh() {
                const ua = navigator.userAgent;
                let device = "Generic Device";
                if (ua.includes("Android")) device = "Android Core";
                if (ua.includes("iPhone") || ua.includes("iPad")) device = "iOS Neural Node";
                if (ua.includes("Windows")) device = "Windows Workstation";

                document.getElementById('device-name').innerText = device;

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            device_type: device,
                            battery: 88, // Simulated from browser
                            cpu_load: Math.random() * 100,
                            memory: Math.random() * 100
                        })
                    });
                    const data = await res.json();
                    document.getElementById('risk-val').innerText = data.risk;
                    document.getElementById('status-text').innerText = "AI INSIGHT: " + data.analysis;
                    document.getElementById('cpu-val').innerText = data.raw.CPU;
                    document.getElementById('ram-val').innerText = data.raw.RAM;
                    document.getElementById('batt-val').innerText = data.raw.BATT;
                    
                    // Visual feedback: change colors based on risk
                    const riskNum = parseFloat(data.risk);
                    document.getElementById('risk-val').style.color = riskNum > 80 ? "#ff4444" : (riskNum > 50 ? "#ffaa00" : "#00ffc8");
                } catch (e) { console.error("Sync Error", e); }
            }
            setInterval(refresh, 2500);
            refresh();
        </script>
    </body>
    </html>
    """