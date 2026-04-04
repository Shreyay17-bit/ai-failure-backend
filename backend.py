import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# 1. THE BRAIN (The AI Logic)
@app.get("/")
async def root():
    return {"status": "AI System Online", "message": "Neural Monitor is Live on Render", "version": "1.0.2"}

@app.post("/predict")
async def predict(data: dict):
    # This receives data from your Phone App (main.py)
    vibration = data.get("vibration", 0)
    battery = data.get("battery", 100)
    
    # Simple AI Logic
    status = "NOMINAL"
    if vibration > 0.8:
        status = "CRITICAL FAILURE"
    elif vibration > 0.5:
        status = "WARNING: MAINTENANCE REQUIRED"
        
    return {"status": status, "vibration": vibration, "battery": battery}

# 2. THE DASHBOARD (The Visual Screen)
@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NEURAL MONITOR | LIVE</title>
        <style>
            body { 
                background-color: #050505; 
                color: #00ffcc; 
                font-family: 'Courier New', monospace; 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                height: 100vh; 
                margin: 0; 
                overflow: hidden;
            }
            .container {
                border: 2px solid #00ffcc;
                padding: 40px;
                background: rgba(0, 255, 204, 0.05);
                box-shadow: 0 0 20px rgba(0, 255, 204, 0.2);
                text-align: center;
            }
            h1 { text-transform: uppercase; letter-spacing: 10px; border-bottom: 1px solid #00ffcc; padding-bottom: 10px; }
            .status-box { font-size: 24px; margin-top: 20px; animation: blink 2s infinite; }
            @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NEURAL MONITOR</h1>
            <div class="status-box">SYSTEM STATUS: ACTIVE</div>
            <p>Cloud Inference Engine is running...</p>
            <p style="font-size: 10px; color: #666;">LISTENING FOR DATA FROM PROBE...</p>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)