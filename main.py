import time, math, json
import requests

SERVER_URL = "https://ai-failure-backend.onrender.com"

while True:
    vib = 0.05
    battery = 50

    try:
        from plyer import accelerometer, battery as batt

        accelerometer.enable()
        acc = accelerometer.acceleration

        if acc and acc[0] is not None:
            vib = abs(math.sqrt(acc[0]**2 + acc[1]**2 + acc[2]**2) - 9.81)

        battery = batt.status.get('percentage', 50)

    except:
        pass

    payload = {
        "vibration": round(vib, 3),
        "battery": battery
    }

    try:
        requests.post(f"{SERVER_URL}/predict", json=payload)
    except:
        pass

    time.sleep(1.5)