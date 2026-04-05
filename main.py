import time, requests, psutil

SERVER_URL = "https://ai-failure-backend.onrender.com"

while True:
    battery = -1
    try:
        b = psutil.sensors_battery()
        if b:
            battery = int(b.percent)
    except:
        pass

    payload = {
        "vibration": 0.05,   # laptop doesn't have accelerometer
        "battery": battery
    }

    try:
        requests.post(f"{SERVER_URL}/predict", json=payload)
    except:
        pass

    time.sleep(2)