"""
Nexus PM - System Probe
Run this on your PC/Phone to send REAL sensor values to the dashboard.
Install: pip install psutil requests
Run:     python probe.py
"""
import time, requests, psutil, platform

SERVER = "https://ai-failure-backend.onrender.com"

def get_real_data():
    # CPU - real % with 1 second interval
    cpu = psutil.cpu_percent(interval=1)

    # Memory - exact
    mem = psutil.virtual_memory()
    memory_pct = mem.percent

    # Battery - real
    battery_pct = -1
    is_charging = False
    bat = psutil.sensors_battery()
    if bat:
        battery_pct = int(bat.percent)
        is_charging = bat.power_plugged

    # Disk - real
    disk = psutil.disk_usage('C:\\' if platform.system()=='Windows' else '/')
    disk_pct = disk.percent

    # Temperature - real (works on Linux/some laptops)
    temp = 40.0
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries:
                    temp = entries[0].current
                    break
    except:
        # Windows fallback estimate based on CPU load
        temp = 35 + (cpu * 0.4)

    # CPU frequency
    freq = psutil.cpu_freq()
    cpu_freq = freq.current if freq else 1800

    # Network stats for vibration proxy (packet errors as anomaly signal)
    net = psutil.net_io_counters()
    err_rate = (net.errin + net.errout) / max(net.packets_recv + net.packets_sent, 1)
    vibration = round(min(err_rate * 10, 1.0), 4)

    # Process count
    proc_count = len(psutil.pids())

    return {
        "vibration":   vibration,
        "cpu":         round(cpu, 1),
        "memory":      round(memory_pct, 1),
        "battery":     battery_pct,
        "temp":        round(temp, 1),
        "disk":        round(disk_pct, 1),
        "cpu_freq":    round(cpu_freq, 1),
        "is_charging": is_charging,
        "ram_total":   round(mem.total / 1e9, 1),
        "ram_used":    round(mem.used / 1e9, 1),
        "proc_count":  proc_count,
        "device":      f"{platform.system()}-{platform.node()}",
        "platform":    platform.system(),
    }

print(f"Nexus PM Probe — sending to {SERVER}")
print("Press Ctrl+C to stop\n")

while True:
    try:
        data = get_real_data()
        r = requests.post(f"{SERVER}/predict", json=data, timeout=5)
        res = r.json()
        print(f"CPU:{data['cpu']}% | MEM:{data['memory']}% | BAT:{data['battery']}% | "
              f"TEMP:{data['temp']}°C | DISK:{data['disk']}% | "
              f"RISK:{res.get('risk','?')} | SCORE:{res.get('score','?')}/100")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(3)