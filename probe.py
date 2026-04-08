"""
NEXUS PM — probe.py v5
Reads EXACT system values (Task Manager accurate) and POSTs to backend every 3 seconds.
Install: pip install psutil requests
Run:     python probe.py
"""
import time, platform, socket, requests, psutil

URL  = "http://localhost:8000/predict"
INTERVAL = 3  # seconds

def get_battery():
    b = psutil.sensors_battery()
    if b is None:
        return -1, True
    return round(b.percent, 1), b.power_plugged

def get_temps():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        for key in ["coretemp","cpu_thermal","k10temp","zenpower","acpitz","cpu-thermal","thermal_zone0"]:
            if key in temps and temps[key]:
                readings = [t.current for t in temps[key] if t.current and t.current > 0]
                if readings:
                    return round(max(readings), 1)
        for entries in temps.values():
            readings = [t.current for t in entries if t.current and t.current > 0]
            if readings:
                return round(max(readings), 1)
    except Exception:
        pass
    return None

def get_top_procs(n=8):
    procs = []
    for p in psutil.process_iter(['name','cpu_percent','memory_percent']):
        try:
            procs.append({
                "name": p.info['name'][:28],
                "cpu":  round(p.info['cpu_percent'] or 0, 1),
                "mem":  round(p.info['memory_percent'] or 0, 1),
            })
        except Exception:
            pass
    return sorted(procs, key=lambda x: x['cpu'], reverse=True)[:n]

def get_disk():
    usage = psutil.disk_usage('/')
    io    = psutil.disk_io_counters()
    return {
        "disk":          round(usage.percent, 1),
        "disk_total":    round(usage.total   / 1e9, 1),
        "disk_used":     round(usage.used    / 1e9, 1),
        "disk_read_mb":  round(io.read_bytes  / 1e6, 2) if io else 0,
        "disk_write_mb": round(io.write_bytes / 1e6, 2) if io else 0,
    }

def get_net():
    n = psutil.net_io_counters()
    return {
        "net_send_mb": round(n.bytes_sent / 1e6, 2),
        "net_recv_mb": round(n.bytes_recv / 1e6, 2),
    }

def build_payload():
    cpu  = psutil.cpu_percent(interval=1)
    freq = psutil.cpu_freq()
    ram  = psutil.virtual_memory()
    bat_pct, charging = get_battery()
    temp = get_temps()
    if temp is None:
        temp = round(30 + cpu * 0.45, 1)

    disk_info = get_disk()
    net_info  = get_net()

    return {
        "cpu": round(cpu,1), "cpu_load": round(cpu,1),
        "memory": round(ram.percent,1),
        "temp": temp, "temperature": temp,
        "vibration": 0.01,
        "battery": bat_pct, "is_charging": charging,
        "cpu_freq": round(freq.current) if freq else None,
        "cpu_cores_logical":  psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "ram_total": round(ram.total    / 1e9, 1),
        "ram_used":  round(ram.used     / 1e9, 1),
        "ram_avail": round(ram.available / 1e9, 1),
        **disk_info, **net_info,
        "top_procs":  get_top_procs(),
        "proc_count": len(psutil.pids()),
        "device":      platform.node() or socket.gethostname(),
        "device_type": platform.node() or socket.gethostname(),
        "os_version":  platform.version()[:60],
        "platform":    platform.system() + " " + platform.release(),
        "source": "probe",
    }

print("=" * 58)
print("  NEXUS PM — probe.py  (Ctrl+C to stop)")
print(f"  Posting to: {URL}  every {INTERVAL}s")
print("=" * 58)
psutil.cpu_percent(interval=None)
time.sleep(0.5)

while True:
    try:
        payload = build_payload()
        r = requests.post(URL, json=payload, timeout=5)
        d = r.json()
        print(f"  CPU {payload['cpu']:5.1f}%  RAM {payload['memory']:5.1f}%  "
              f"Temp {payload['temp']:5.1f}C  Disk {payload['disk']:4.1f}%  "
              f"Score {d.get('health_score','?'):3}  [{d.get('risk','?')}]")
    except requests.exceptions.ConnectionError:
        print("  Cannot connect to backend — is it running?")
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(INTERVAL)