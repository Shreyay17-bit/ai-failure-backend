"""
NEXUS Probe — sends exact Task Manager values to your Render backend.
Install:  pip install psutil requests
Run:      python probe.py
"""
import time, platform, socket, requests
import psutil

SERVER   = "https://ai-failure-backend.onrender.com"
INTERVAL = 3

def get_data():
    cpu_pct = psutil.cpu_percent(interval=1)
    cpu_freq = psutil.cpu_freq()
    cpu_freq_mhz = round(cpu_freq.current) if cpu_freq else 0
    cpu_cores_l = psutil.cpu_count(logical=True)
    cpu_cores_p = psutil.cpu_count(logical=False)

    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    try:
        d1 = psutil.disk_io_counters(); time.sleep(0.3); d2 = psutil.disk_io_counters()
        dr = round((d2.read_bytes-d1.read_bytes)/1e6/0.3,1)
        dw = round((d2.write_bytes-d1.write_bytes)/1e6/0.3,1)
    except: dr=dw=0

    try:
        n1=psutil.net_io_counters(); time.sleep(0.2); n2=psutil.net_io_counters()
        ns=round((n2.bytes_sent-n1.bytes_sent)/1e6/0.2,2)
        nr=round((n2.bytes_recv-n1.bytes_recv)/1e6/0.2,2)
    except: ns=nr=0

    bat=psutil.sensors_battery()
    if bat:
        bat_pct=round(bat.percent); charging=bat.power_plugged
        secs=bat.secsleft; bat_mins=round(secs/60) if secs and secs>0 else -1
    else:
        bat_pct=-1; charging=True; bat_mins=-1

    temp_c=40.0
    try:
        temps=psutil.sensors_temperatures()
        if temps:
            all_t=[s.current for ss in temps.values() for s in ss if s.current and s.current>0]
            if all_t: temp_c=round(max(all_t),1)
    except: pass

    procs=[]
    try:
        pl=[]
        for p in psutil.process_iter(['name','cpu_percent','memory_percent']):
            try: pl.append((p.info['name'],p.info['cpu_percent'],round(p.info['memory_percent'],1)))
            except: pass
        pl.sort(key=lambda x:x[1],reverse=True)
        procs=[{"name":p[0][:20],"cpu":p[1],"mem":p[2]} for p in pl[:5]]
    except: pass

    vib_proxy=min(0.4,(dr+dw)/500)

    return {
        "vibration":vib_proxy, "cpu":cpu_pct, "cpu_load":cpu_pct,
        "memory":ram.percent, "battery":bat_pct, "temp":temp_c,
        "temperature":temp_c, "disk":disk.percent, "is_charging":charging,
        "device":socket.gethostname()[:20],
        "platform":f"{platform.system()} {platform.machine()}",
        "os_version":platform.version()[:40],
        "cpu_freq":cpu_freq_mhz, "cpu_cores_logical":cpu_cores_l,
        "cpu_cores_physical":cpu_cores_p,
        "ram_total":round(ram.total/1e9,1), "ram_used":round(ram.used/1e9,1),
        "ram_avail":round(ram.available/1e9,1),
        "disk_total":round(disk.total/1e9,1), "disk_used":round(disk.used/1e9,1),
        "disk_read_mb":dr, "disk_write_mb":dw,
        "net_send_mb":ns, "net_recv_mb":nr,
        "bat_mins_left":bat_mins, "proc_count":len(psutil.pids()),
        "top_procs":procs, "source":"probe",
    }

def main():
    print(f"\n{'='*55}\n  NEXUS Probe v2 — Task Manager Bridge\n  Server: {SERVER}\n  Interval: {INTERVAL}s\n{'='*55}\n")
    n=0
    while True:
        n+=1
        try:
            data=get_data()
            r=requests.post(f"{SERVER}/predict",json=data,timeout=10)
            res=r.json()
            sym={"NOMINAL":"✅","WARNING":"⚠️ ","CRITICAL":"🚨"}.get(res.get("risk"),"?")
            print(f"[{n:04d}] {sym} {res.get('risk','?'):8s} score={res.get('health_score','?'):3}/100  "
                  f"cpu={data['cpu']:4.1f}%  ram={data['memory']:4.1f}%  "
                  f"temp={data['temp']:4.1f}°C  bat={'N/A' if data['battery']<0 else str(data['battery'])+'%'}")
        except Exception as e:
            print(f"[{n:04d}] ❌ {e}")
        time.sleep(INTERVAL)

if __name__=="__main__": main()