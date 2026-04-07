"""
NEXUS PM Backend v4
Single ML model: Isolation Forest, properly calibrated.
Exact Task Manager values via probe.py.
"""
import os, time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import IsolationForest

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

rng=np.random.default_rng(42); N=2000
X_h=np.column_stack([rng.uniform(5,75,N),rng.uniform(20,80,N),rng.uniform(30,68,N),
                     rng.uniform(5,80,N),rng.uniform(0,0.35,N),rng.uniform(25,100,N)])
iforest=IsolationForest(contamination=0.02,n_estimators=300,random_state=42)
iforest.fit(X_h)
_sh=iforest.score_samples(X_h); GOOD_T=float(np.percentile(_sh,95))
X_f=np.column_stack([rng.uniform(92,100,500),rng.uniform(92,100,500),rng.uniform(88,105,500),
                     rng.uniform(92,100,500),rng.uniform(1.5,3.5,500),rng.uniform(1,8,500)])
BAD_T=float(np.percentile(iforest.score_samples(X_f),5))

def if_score(cpu,ram,temp,disk,vib,bat):
    raw=float(iforest.score_samples([[cpu,ram,temp,disk,vib,bat]])[0])
    return float(np.clip((GOOD_T-raw)/(GOOD_T-BAD_T)*100,0,100))

def rule_penalty(cpu,ram,temp,disk,vib,bat,charging):
    p=0; trig=[]
    if   temp>90: p+=35; trig.append(("TEMP","CRITICAL",f"{temp:.1f}C thermal throttling"))
    elif temp>80: p+=22; trig.append(("TEMP","HOT",f"{temp:.1f}C cooling stressed"))
    elif temp>72: p+=12; trig.append(("TEMP","ELEVATED",f"{temp:.1f}C above comfort"))
    elif temp>62: p+=5;  trig.append(("TEMP","WARM",f"{temp:.1f}C monitor"))
    if   cpu>97:  p+=25; trig.append(("CPU","SATURATED",f"{cpu:.1f}% system unresponsive"))
    elif cpu>90:  p+=14; trig.append(("CPU","OVERLOAD",f"{cpu:.1f}% heavy load"))
    elif cpu>80:  p+=7;  trig.append(("CPU","HIGH",f"{cpu:.1f}% elevated"))
    if   ram>96:  p+=20; trig.append(("RAM","CRITICAL",f"{ram:.1f}% paging to disk"))
    elif ram>88:  p+=10; trig.append(("RAM","HIGH",f"{ram:.1f}% near exhausted"))
    elif ram>78:  p+=4;  trig.append(("RAM","ELEVATED",f"{ram:.1f}%"))
    if not charging and bat>=0:
        if   bat<5:  p+=35; trig.append(("BAT","EMPTY",f"{bat:.0f}% shutdown imminent"))
        elif bat<12: p+=22; trig.append(("BAT","CRITICAL",f"{bat:.0f}% plug in now"))
        elif bat<20: p+=12; trig.append(("BAT","LOW",f"{bat:.0f}% drain risk"))
        elif bat<30: p+=5;  trig.append(("BAT","WARNING",f"{bat:.0f}% below 30%"))
    if   disk>97: p+=18; trig.append(("DISK","FULL",f"{disk:.0f}% write errors likely"))
    elif disk>92: p+=10; trig.append(("DISK","CRITICAL",f"{disk:.0f}% nearly full"))
    elif disk>85: p+=5;  trig.append(("DISK","HIGH",f"{disk:.0f}% low free space"))
    if   vib>2.0: p+=20; trig.append(("VIB","CRITICAL",f"{vib:.3f} m/s2 bearing failure risk"))
    elif vib>1.0: p+=10; trig.append(("VIB","HIGH",f"{vib:.3f} m/s2 abnormal"))
    elif vib>0.5: p+=4;  trig.append(("VIB","ELEVATED",f"{vib:.3f} m/s2"))
    return min(p,60),trig

def classify_fault(cpu,ram,temp,disk,vib,bat,score):
    if score<28: return "NORMAL","All parameters within healthy envelope"
    c={"OVERHEATING":(temp-45)*1.8+cpu*0.15,"CPU OVERLOAD":cpu*0.9+ram*0.2,
       "MEMORY PRESSURE":ram*0.85+disk*0.2,"DISK CRITICAL":disk*0.9,
       "BEARING FAULT":vib*40,"LOW POWER":max(0,50-bat)*0.8}
    f=max(c,key=c.get)
    d={"OVERHEATING":"Thermal anomaly — cooling system under stress",
       "CPU OVERLOAD":"Processor saturation — background process surge",
       "MEMORY PRESSURE":"RAM exhaustion — swap activity elevated",
       "DISK CRITICAL":"Storage nearly full — write performance degraded",
       "BEARING FAULT":"Mechanical vibration outside normal envelope",
       "LOW POWER":"Insufficient power — battery drain accelerated"}
    return f,d.get(f,"")

devices={}; device_history={}

@app.post("/predict")
async def predict(data:dict):
    cpu =float(data.get("cpu", data.get("cpu_load",  50)))
    ram =float(data.get("memory", 50))
    temp=float(data.get("temp", data.get("temperature",42)))
    disk=float(data.get("disk", data.get("disk_usage", 50)))
    vib =float(data.get("vibration",0.02))
    bat =float(data.get("battery",-1))
    dev =str(data.get("device", data.get("device_type","Unknown")))
    chg =bool(data.get("is_charging",True))

    bat_if=bat if bat>=0 else 85
    ifc=if_score(cpu,ram,temp,disk,vib,bat_if)
    pen,trig=rule_penalty(cpu,ram,temp,disk,vib,bat,chg)
    combined=ifc*0.55+(pen/60*100)*0.45
    hs=int(np.clip(combined,0,100))
    ok=cpu<80 and ram<82 and temp<70 and disk<88 and vib<0.5 and (bat<0 or bat>20 or chg)
    if ok: hs=min(hs,30)

    fault,fdesc=classify_fault(cpu,ram,temp,disk,vib,bat,hs)

    dh=device_history.get(dev,[])
    bat_life="N/A"; bat_trend="STABLE"
    if bat>=0:
        if chg: bat_life="Charging"; bat_trend="CHARGING"
        elif len(dh)>=4:
            bats=[h["bat"] for h in dh[-8:] if h["bat"]>=0]
            if len(bats)>=4:
                drain=(bats[0]-bats[-1])/(len(bats)*3)
                dph=drain*3600
                if dph>0.2:
                    r=bat/dph
                    bat_life=f"{int(r*60)}min" if r<1 else (f"{r:.1f}h" if r<10 else f"{int(r)}h")
                    bat_trend="DRAINING"
                else: bat_life="Stable"
    elif chg: bat_life="AC Power"

    if len(dh)>=5:
        cpus=[h["cpu"] for h in dh[-10:]]
        sl=(cpus[-1]-cpus[0])/len(cpus)
        cpu_trend="RISING" if sl>2 else "FALLING" if sl<-2 else "STABLE"
    else: cpu_trend="STABLE"

    vib_lbl=("SMOOTH" if vib<0.08 else "NORMAL" if vib<0.35 else
              "ELEVATED" if vib<0.7 else "HIGH" if vib<1.2 else "CRITICAL")
    thermal=("COOL" if temp<38 else "NORMAL" if temp<58 else
             "WARM" if temp<70 else "HOT" if temp<82 else "OVERHEATING")

    if hs>=65:
        risk,status="CRITICAL","CRITICAL IMMEDIATE ACTION"
        recs=[f"HALT OPERATIONS primary fault {fault}",
              f"Health score {hs}/100 exceeds critical threshold 65",
              "Halt non-essential processes immediately",
              "Inspect thermal and mechanical subsystems"]
    elif hs>=35:
        risk,status="WARNING","WARNING SCHEDULE MAINTENANCE"
        recs=[f"Fault pattern {fault}",
              f"Score {hs}/100 {len(trig)} parameter(s) above threshold",
              f"Battery {bat_life}" if bat>=0 else "Monitor CPU and thermal",
              "Reduce load or increase monitoring frequency"]
    else:
        risk,status="NOMINAL","NOMINAL ALL SYSTEMS HEALTHY"
        recs=[f"Isolation Forest no anomalies detected score {hs}/100",
              f"CPU {cpu:.1f}%  RAM {ram:.1f}%  Temp {temp:.1f}C",
              f"Battery {bat_life}" if bat>=0 else "AC Power",
              "Continue standard monitoring"]

    extras={k:data[k] for k in ["cpu_freq","cpu_cores_logical","cpu_cores_physical",
        "ram_total","ram_used","ram_avail","disk_total","disk_used",
        "disk_read_mb","disk_write_mb","net_send_mb","net_recv_mb",
        "bat_mins_left","proc_count","top_procs","os_version","platform","source"] if k in data}

    result={"cpu":round(cpu,1),"memory":round(ram,1),"temp":round(temp,1),
            "disk":round(disk,1),"vibration":round(vib,4),"battery":round(bat,1),
            "is_charging":chg,"device":dev,"health_score":hs,
            "if_component":round(ifc,1),"penalty":pen,"risk":risk,"status":status,
            "fault":fault,"fault_desc":fdesc,"triggered":trig,
            "bat_life":bat_life,"bat_trend":bat_trend,"cpu_trend":cpu_trend,
            "vib_label":vib_lbl,"thermal":thermal,"recommendations":recs,**extras}

    devices[dev]=result
    device_history.setdefault(dev,[]).append(
        {"cpu":cpu,"ram":ram,"temp":temp,"disk":disk,"bat":bat,"score":hs,"ts":time.time()})
    if len(device_history[dev])>40: device_history[dev].pop(0)
    return result

@app.get("/data")
async def get_data():
    if not devices: return {"error":"No data yet"}
    return {"all_devices":devices,
            "score_hist":{k:[h["score"] for h in v] for k,v in device_history.items()},
            "cpu_hist":  {k:[h["cpu"]   for h in v] for k,v in device_history.items()},
            "ram_hist":  {k:[h["ram"]   for h in v] for k,v in device_history.items()},
            "tmp_hist":  {k:[h["temp"]  for h in v] for k,v in device_history.items()}}

@app.get("/")
async def root(): return {"status":"online","version":"4.0","model":"IsolationForest"}


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS PM</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{--void:#03050a;--deep:#060c16;--panel:#080f1c;--panel2:#0b1525;--rim:#0e1e35;--rim2:#162840;
--blue:#0af;--blue2:#0088cc;--teal:#00e5c4;--green:#00ff87;--yellow:#ffd24d;--orange:#ff8c42;
--red:#ff3355;--purple:#a78bfa;--text:#c8dff5;--text2:#4a7090;--text3:#1e3a55;
--mono:'JetBrains Mono',monospace;--display:'Orbitron',monospace;--ui:'Syne',sans-serif}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:14px}body{background:var(--void);color:var(--text);font-family:var(--ui);min-height:100vh;overflow-x:hidden}
#stars{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden}
.star{position:absolute;border-radius:50%;background:#fff;animation:twinkle var(--d,3s) ease-in-out infinite var(--delay,0s)}
@keyframes twinkle{0%,100%{opacity:var(--lo,.1)}50%{opacity:var(--hi,.8)}}
body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
background-image:linear-gradient(rgba(0,170,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,170,255,.025) 1px,transparent 1px);background-size:60px 60px}
.app{position:relative;z-index:1;max-width:1500px;margin:0 auto;padding:14px;display:flex;flex-direction:column;gap:12px}
.hdr{display:flex;align-items:center;gap:14px;padding:12px 22px;background:linear-gradient(135deg,rgba(8,15,28,.96),rgba(6,12,22,.96));
border:1px solid var(--rim2);border-radius:14px;backdrop-filter:blur(20px);flex-wrap:wrap;
box-shadow:0 0 40px rgba(0,170,255,.07),inset 0 1px 0 rgba(0,170,255,.1)}
.hdr-logo{font-family:var(--display);font-size:22px;font-weight:900;letter-spacing:4px;color:var(--blue);text-shadow:0 0 20px rgba(0,170,255,.6);position:relative}
.hdr-logo span{font-size:10px;font-weight:400;color:var(--teal);letter-spacing:2px;margin-left:4px}
.hdr-chip{font-family:var(--mono);font-size:8px;letter-spacing:2px;color:var(--text3);padding:3px 10px;border:1px solid var(--rim2);border-radius:4px}
.hdr-st{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--text2)}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pa 2.5s ease-in-out infinite}
@keyframes pa{0%,100%{box-shadow:0 0 0 0 rgba(0,255,135,.5)}70%{box-shadow:0 0 0 8px rgba(0,255,135,0)}}
.hdr-r{margin-left:auto;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.clk{font-family:var(--display);font-size:13px;color:var(--blue);letter-spacing:2px}
.dtl{font-family:var(--mono);font-size:8px;color:var(--text3);letter-spacing:1px}
.dev-tabs{display:flex;gap:6px}
.dev-tab{padding:4px 12px;border:1px solid var(--rim2);border-radius:6px;font-family:var(--mono);font-size:8px;letter-spacing:1px;color:var(--text2);background:transparent;cursor:pointer;transition:all .2s}
.dev-tab:hover{border-color:var(--blue2);color:var(--text)}.dev-tab.on{border-color:var(--blue);color:var(--blue);background:rgba(0,170,255,.08)}
.modes{display:flex;gap:8px;padding:10px 16px;background:rgba(8,15,28,.8);border:1px solid var(--rim);border-radius:10px;flex-wrap:wrap}
.mbtn{padding:7px 18px;border:1px solid var(--rim2);border-radius:7px;font-family:var(--mono);font-size:8px;letter-spacing:2px;color:var(--text2);background:transparent;cursor:pointer;transition:all .25s;flex:1;text-align:center;min-width:100px}
.mbtn:hover{border-color:var(--teal);color:var(--teal)}.mbtn.on{border-color:var(--blue);color:var(--blue);background:rgba(0,170,255,.1);box-shadow:0 0 12px rgba(0,170,255,.15)}
.banner{display:flex;align-items:center;gap:18px;padding:16px 24px;border-radius:12px;border:1px solid;transition:all .5s;flex-wrap:wrap;position:relative;overflow:hidden}
.banner::before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.015),transparent);animation:sweep 5s linear infinite;pointer-events:none}
@keyframes sweep{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
.banner.NOMINAL{background:rgba(0,255,135,.04);border-color:rgba(0,255,135,.2)}
.banner.WARNING{background:rgba(255,210,77,.05);border-color:rgba(255,210,77,.35);animation:bw 2.5s infinite}
.banner.CRITICAL{background:rgba(255,51,85,.07);border-color:rgba(255,51,85,.5);animation:bc 1s infinite}
@keyframes bw{0%,100%{border-color:rgba(255,210,77,.3)}50%{border-color:rgba(255,210,77,.8)}}
@keyframes bc{0%,100%{border-color:rgba(255,51,85,.4)}50%{border-color:rgba(255,51,85,1)}}
.b-ico{font-size:32px;flex-shrink:0}
.b-ttl{font-family:var(--display);font-size:18px;font-weight:700;letter-spacing:3px}
.b-sub{font-family:var(--mono);font-size:9px;color:var(--text2);margin-top:3px;letter-spacing:1px}
.b-sc{margin-left:auto;text-align:right;flex-shrink:0}
.b-num{font-family:var(--display);font-size:52px;font-weight:900;line-height:1;transition:color .5s}
.b-lbl{font-family:var(--mono);font-size:8px;color:var(--text3);letter-spacing:2px;margin-top:2px}
.b-bar{position:absolute;bottom:0;left:0;width:100%;height:4px;background:var(--rim2);border-radius:0 0 12px 12px;overflow:hidden}
.b-bar-f{height:100%;border-radius:0 0 12px 12px;transition:width 1s cubic-bezier(.4,0,.2,1),background .5s}
.mg{display:grid;grid-template-columns:280px 1fr 260px;gap:12px}
@media(max-width:1100px){.mg{grid-template-columns:1fr 1fr}}@media(max-width:680px){.mg{grid-template-columns:1fr}}
.lc,.cc,.rc{display:flex;flex-direction:column;gap:12px}
.card{background:linear-gradient(135deg,var(--panel),var(--panel2));border:1px solid var(--rim);border-radius:12px;padding:16px;animation:fu .4s ease both;box-shadow:0 4px 24px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.03)}
@keyframes fu{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.ch{font-family:var(--mono);font-size:8px;letter-spacing:3px;color:var(--text3);text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.ch::before{content:'';width:2px;height:10px;background:var(--blue);border-radius:2px;flex-shrink:0}
.rw{display:flex;flex-direction:column;align-items:center;padding:4px 0}
.rb{display:inline-block;margin:10px auto 0;padding:4px 16px;font-family:var(--mono);font-size:9px;letter-spacing:2px;border:1px solid;border-radius:4px;text-align:center;transition:all .4s}
.rb.NOMINAL{border-color:rgba(0,255,135,.4);color:var(--green);background:rgba(0,255,135,.06)}
.rb.WARNING{border-color:rgba(255,210,77,.5);color:var(--yellow);background:rgba(255,210,77,.07)}
.rb.CRITICAL{border-color:rgba(255,51,85,.6);color:var(--red);background:rgba(255,51,85,.09);animation:bc 1s infinite}
.mr4{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.m4{display:flex;flex-direction:column;align-items:center;padding:10px 8px;background:var(--panel);border:1px solid var(--rim);border-radius:10px}
.m4v{font-family:var(--display);font-size:16px;font-weight:700;text-align:center;margin-top:4px;line-height:1}
.m4l{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--text3);text-align:center;margin-top:2px}
.m4t{font-family:var(--mono);font-size:7px;letter-spacing:1px;margin-top:3px;text-align:center;padding:1px 6px;border-radius:3px}
.mrow{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--rim)}
.mrow:last-child{border:none}
.mic{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0}
.mn{font-family:var(--mono);font-size:9px;letter-spacing:1px;color:var(--text2)}
.mv{font-family:var(--display);font-size:15px;font-weight:600;line-height:1.2}
.mbar{height:2px;background:var(--rim2);border-radius:2px;margin-top:3px;overflow:hidden}
.mbf{height:100%;border-radius:2px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.fc{border-radius:9px;padding:12px;margin-bottom:10px;border:1px solid}
.fn{font-family:var(--display);font-size:15px;font-weight:700;letter-spacing:2px;margin-bottom:3px}
.fd{font-family:var(--mono);font-size:9px;color:var(--text2);line-height:1.6}
.al{display:flex;flex-direction:column;gap:6px}
.ai{display:flex;align-items:flex-start;gap:10px;padding:7px 10px;border-radius:7px;border:1px solid;font-family:var(--mono);font-size:9px;line-height:1.5}
.ai.CRITICAL,.ai.EMPTY{background:rgba(255,51,85,.07);border-color:rgba(255,51,85,.25);color:var(--red)}
.ai.HOT,.ai.OVERLOAD,.ai.HIGH,.ai.SATURATED,.ai.FULL{background:rgba(255,140,66,.07);border-color:rgba(255,140,66,.25);color:var(--orange)}
.ai.ELEVATED,.ai.WARM,.ai.LOW,.ai.WARNING{background:rgba(255,210,77,.05);border-color:rgba(255,210,77,.2);color:var(--yellow)}
.asv{font-weight:700;min-width:70px;flex-shrink:0}.amg{color:var(--text2);flex:1}
.na{font-family:var(--mono);font-size:10px;color:var(--text3);padding:12px;text-align:center}
.sp{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.sc{background:var(--panel);border:1px solid var(--rim);border-radius:8px;padding:9px 11px}
.sl{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--text3);margin-bottom:3px;text-transform:uppercase}
.sv{font-family:var(--mono);font-size:12px;color:var(--blue);font-weight:500}
.ri{display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid var(--rim);font-size:12px;line-height:1.5}
.ri:last-child{border:none}
.rn{font-family:var(--mono);font-size:9px;color:var(--blue);min-width:22px;flex-shrink:0;padding-top:1px}
.bd{display:flex;flex-direction:column;gap:8px;margin-top:4px}
.bdr{display:flex;align-items:center;gap:10px}
.bdl{font-family:var(--mono);font-size:8px;color:var(--text2);min-width:120px;flex-shrink:0}
.bdt{flex:1;height:5px;background:var(--rim2);border-radius:3px;overflow:hidden}
.bdf{height:100%;border-radius:3px;transition:width .9s cubic-bezier(.4,0,.2,1)}
.bdv{font-family:var(--mono);font-size:9px;color:var(--text2);min-width:36px;text-align:right;flex-shrink:0}
.cs{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media(max-width:750px){.cs{grid-template-columns:1fr}}
.cc2{background:var(--panel);border:1px solid var(--rim);border-radius:12px;padding:14px}
.cth{font-family:var(--mono);font-size:8px;letter-spacing:3px;color:var(--text3);margin-bottom:4px;text-transform:uppercase}
.cts{font-family:var(--mono);font-size:9px;color:var(--text2);margin-bottom:12px}
.spk{display:flex;align-items:flex-end;gap:2.5px;height:70px}
.spb{flex:1;min-width:3px;border-radius:1.5px 1.5px 0 0;transition:height .5s ease,background .3s}
.ms{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:8px 0}
@media(max-width:750px){.ms{grid-template-columns:repeat(2,1fr)}}
.mc2{background:var(--panel);border:1px solid var(--rim);border-radius:9px;padding:10px 14px}
.mk{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--text3);margin-bottom:5px;text-transform:uppercase}
.mv2{font-family:var(--mono);font-size:11px;color:var(--text)}
.tag{display:inline-block;padding:2px 8px;border-radius:3px;font-family:var(--mono);font-size:8px;letter-spacing:1px}
.tn{background:rgba(0,255,135,.1);color:var(--green)}.tw{background:rgba(255,210,77,.1);color:var(--yellow)}
.tc{background:rgba(255,51,85,.12);color:var(--red)}.tb{background:rgba(0,170,255,.1);color:var(--blue)}
.ig{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
@media(max-width:500px){.ig{grid-template-columns:1fr}}
.if2{display:flex;flex-direction:column;gap:4px}
.if2 label{font-family:var(--mono);font-size:8px;letter-spacing:2px;color:var(--text3);text-transform:uppercase}
.if2 input{background:var(--panel2);border:1px solid var(--rim2);border-radius:7px;color:var(--text);font-family:var(--mono);font-size:13px;padding:9px 12px;outline:none;transition:border-color .2s,box-shadow .2s;width:100%}
.if2 input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(0,170,255,.12)}
.if2 .hint{font-family:var(--mono);font-size:8px;color:var(--text3);line-height:1.5}
.rbtn{width:100%;padding:13px;background:rgba(0,170,255,.1);border:1px solid var(--blue);border-radius:10px;color:var(--blue);font-family:var(--display);font-size:14px;letter-spacing:3px;cursor:pointer;transition:all .2s}
.rbtn:hover{background:rgba(0,170,255,.2);box-shadow:0 0 20px rgba(0,170,255,.2)}
.ph-h{background:rgba(0,229,196,.04);border:1px solid rgba(0,229,196,.2);border-radius:10px;padding:14px 16px;margin-bottom:12px}
.ph-t{font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--teal);margin-bottom:8px;text-transform:uppercase}
.ph-c{background:var(--void);border:1px solid var(--rim);border-radius:6px;padding:8px 12px;font-family:var(--mono);font-size:11px;color:var(--green);display:block;margin-top:4px;word-break:break-all;line-height:1.7}
.ph-n{font-family:var(--mono);font-size:9px;color:var(--text3);margin-top:8px;line-height:1.6}
.bat-big{font-family:var(--display);font-size:42px;font-weight:700;text-align:center;line-height:1}
.bat-tr{height:8px;background:var(--rim2);border-radius:4px;overflow:hidden;margin:10px 0}
.bat-f{height:100%;border-radius:4px;transition:width .8s,background .4s}
.pp{text-align:center;padding:28px}
.gb{padding:13px 32px;background:rgba(0,229,196,.1);border:1px solid var(--teal);border-radius:10px;color:var(--teal);font-family:var(--display);font-size:13px;letter-spacing:3px;cursor:pointer;transition:all .2s;margin-top:14px;display:inline-block}
.gb:hover{background:rgba(0,229,196,.2);box-shadow:0 0 20px rgba(0,229,196,.15)}
.pt-t{width:100%;border-collapse:collapse}
.pt-t th{font-family:var(--mono);font-size:7px;letter-spacing:2px;color:var(--text3);text-align:left;padding:4px 8px;border-bottom:1px solid var(--rim);text-transform:uppercase}
.pt-t td{font-family:var(--mono);font-size:10px;padding:5px 8px;border-bottom:1px solid var(--rim);color:var(--text2)}
.pt-t tr:last-child td{border:none}.pt-t td:first-child{color:var(--text)}
.pt-t td.warn{color:var(--yellow)}.pt-t td.crit{color:var(--red)}
</style></head><body>
<div id="stars"></div>
<div class="app">
<header class="hdr">
  <div class="hdr-logo">NEXUS <span>PM</span></div>
  <div class="hdr-chip">ISOLATION FOREST v4</div>
  <div class="hdr-st"><span class="pulse"></span><span id="clbl">CONNECTING</span></div>
  <div class="hdr-r">
    <div id="dtabs" class="dev-tabs"></div>
    <div><div class="clk" id="clk">--:--:--</div><div class="dtl" id="dtl"></div></div>
  </div>
</header>
<div class="modes">
  <button class="mbtn on" onclick="sm('live')"    id="mb-live">⬤ &nbsp;LIVE SENSOR</button>
  <button class="mbtn"    onclick="sm('ind')"     id="mb-ind">⬡ &nbsp;INDUSTRIAL</button>
  <button class="mbtn"    onclick="sm('custom')"  id="mb-custom">◧ &nbsp;CUSTOM INPUT</button>
</div>
<div class="banner NOMINAL" id="bnr">
  <div class="b-ico" id="b-ico">⏳</div>
  <div><div class="b-ttl" id="b-ttl" style="color:var(--green)">WAITING</div>
    <div class="b-sub" id="b-sub">Open on phone or run probe.py on PC</div></div>
  <div class="b-sc"><div class="b-num" id="b-num" style="color:var(--green)">--</div><div class="b-lbl">HEALTH SCORE / 100</div></div>
  <div class="b-bar"><div class="b-bar-f" id="b-barf" style="width:0%"></div></div>
</div>
<div id="m-live">
<div class="ph-h" id="ph-hint">
  <div class="ph-t">📡 For exact Windows Task Manager values — run probe.py on your PC:</div>
  <code class="ph-c">pip install psutil requests</code>
  <code class="ph-c">python probe.py</code>
  <div class="ph-n">probe.py sends real CPU%, RAM%, Disk%, Temperature, Battery, Processes — matching Task Manager exactly.<br>Without it, the browser sends estimated values.</div>
</div>
<div class="card pp" id="pp" style="display:none">
  <div class="ch" style="justify-content:center">PHONE SENSOR ACCESS</div>
  <div style="font-size:48px;margin:8px 0">📱</div>
  <div style="font-family:var(--mono);font-size:10px;color:var(--text2);margin-bottom:16px;letter-spacing:1px">Enable real accelerometer + battery readings</div>
  <button class="gb" onclick="startPhone()">▶ GRANT ACCESS</button>
</div>
<div class="mg">
<div class="lc">
  <div class="card">
    <div class="ch">Health Score</div>
    <div class="rw">
      <svg width="200" height="124" viewBox="0 0 200 124" style="overflow:visible">
        <defs>
          <linearGradient id="rg" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#00ff87"/><stop offset="50%" stop-color="#ffd24d"/><stop offset="100%" stop-color="#ff3355"/>
          </linearGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <path d="M18,114 A82,82 0 0,1 182,114" fill="none" stroke="#0e1e35" stroke-width="14" stroke-linecap="round"/>
        <path id="rf" d="M18,114 A82,82 0 0,1 182,114" fill="none" stroke="url(#rg)" stroke-width="14"
          stroke-linecap="round" stroke-dasharray="257.6" stroke-dashoffset="257.6"
          style="transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)" filter="url(#glow)"/>
        <path id="rg2" d="M18,114 A82,82 0 0,1 182,114" fill="none" stroke="#04080f" stroke-width="14"
          stroke-linecap="round" stroke-dasharray="257.6" stroke-dashoffset="0"
          style="transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)"/>
        <g stroke="#162840" stroke-width="1.5">
          <line x1="18" y1="114" x2="8" y2="114"/><line x1="182" y1="114" x2="192" y2="114"/>
          <line x1="100" y1="32" x2="100" y2="20"/><line x1="53" y1="57" x2="44" y2="48"/>
          <line x1="147" y1="57" x2="156" y2="48"/>
        </g>
        <text x="10" y="128" fill="#1e3a55" font-family="Orbitron" font-size="7">0</text>
        <text x="169" y="128" fill="#1e3a55" font-family="Orbitron" font-size="7">100</text>
        <text id="rn" x="100" y="100" fill="#00ff87" font-family="Orbitron" font-size="38" font-weight="700" text-anchor="middle" dominant-baseline="central" style="transition:fill .5s">--</text>
        <text x="100" y="118" fill="#1e3a55" font-family="JetBrains Mono" font-size="7" text-anchor="middle">OUT OF 100</text>
      </svg>
      <div class="rb NOMINAL" id="rb">AWAITING DATA</div>
    </div>
  </div>
  <div class="card">
    <div class="ch">Score Breakdown</div>
    <div class="bd">
      <div class="bdr"><span class="bdl">IF Anomaly (55%)</span><div class="bdt"><div class="bdf" id="bd-if" style="width:0%;background:var(--blue)"></div></div><span class="bdv" id="bv-if">--</span></div>
      <div class="bdr"><span class="bdl">Rule Penalty (45%)</span><div class="bdt"><div class="bdf" id="bd-rl" style="width:0%;background:var(--red)"></div></div><span class="bdv" id="bv-rl">--</span></div>
      <div class="bdr"><span class="bdl">Final Score</span><div class="bdt"><div class="bdf" id="bd-fn" style="width:0%;background:var(--green)"></div></div><span class="bdv" id="bv-fn">--</span></div>
    </div>
  </div>
  <div class="card">
    <div class="ch">Battery Intelligence</div>
    <div class="bat-big" id="bat-big" style="color:var(--green)">--%</div>
    <div class="bat-tr"><div class="bat-f" id="bat-f" style="width:0%;background:var(--green)"></div></div>
    <div style="font-family:var(--mono);font-size:9px;color:var(--text2);text-align:center;letter-spacing:1px" id="bat-i">--</div>
  </div>
</div>
<div class="cc">
  <div class="card"><div class="ch">Live Parameters</div><div class="mr4" id="mr4"><div style="font-family:var(--mono);font-size:10px;color:var(--text3);grid-column:span 2;padding:12px;text-align:center">Awaiting data…</div></div></div>
  <div class="card">
    <div class="ch">Fault Classification</div>
    <div class="fc" id="fca" style="background:rgba(0,255,135,.04);border-color:rgba(0,255,135,.2)">
      <div class="fn" id="fna" style="color:var(--green)">AWAITING SCAN</div>
      <div class="fd" id="fda">Run a scan to detect fault type</div>
    </div>
    <div class="ch" style="margin-top:10px">Active Alerts</div>
    <div class="al" id="alist"><div class="na">No alerts — system healthy</div></div>
  </div>
  <div class="card"><div class="ch">AI Recommendations</div><div id="rlist"><div style="color:var(--text3);font-family:var(--mono);font-size:10px">Awaiting analysis…</div></div></div>
</div>
<div class="rc">
  <div class="card"><div class="ch">System Metrics</div><div id="mrows"><div style="color:var(--text3);font-family:var(--mono);font-size:10px">Awaiting data…</div></div></div>
  <div class="card" id="spcard" style="display:none"><div class="ch">System Info</div><div class="sp" id="spgrid"></div></div>
  <div class="card" id="pcard" style="display:none"><div class="ch">Top Processes</div>
    <table class="pt-t"><thead><tr><th>Process</th><th>CPU%</th><th>RAM%</th></tr></thead><tbody id="pbody"></tbody></table>
  </div>
</div>
</div>
<div class="cs">
  <div class="cc2"><div class="cth">Health Score History</div><div class="cts" id="cs-s">Last 30 scans</div><div class="spk" id="sk-s"></div></div>
  <div class="cc2"><div class="cth">CPU + RAM Trend</div><div class="cts" id="cs-c">Correlated load</div><div class="spk" id="sk-c" style="gap:1.5px"></div></div>
  <div class="cc2"><div class="cth">Temperature Curve</div><div class="cts" id="cs-t">Celsius over time</div><div class="spk" id="sk-t"></div></div>
</div>
<div class="ms" id="metas">
  <div class="mc2"><div class="mk">Vibration State</div><div class="mv2" id="m-vib">--</div></div>
  <div class="mc2"><div class="mk">Thermal State</div><div class="mv2"  id="m-thm">--</div></div>
  <div class="mc2"><div class="mk">Battery Trend</div><div class="mv2"  id="m-btr">--</div></div>
  <div class="mc2"><div class="mk">Battery Life Est.</div><div class="mv2" id="m-blt">--</div></div>
  <div class="mc2"><div class="mk">CPU Trend</div><div class="mv2"      id="m-ctr">--</div></div>
</div>
</div>
<div id="m-ind" style="display:none"><div class="card"><div class="ch">Industrial Machine Parameters</div>
<div class="ig">
<div class="if2"><label>Vibration (m/s²)</label><input type="number" id="iv" value="0.08" step="0.01"><div class="hint">Normal &lt;0.35 · High &gt;0.7 · Critical &gt;1.2</div></div>
<div class="if2"><label>Temperature (°C)</label><input type="number" id="it" value="45" step="1"><div class="hint">Normal &lt;62 · Hot &gt;80</div></div>
<div class="if2"><label>CPU / Node Load (%)</label><input type="number" id="ic" value="55" step="1" min="0" max="100"><div class="hint">Task Manager CPU column</div></div>
<div class="if2"><label>Memory / RAM (%)</label><input type="number" id="im" value="50" step="1" min="0" max="100"><div class="hint">Task Manager Memory column</div></div>
<div class="if2"><label>Power / Battery (%)</label><input type="number" id="ib" value="90" step="1" min="0" max="100"></div>
<div class="if2"><label>Disk / Wear (%)</label><input type="number" id="id" value="40" step="1" min="0" max="100"><div class="hint">Task Manager Disk column</div></div>
</div><button class="rbtn" onclick="sInd()">⚙ RUN ML ANALYSIS</button></div></div>
<div id="m-custom" style="display:none"><div class="card"><div class="ch">Custom Parameter Input</div>
<div class="ig">
<div class="if2"><label>CPU %</label><input type="number" id="xc" value="50" step="1" min="0" max="100"></div>
<div class="if2"><label>RAM / Memory %</label><input type="number" id="xm" value="50" step="1" min="0" max="100"></div>
<div class="if2"><label>Temperature °C</label><input type="number" id="xt" value="42" step="1"></div>
<div class="if2"><label>Disk %</label><input type="number" id="xd" value="50" step="1" min="0" max="100"></div>
<div class="if2"><label>Vibration m/s²</label><input type="number" id="xv" value="0.05" step="0.01"></div>
<div class="if2"><label>Battery % (-1=no battery)</label><input type="number" id="xb" value="80" step="1" min="-1" max="100"></div>
</div><button class="rbtn" onclick="sCus()">🧠 ANALYZE WITH ML</button></div></div>
</div>
<script>
let selDev=null,curMode='live',liveInt,batLvl=null,batOK=false,lastVib=0;
(function(){const c=document.getElementById('stars');for(let i=0;i<120;i++){const s=document.createElement('div');s.className='star';const sz=Math.random()*2+0.5;s.style.cssText=`width:${sz}px;height:${sz}px;left:${Math.random()*100}%;top:${Math.random()*100}%;--lo:${(Math.random()*.1+.05).toFixed(2)};--hi:${(Math.random()*.7+.2).toFixed(2)};--d:${(Math.random()*4+2).toFixed(1)}s;--delay:-${(Math.random()*4).toFixed(1)}s`;c.appendChild(s);}})();
function tick(){const n=new Date();document.getElementById('clk').textContent=n.toLocaleTimeString('en-GB');document.getElementById('dtl').textContent=n.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}).toUpperCase();}
setInterval(tick,1000);tick();
function dn(){const u=navigator.userAgent;if(/iPhone/i.test(u))return'iPhone';if(/iPad/i.test(u))return'iPad';if(/Android/i.test(u))return'Android';if(/Mac OS X/i.test(u)&&!/iPhone|iPad/i.test(u))return'macOS';if(/Windows/i.test(u))return'Windows';if(/Linux/i.test(u))return'Linux';return'Unknown';}
function isIOS(){return/iPhone|iPad|iPod/i.test(navigator.userAgent);}
function isMob(){return/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);}
async function initBat(){if(isIOS()){batOK=false;return;}try{if(navigator.getBattery){const b=await navigator.getBattery();batLvl=Math.round(b.level*100);batOK=true;b.addEventListener('levelchange',()=>{batLvl=Math.round(b.level*100);});}}catch(e){batOK=false;}}
function listenMotion(){window.addEventListener('devicemotion',e=>{const a=e.accelerationIncludingGravity;if(!a||a.x===null)return;const mag=Math.sqrt((a.x||0)**2+(a.y||0)**2+(a.z||0)**2);lastVib=Math.round(Math.max(0,(mag-9.81)/9.81*0.5)*10000)/10000;});}
function startPhone(){if(typeof DeviceMotionEvent!=='undefined'&&typeof DeviceMotionEvent.requestPermission==='function'){DeviceMotionEvent.requestPermission().then(p=>{if(p==='granted'){listenMotion();document.getElementById('pp').style.display='none';}}).catch(()=>{});}else{listenMotion();document.getElementById('pp').style.display='none';}}
function startLive(){if(!isIOS()){if(typeof DeviceMotionEvent!=='undefined'&&DeviceMotionEvent.requestPermission){DeviceMotionEvent.requestPermission().then(s=>{if(s==='granted')listenMotion();}).catch(()=>{});}else{listenMotion();}}clearInterval(liveInt);doScan();liveInt=setInterval(doScan,4000);}
async function doScan(){const dev=dn();const bat=(batOK&&batLvl!==null)?batLvl:-1;let mem=50;if(navigator.deviceMemory)mem=Math.max(20,Math.min(85,Math.round(100-navigator.deviceMemory*8)));if(performance.memory){const p=Math.round(performance.memory.usedJSHeapSize/performance.memory.jsHeapSizeLimit*100);mem=Math.max(mem,p);}const vib=isMob()?lastVib:0.02;const cpu=isMob()?Math.round(20+lastVib*60):50;await par({device_type:dev,device:dev,battery:bat,vibration:vib,cpu:cpu,cpu_load:cpu,memory:mem,temperature:Math.round(32+(100-(bat>=0?bat:85))*0.18+lastVib*6),disk:50,is_charging:!batOK||bat<0||(batLvl!==null&&batLvl>95),source:'browser'});}
async function par(p){try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});const d=await r.json();selDev=d.device||selDev;applyData(d);await rdTabs();return d;}catch(e){document.getElementById('clbl').textContent='ERROR';}}
async function sInd(){await par({device:'Industrial',device_type:'Industrial',cpu:+document.getElementById('ic').value||55,memory:+document.getElementById('im').value||50,temp:+document.getElementById('it').value||45,disk:+document.getElementById('id').value||40,vibration:+document.getElementById('iv').value||0.08,battery:+document.getElementById('ib').value||90,is_charging:true,source:'industrial'});sm('live');}
async function sCus(){await par({device:'Custom',device_type:'Custom',cpu:+document.getElementById('xc').value||50,memory:+document.getElementById('xm').value||50,temp:+document.getElementById('xt').value||42,disk:+document.getElementById('xd').value||50,vibration:+document.getElementById('xv').value||0.05,battery:+document.getElementById('xb').value,is_charging:false,source:'custom'});sm('live');}
function col(sc){return sc>=65?'#ff3355':sc>=35?'#ffd24d':'#00ff87';}
function applyData(d){
if(!d||d.error)return;
const C=col(d.health_score||0);
const icons={NOMINAL:'✅',WARNING:'⚠️',CRITICAL:'🚨'};
const desc={NOMINAL:'All systems operating within healthy parameters',WARNING:'Elevated risk — schedule maintenance',CRITICAL:'HALT — immediate inspection required'};
document.getElementById('bnr').className='banner '+(d.risk||'NOMINAL');
document.getElementById('b-ico').textContent=icons[d.risk]||'⏳';
document.getElementById('b-ttl').textContent=d.risk||'--';
document.getElementById('b-ttl').style.color=C;
document.getElementById('b-sub').textContent=desc[d.risk]||d.status||'';
document.getElementById('b-num').textContent=d.health_score??'--';
document.getElementById('b-num').style.color=C;
const bf=document.getElementById('b-barf');bf.style.width=(d.health_score||0)+'%';bf.style.background=C;
const sc=d.health_score||0;
document.getElementById('rf').style.strokeDashoffset=257.6-(sc/100)*257.6;
document.getElementById('rg2').style.strokeDashoffset=(sc/100)*257.6;
document.getElementById('rn').textContent=sc;document.getElementById('rn').setAttribute('fill',C);
const rb=document.getElementById('rb');rb.textContent=sc>=65?'CRITICAL':sc>=35?'WARNING':'NOMINAL';
rb.className='rb '+(sc>=65?'CRITICAL':sc>=35?'WARNING':'NOMINAL');
const ifc=d.if_component||0;const pen=d.penalty||0;
document.getElementById('bd-if').style.width=ifc+'%';
document.getElementById('bd-rl').style.width=Math.min(100,pen/60*100)+'%';
document.getElementById('bd-fn').style.width=sc+'%';document.getElementById('bd-fn').style.background=C;
document.getElementById('bv-if').textContent=ifc.toFixed(1);
document.getElementById('bv-rl').textContent='+'+pen;
document.getElementById('bv-fn').textContent=sc+'/100';
const bat=d.battery;const bc=bat<0?'var(--blue)':bat<15?'var(--red)':bat<30?'var(--yellow)':'var(--green)';
document.getElementById('bat-big').textContent=bat<0?'AC':(bat+'%');document.getElementById('bat-big').style.color=bc;
document.getElementById('bat-f').style.width=Math.max(0,bat)+'%';document.getElementById('bat-f').style.background=bc;
document.getElementById('bat-i').textContent=d.is_charging?'⚡ Charging — '+d.bat_life:bat>=0?'🔋 '+d.bat_life:'🔌 AC Power';
const ps=[{l:'CPU',v:d.cpu,u:'%',mx:100,c:d.cpu>90?'#ff3355':d.cpu>75?'#ffd24d':'#0af',t:d.cpu>90?'SATURATED':d.cpu>75?'HIGH':'OK'},
{l:'MEMORY',v:d.memory,u:'%',mx:100,c:d.memory>90?'#ff3355':d.memory>75?'#ffd24d':'#a78bfa',t:d.memory>90?'CRITICAL':d.memory>75?'HIGH':'OK'},
{l:'TEMP',v:d.temp,u:'°C',mx:110,c:d.temp>80?'#ff3355':d.temp>65?'#ff8c42':'#00e5c4',t:d.thermal||'NORMAL'},
{l:'DISK',v:d.disk,u:'%',mx:100,c:d.disk>90?'#ff3355':d.disk>80?'#ffd24d':'#00ff87',t:d.disk>90?'CRITICAL':d.disk>80?'HIGH':'OK'}];
document.getElementById('mr4').innerHTML=ps.map((p,i)=>{const pct=Math.min(100,p.v/p.mx*100);const circ=2*Math.PI*36;const off=circ*(1-pct/100);const tg=p.t==='OK'||p.t==='NORMAL'?'tn':p.t==='CRITICAL'?'tc':'tw';
return`<div class="m4"><svg width="84" height="52" viewBox="0 0 84 52" style="overflow:visible"><defs><filter id="g${i}"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><path d="M8,48 A36,36 0 0,1 76,48" fill="none" stroke="#0e1e35" stroke-width="8" stroke-linecap="round"/><path d="M8,48 A36,36 0 0,1 76,48" fill="none" stroke="${p.c}" stroke-width="8" stroke-linecap="round" stroke-dasharray="${circ*0.707}" stroke-dashoffset="${circ*0.707*(1-pct/100)}" style="transition:stroke-dashoffset .7s ease" filter="url(#g${i})"/></svg><div class="m4v" style="color:${p.c};margin-top:-4px">${typeof p.v==='number'?p.v.toFixed(p.u==='°C'?1:0):p.v}${p.u}</div><div class="m4l">${p.l}</div><div class="m4t tag ${tg}">${p.t}</div></div>`;}).join('');
const fC={'NORMAL':['rgba(0,255,135,.04)','rgba(0,255,135,.2)','#00ff87'],'OVERHEATING':['rgba(255,51,85,.08)','rgba(255,51,85,.4)','#ff3355'],'CPU OVERLOAD':['rgba(255,140,66,.07)','rgba(255,140,66,.35)','#ff8c42'],'MEMORY PRESSURE':['rgba(167,139,250,.07)','rgba(167,139,250,.3)','#a78bfa'],'DISK CRITICAL':['rgba(255,210,77,.06)','rgba(255,210,77,.3)','#ffd24d'],'BEARING FAULT':['rgba(0,229,196,.06)','rgba(0,229,196,.25)','#00e5c4'],'LOW POWER':['rgba(255,51,85,.08)','rgba(255,51,85,.4)','#ff3355']};
const fc=fC[d.fault]||fC['NORMAL'];
const fcard=document.getElementById('fca');fcard.style.background=fc[0];fcard.style.borderColor=fc[1];
document.getElementById('fna').textContent=d.fault||'NORMAL';document.getElementById('fna').style.color=fc[2];
document.getElementById('fda').textContent=d.fault_desc||'All parameters within healthy envelope';
const trig=d.triggered||[];
document.getElementById('alist').innerHTML=trig.length?trig.map(t=>`<div class="ai ${t[1]}"><span class="asv">[${t[0]}] ${t[1]}</span><span class="amg">${t[2]}</span></div>`).join(''):'<div class="na">✓ No active alerts</div>';
document.getElementById('rlist').innerHTML=(d.recommendations||[]).map((r,i)=>`<div class="ri"><span class="rn">${String(i+1).padStart(2,'0')}.</span>${r}</div>`).join('');
const mx=[{ico:'⚙️',n:'CPU Load',v:d.cpu+'%',b:d.cpu,c:d.cpu>90?'var(--red)':d.cpu>75?'var(--yellow)':'var(--blue)'},
{ico:'💾',n:'Memory',v:d.memory+'%',b:d.memory,c:d.memory>90?'var(--red)':d.memory>75?'var(--yellow)':'var(--purple)'},
{ico:'🌡️',n:'Temperature',v:d.temp+'°C',b:Math.min(100,d.temp),c:d.temp>80?'var(--red)':d.temp>65?'var(--orange)':'var(--teal)'},
{ico:'💿',n:'Disk',v:d.disk+'%',b:d.disk,c:d.disk>90?'var(--red)':d.disk>80?'var(--yellow)':'var(--green)'},
{ico:'📳',n:'Vibration',v:(d.vibration||0).toFixed(4),b:Math.min(100,(d.vibration||0)/2*100),c:(d.vibration||0)>1?'var(--red)':(d.vibration||0)>0.5?'var(--orange)':'var(--teal)'}];
document.getElementById('mrows').innerHTML=mx.map(m=>`<div class="mrow"><div class="mic" style="background:${m.c.replace('var(','').replace(')','').replace('--','')}18">${m.ico}</div><div style="flex:1;min-width:0"><div class="mn">${m.n}</div><div class="mv" style="color:${m.c}">${m.v}</div><div class="mbar"><div class="mbf" style="width:${m.b}%;background:${m.c}"></div></div></div></div>`).join('');
if(d.cpu_freq||d.ram_total){
document.getElementById('spcard').style.display='block';
document.getElementById('ph-hint').style.display='none';
const sp=[{l:'Device',v:d.device||'--'},{l:'Platform',v:d.platform||'--'},{l:'CPU Freq',v:d.cpu_freq?d.cpu_freq+' MHz':'--'},{l:'Cores',v:d.cpu_cores_logical?d.cpu_cores_logical+'L/'+d.cpu_cores_physical+'P':'--'},{l:'RAM',v:d.ram_total?d.ram_used+'/'+d.ram_total+' GB':'--'},{l:'RAM Free',v:d.ram_avail?d.ram_avail+' GB':'--'},{l:'Disk',v:d.disk_total?d.disk_used+'/'+d.disk_total+' GB':'--'},{l:'Processes',v:d.proc_count||'--'},{l:'Disk R',v:d.disk_read_mb!=null?d.disk_read_mb+' MB/s':'--'},{l:'Disk W',v:d.disk_write_mb!=null?d.disk_write_mb+' MB/s':'--'},{l:'Net ↑',v:d.net_send_mb!=null?d.net_send_mb+' MB/s':'--'},{l:'Net ↓',v:d.net_recv_mb!=null?d.net_recv_mb+' MB/s':'--'}];
document.getElementById('spgrid').innerHTML=sp.map(s=>`<div class="sc"><div class="sl">${s.l}</div><div class="sv">${s.v}</div></div>`).join('');}
if(d.top_procs&&d.top_procs.length){document.getElementById('pcard').style.display='block';document.getElementById('pbody').innerHTML=d.top_procs.map(p=>`<tr><td>${p.name}</td><td class="${p.cpu>50?'crit':p.cpu>20?'warn':''}">${p.cpu}%</td><td>${p.mem}%</td></tr>`).join('');}
const vt=d.vibration>1?'tc':d.vibration>0.5?'tw':'tn';const tt=d.thermal==='OVERHEATING'||d.thermal==='HOT'?'tc':d.thermal==='WARM'?'tw':'tn';
const bt=d.bat_trend==='DRAINING'?'tw':d.bat_trend==='CHARGING'?'tb':'tn';const ct=d.cpu_trend==='RISING'?'tw':'tn';
document.getElementById('m-vib').innerHTML=`<span class="tag ${vt}">${d.vib_label||'--'}</span>`;
document.getElementById('m-thm').innerHTML=`<span class="tag ${tt}">${d.thermal||'--'}</span>`;
document.getElementById('m-btr').innerHTML=`<span class="tag ${bt}">${d.bat_trend||'STABLE'}</span>`;
document.getElementById('m-blt').textContent=d.bat_life||'--';
document.getElementById('m-ctr').innerHTML=`<span class="tag ${ct}">${d.cpu_trend||'STABLE'}</span>`;}
function mkSpk(cid,vals,cfn,mx){const c=document.getElementById(cid);if(!vals||!vals.length){c.innerHTML='<span style="color:var(--text3);font-family:var(--mono);font-size:9px">COLLECTING…</span>';return;}const m=mx||Math.max(...vals,1);c.innerHTML=vals.map(v=>`<div class="spb" style="height:${Math.max(4,Math.min(100,v/m*100))}%;background:${cfn(v)}"></div>`).join('');}
async function rdTabs(){try{const r=await fetch('/data');const d=await r.json();if(d.error)return;document.getElementById('clbl').textContent='ONLINE';const devs=d.all_devices||{};const names=Object.keys(devs);
document.getElementById('dtabs').innerHTML=names.map(n=>`<button class="dev-tab ${n===selDev?'on':''}" onclick="swDev('${n}')">${n.toUpperCase()}</button>`).join('');
const sh=d.score_hist||{};const ch=d.cpu_hist||{};const th=d.tmp_hist||{};const rh=sh[selDev]||[];const cph=ch[selDev]||[];const tph=th[selDev]||[];
mkSpk('sk-s',rh,v=>v>=65?'#ff3355':v>=35?'#ffd24d':'#00ff87',100);
mkSpk('sk-c',cph,v=>v>=90?'#ff3355':v>=75?'#ffd24d':'#0af',100);
mkSpk('sk-t',tph,v=>v>=80?'#ff3355':v>=65?'#ff8c42':'#00e5c4',110);
document.getElementById('cs-s').textContent=`Last ${rh.length} scans · ${selDev||''}`;
document.getElementById('cs-c').textContent=`Avg ${cph.length?Math.round(cph.reduce((a,b)=>a+b,0)/cph.length):0}% · ${cph.length} scans`;
document.getElementById('cs-t').textContent=`Max ${tph.length?Math.max(...tph).toFixed(1):0}°C in window`;}catch(e){}}
function swDev(n){selDev=n;rdTabs().then(()=>{fetch('/data').then(r=>r.json()).then(d=>{if(d.all_devices&&d.all_devices[n])applyData(d.all_devices[n]);});});}
function sm(m){curMode=m;['live','ind','custom'].forEach(id=>{document.getElementById('m-'+id).style.display=id===m?'block':'none';document.getElementById('mb-'+id).classList.toggle('on',id===m);});if(m==='live')startLive();else clearInterval(liveInt);}
selDev=dn();if(isMob())document.getElementById('pp').style.display='block';
initBat().then(()=>startLive());setInterval(rdTabs,5000);rdTabs();
</script></body></html>"""

@app.get("/monitor", response_class=HTMLResponse)
async def monitor(): return DASHBOARD_HTML

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)