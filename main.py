from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.network.urlrequest import UrlRequest
import json, platform

try:
    from plyer import battery, accelerometer
    PLYER_OK = True
except:
    PLYER_OK = False

SERVER_URL = "https://ai-failure-backend.onrender.com"

class PdMProbe(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=10, **kwargs)

        self.add_widget(Label(text="NEURAL PROBE AI", font_size='24sp', bold=True, size_hint_y=None, height=60))

        self.status  = Label(text="System Ready", color=(0, 1, 0.8, 1), font_size='16sp')
        self.result  = Label(text="Press START SCAN to begin", color=(0.5, 0.8, 0.7, 1), font_size='13sp')
        self.details = Label(text="", color=(0.4, 0.6, 0.5, 1), font_size='11sp')

        self.add_widget(self.status)
        self.add_widget(self.result)
        self.add_widget(self.details)

        btn = Button(text="START SCAN", size_hint_y=None, height=100,
                     background_color=(0, 0.4, 0.3, 1), font_size='18sp', bold=True)
        btn.bind(on_press=self.scan_and_send)
        self.add_widget(btn)

        self.add_widget(Label(
            text=f"Dashboard: {SERVER_URL}/monitor",
            color=(0, 0.8, 0.6, 0.6), font_size='10sp', size_hint_y=None, height=30
        ))

    def scan_and_send(self, instance):
        self.status.text  = "Scanning sensors..."
        self.status.color = (1, 0.8, 0, 1)

        v, b = 0.2, 100
        if PLYER_OK:
            try:
                accelerometer.enable()
                acc = accelerometer.acceleration
                if acc and acc[0] is not None:
                    v = round(sum(abs(x) for x in acc) / 30.0, 3)
            except:
                pass
            try:
                b = battery.status.get('percentage', 100) or 100
            except:
                pass

        try:
            import psutil
            cpu_load = psutil.cpu_percent(interval=0.5)
            mem      = psutil.virtual_memory().percent
        except:
            cpu_load, mem = 55.0, 40.0

        payload = {
            "device_type": platform.system() or "Mobile",
            "battery":     b,
            "cpu_load":    cpu_load,
            "memory":      mem,
            "vibration":   v,
        }
        self.details.text = f"Sending: vib={v}  bat={b}%  cpu={cpu_load}%  mem={mem}%"

        UrlRequest(
            SERVER_URL + "/predict",
            req_body=json.dumps(payload),
            req_headers={"Content-Type": "application/json"},
            on_success=self.on_ok,
            on_failure=self.on_fail,
            on_error=self.on_error,
        )

    def on_ok(self, req, res):
        risk     = res.get("risk", "UNKNOWN")
        analysis = res.get("analysis", "No analysis")
        score    = res.get("score", 0)
        color_map = {
            "NOMINAL":  (0, 1, 0.8, 1),
            "WARNING":  (1, 0.85, 0, 1),
            "CRITICAL": (1, 0.3, 0.3, 1),
        }
        self.status.text  = f"RISK: {risk}"
        self.status.color = color_map.get(risk, (1, 1, 1, 1))
        self.result.text  = analysis
        self.details.text = f"Score: {score}/100 | See {SERVER_URL}/monitor"

    def on_fail(self, req, res):
        self.status.text  = "Server Error!"
        self.status.color = (1, 0.3, 0.3, 1)
        self.result.text  = str(res)[:80]

    def on_error(self, req, err):
        self.status.text  = "Connection Failed!"
        self.status.color = (1, 0.3, 0.3, 1)
        self.result.text  = f"Cannot reach {SERVER_URL}"
        self.details.text = "Check internet or server URL"


class PdMApp(App):
    def build(self):
        return PdMProbe()

if __name__ == "__main__":
    PdMApp().run()