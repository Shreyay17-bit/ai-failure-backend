from kivy.app import App
from kivy.lang import Builder
from kivy.network.urlrequest import UrlRequest
from kivy.clock import Clock
from kivy.utils import platform as kivy_platform
import psutil
import json

KV = """
#:import hex kivy.utils.get_color_from_hex
#:import dp kivy.metrics.dp

<CircularProgress@Widget>:
    value: 0
    canvas:
        Color:
            rgba: 0.2, 0.2, 0.2, 1
        Line:
            width: dp(8)
            circle: (self.center_x, self.center_y, min(self.size)/2 - dp(10))
        Color:
            rgba: (0, 1, 0.8, 1) if self.value < 40 else (1, 0.6, 0, 1) if self.value < 75 else (1, 0.2, 0.2, 1)
        Line:
            width: dp(8)
            circle: (self.center_x, self.center_y, min(self.size)/2 - dp(10), 0, self.value * 3.6)

BoxLayout:
    orientation: 'vertical'
    padding: dp(20)
    spacing: dp(15)
    canvas.before:
        Color:
            rgba: 0.05, 0.05, 0.1, 1
        Rectangle:
            pos: self.pos
            size: self.size

    Label:
        text: "NEURAL SYSTEM MONITOR"
        font_size: '18sp'
        color: hex("#00FFC8")
        bold: True
        size_hint_y: None
        height: dp(50)

    BoxLayout:
        orientation: 'vertical'
        size_hint_y: 0.6
        CircularProgress:
            id: bar
            size_hint: None, None
            size: dp(220), dp(220)
            pos_hint: {"center_x": 0.5}
        Label:
            id: risk_val
            text: "0%"
            font_size: '55sp'
            bold: True

    GridLayout:
        cols: 3
        spacing: dp(10)
        size_hint_y: 0.2
        BoxLayout:
            orientation: 'vertical'
            Label:
                text: "CPU"
                color: hex("#888888")
            Label:
                id: cpu_text
                text: "0%"
        BoxLayout:
            orientation: 'vertical'
            Label:
                text: "RAM"
                color: hex("#888888")
            Label:
                id: ram_text
                text: "0%"
        BoxLayout:
            orientation: 'vertical'
            Label:
                text: "BATT"
                color: hex("#888888")
            Label:
                id: bat_text
                text: "0%"

    Label:
        id: analysis
        text: "AI Status: Connecting to Render..."
        italic: True
        color: hex("#AAAAAA")
        size_hint_y: None
        height: dp(40)
"""

class FailurePredictor(App):
    def build(self):
        # UPDATE: Pointing to your cloud backend
        self.api_url = "https://ai-failure-backend.onrender.com/predict"
        self.ui = Builder.load_string(KV)
        # Slower interval (3s) to respect free-tier Render limits
        Clock.schedule_interval(self.update_telemetry, 3.0)
        return self.ui

    def update_telemetry(self, dt):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        batt = psutil.sensors_battery()
        batt_p = batt.percent if batt else 100.0
        
        payload = {
            "device_type": str(kivy_platform),
            "battery": float(batt_p),
            "cpu_load": float(cpu),
            "memory": float(mem)
        }
        
        # Adding a timeout=5 to prevent the app from "hanging"
        UrlRequest(
            self.api_url,
            req_body=json.dumps(payload),
            req_headers={'Content-type': 'application/json'},
            on_success=self.on_api_success,
            on_failure=lambda r, e: self.set_status("⚠️ Server waking up..."),
            on_error=lambda r, e: self.set_status("⚠️ Check Internet"),
            timeout=5
        )

    def on_api_success(self, req, res):
        risk_str = res.get('risk', '0%').replace('%', '')
        risk = float(risk_str)
        self.ui.ids.bar.value = risk
        self.ui.ids.risk_val.text = f"{int(risk)}%"
        
        metrics = res.get('raw_metrics', {})
        self.ui.ids.cpu_text.text = metrics.get('CPU', '0%')
        self.ui.ids.ram_text.text = metrics.get('RAM', '0%')
        self.ui.ids.bat_text.text = metrics.get('BATT', '0%')
        self.ui.ids.analysis.text = f"AI Insight: {res.get('analysis', 'Stable')}"

    def set_status(self, msg):
        self.ui.ids.analysis.text = msg

if __name__ == "__main__":
    FailurePredictor().run()