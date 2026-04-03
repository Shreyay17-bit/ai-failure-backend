from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.network.urlrequest import UrlRequest
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.utils import platform
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
            width: 8
            circle: (self.center_x, self.center_y, min(self.size)/2 - 10)

        Color:
            rgba: 0, 1, 0, 1
        Line:
            width: 8
            circle: (self.center_x, self.center_y, min(self.size)/2 - 10, 0, self.value * 3.6)

<SensorCard@BoxLayout>:
    orientation: 'vertical'
    padding: dp(15)
    opacity: 1
    canvas.before:
        Color:
            rgba: hex("#1e1e1e")
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(15),]

BoxLayout:
    orientation: 'vertical'
    padding: dp(25)
    spacing: dp(20)

    canvas.before:
        Color:
            rgba: 0.05, 0.05, 0.1, 1
        Rectangle:
            pos: self.pos
            size: self.size

    Label:
        text: "NEURAL SYSTEM MONITOR"
        font_size: '16sp'
        color: hex("#00FFC8")
        bold: True
        size_hint_y: None
        height: dp(40)

    BoxLayout:
        orientation: 'vertical'
        size_hint_y: 0.5

        CircularProgress:
            id: bar
            size_hint: None, None
            size: dp(180), dp(180)
            pos_hint: {"center_x": 0.5}

        Label:
            id: risk_val
            text: "0%"
            font_size: '75sp'
            bold: True
            color: hex("#00FF00")

            canvas.before:
                Color:
                    rgba: self.color[0], self.color[1], self.color[2], 0.25
                Rectangle:
                    pos: self.x - 10, self.y - 10
                    size: self.width + 20, self.height + 20

    GridLayout:
        cols: 3
        spacing: dp(12)
        size_hint_y: 0.3
        
        SensorCard:
            Label:
                text: "CPU"
                font_size: '12sp'
                color: hex("#666666")
            Label:
                id: cpu_text
                text: "0%"
                font_size: '18sp'
                bold: True
        
        SensorCard:
            Label:
                text: "RAM"
                font_size: '12sp'
                color: hex("#666666")
            Label:
                id: ram_text
                text: "0%"
                font_size: '18sp'
                bold: True

        SensorCard:
            Label:
                text: "BATT"
                font_size: '12sp'
                color: hex("#666666")
            Label:
                id: bat_text
                text: "0%"
                font_size: '18sp'
                bold: True

    Label:
        id: analysis
        text: "AI Status: Syncing Hardware..."
        font_size: '14sp'
        italic: True
        color: hex("#888888")
        size_hint_y: None
        height: dp(30)
"""

class FailurePredictor(App):
    def build(self):
        self.title = "AI Failure Predictor"
        self.ui = Builder.load_string(KV)

        Clock.schedule_interval(self.update_telemetry, 1.5)
        Clock.schedule_once(self.start_animations, 1)

        return self.ui

    def start_animations(self, *args):
        for widget in self.ui.walk():
            if isinstance(widget, BoxLayout):
                anim = Animation(opacity=0.85, duration=1) + Animation(opacity=1, duration=1)
                anim.repeat = True
                anim.start(widget)

    def update_telemetry(self, dt):
        self.ui.ids.analysis.text = "Analyzing system..."

        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        batt = psutil.sensors_battery()
        batt_p = batt.percent if batt else 100.0
        
        payload = {
            "device_type": platform,
            "battery": float(batt_p),
            "cpu_load": float(cpu),
            "memory": float(mem)
        }
        
        UrlRequest(
            "http://127.0.0.1:8000/predict",
            req_body=json.dumps(payload),
            req_headers={'Content-type': 'application/json'},
            on_success=self.on_api_success,
            on_error=lambda r, e: self.set_offline()
        )

    def animate_number(self, label, target):
        start = float(label.text.replace('%', '') or 0)

        def update(dt):
            nonlocal start
            if abs(start - target) < 1:
                label.text = f"{int(target)}%"
                return False
            start += (target - start) * 0.2
            label.text = f"{int(start)}%"

        Clock.schedule_interval(update, 0.02)

    def on_api_success(self, req, res):
        risk = float(res['risk'].replace('%',''))

        # animate circular bar
        Animation(value=risk, duration=0.7).start(self.ui.ids.bar)

        # animate number
        self.animate_number(self.ui.ids.risk_val, risk)

        # update metrics
        self.ui.ids.cpu_text.text = res['raw_metrics']['CPU']
        self.ui.ids.ram_text.text = res['raw_metrics']['RAM']
        self.ui.ids.bat_text.text = res['raw_metrics']['BATT']
        self.ui.ids.analysis.text = f"AI Insight: {res['analysis']}"

        from kivy.utils import get_color_from_hex as hex
        color = hex("#00FF00") if risk < 35 else hex("#FFA500") if risk < 70 else hex("#FF3333")

        Animation(color=color, duration=0.5).start(self.ui.ids.risk_val)

    def set_offline(self):
        self.ui.ids.analysis.text = "⚠️ Backend Offline"

if __name__ == "__main__":
    FailurePredictor().run()