from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.network.urlrequest import UrlRequest
import json, platform
from plyer import battery, accelerometer

# IMPORTANT: Change 'localhost' to your PC's IP address (e.g., 192.168.1.5)
# UPDATE THIS LINE
SERVER_URL = "http://172.20.10.11:8000/predict"

class PdMProbe(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=10, **kwargs)
        self.add_widget(Label(text="NEURAL PROBE AI", font_size='24sp', bold=True))
        self.status = Label(text="System Ready", color=(0, 1, 0.8, 1))
        self.add_widget(self.status)
        
        btn = Button(text="START SCAN", size_hint_y=None, height=100)
        btn.bind(on_press=self.scan_and_send)
        self.add_widget(btn)

    def scan_and_send(self, instance):
        try:
            accelerometer.enable()
            v = sum(abs(x) for x in (accelerometer.acceleration or [0,0,0]))
            b = battery.status.get('percentage', 100)
        except: v, b = 0.2, 100 # PC Fallback

        payload = {"device_type": platform.system(), "battery": b, "cpu_load": 55.0, "memory": 40.0, "vibration": v}
        UrlRequest(SERVER_URL, req_body=json.dumps(payload), req_headers={'Content-Type': 'application/json'},
                   on_success=self.on_ok, on_failure=self.on_fail)

    def on_ok(self, req, res): self.status.text = f"AI Result: {res['analysis']}\nRisk: {res['risk']}"
    def on_fail(self, req, err): self.status.text = "Connection Failed!"

class PdMApp(App):
    def build(self): return PdMProbe()

if __name__ == "__main__": PdMApp().run()