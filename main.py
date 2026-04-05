from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.network.urlrequest import UrlRequest
import json, math, platform

# REPLACE with your actual Render URL
SERVER_URL = "https://ai-failure-backend.onrender.com" 

class NeuralProbeApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=50, spacing=30, **kwargs)
        
        self.add_widget(Label(text="NEURAL HARDWARE PROBE", font_size='24sp', bold=True, color=(0, 1, 0.8, 1)))
        
        self.health_label = Label(text="SYNCING...", font_size='48sp', bold=True)
        self.stats_label  = Label(text="Vib: 0.000 G", font_size='20sp', color=(0.7, 0.7, 0.7, 1))
        self.status_msg   = Label(text="Connecting to AI Server...", font_size='16sp', italic=True)

        self.add_widget(self.health_label)
        self.add_widget(self.stats_label)
        self.add_widget(self.status_msg)

        # Bridge Loop: 1.5 second intervals
        Clock.schedule_interval(self.hardware_bridge, 1.5)

    def hardware_bridge(self, dt):
        v_mag = 0.0
        try:
            from plyer import accelerometer
            accelerometer.enable()
            acc = accelerometer.acceleration # Raw 3-axis data
            if acc and acc[0] is not None:
                # Vector Magnitude Math
                v_mag = abs(math.sqrt(acc[0]**2 + acc[1]**2 + acc[2]**2) - 9.81)
        except:
            v_mag = 0.05 

        self.stats_label.text = f"Live Vibration: {round(v_mag, 3)} G"

        payload = {"vibration": round(v_mag, 3), "device_type": "Android-Hardware-Link"}

        UrlRequest(
            f"{SERVER_URL}/predict",
            req_body=json.dumps(payload),
            req_headers={'Content-Type': 'application/json'},
            on_success=self.update_ui,
            on_failure=lambda r, e: setattr(self.status_msg, 'text', "Cloud Error")
        )

    def update_ui(self, req, res):
        self.health_label.text = f"{res['score']}% HEALTH"
        self.status_msg.text = res['risk']
        # Dynamic color coding
        if res['risk'] == "CRITICAL": self.health_label.color = (1, 0.2, 0.2, 1)
        else: self.health_label.color = (0, 1, 0.5, 1)

class PredictApp(App):
    def build(self): return NeuralProbeApp()

if __name__ == "__main__":
    PredictApp().run()