from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.network.urlrequest import UrlRequest

import json, math

SERVER_URL = "https://ai-failure-backend.onrender.com"


class NeuralProbeApp(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=20, **kwargs)

        self.add_widget(Label(text="AI DIAGNOSTICS", font_size='24sp'))

        grid = GridLayout(cols=2)

        self.health = Label(text="--%", font_size='32sp')
        self.vib = Label(text="0 G")
        self.trend = Label(text="--")
        self.freq = Label(text="0")
        self.status = Label(text="Connecting...")

        grid.add_widget(Label(text="Health"))
        grid.add_widget(self.health)

        grid.add_widget(Label(text="Vibration"))
        grid.add_widget(self.vib)

        grid.add_widget(Label(text="Trend"))
        grid.add_widget(self.trend)

        grid.add_widget(Label(text="Frequency"))
        grid.add_widget(self.freq)

        self.add_widget(grid)
        self.add_widget(self.status)

        Clock.schedule_interval(self.send_data, 1.5)

    def send_data(self, dt):
        v_mag = 0.05

        try:
            from plyer import accelerometer
            accelerometer.enable()
            acc = accelerometer.acceleration

            if acc and acc[0] is not None:
                v_mag = abs(math.sqrt(acc[0]**2 + acc[1]**2 + acc[2]**2) - 9.81)

        except:
            pass

        payload = {
            "vibration": round(v_mag, 3),
            "device": "mobile"
        }

        UrlRequest(
            f"{SERVER_URL}/predict",
            req_body=json.dumps(payload),
            req_headers={'Content-Type': 'application/json'},
            on_success=self.update_ui,
            on_failure=lambda r, e: setattr(self.status, 'text', "Server Error")
        )

    def update_ui(self, req, res):
        self.health.text = f"{res.get('score', '--')}%"
        self.vib.text = f"{res.get('vibration', 0)} G"
        self.trend.text = res.get('trend', 'N/A')
        self.freq.text = str(res.get('frequency', '0'))
        self.status.text = res.get('risk', 'UNKNOWN')

        if res.get('risk') == "CRITICAL":
            self.health.color = (1, 0, 0, 1)
        elif res.get('risk') == "WARNING":
            self.health.color = (1, 0.6, 0, 1)
        else:
            self.health.color = (0, 1, 0, 1)


class PredictApp(App):
    def build(self):
        return NeuralProbeApp()


if __name__ == "__main__":
    PredictApp().run()