import threading
import time

from fovea import GazeSettings, WebcamEventSource


class FoveaBridge:

    def __init__(self):

        self.source = None
        self.thread = None
        self.running = False

        self.lock = threading.Lock()

        # Latest gaze information
        self.gaze_x = 0.5
        self.gaze_y = 0.5
        self.confidence = 0.0
        self.tracking_active = False

        # Latest webcam frame from Fovea
        self.latest_frame = None

    def start(self):

        try:

            settings = GazeSettings()

            self.source = WebcamEventSource(
                settings=settings,
                device_index=0,
                width=640,
                height=480,
                mirror=True,
                show_calibration=False,
            )

            self.running = True

            self.thread = threading.Thread(
                target=self._run,
                daemon=True
            )

            self.thread.start()

            print("[Fovea] Single-camera gaze engine started.")

        except Exception as e:

            print(f"[Fovea] Failed to start: {e}")

    def _run(self):

        try:

            for event in self.source.events():

                if not self.running:
                    break

                # ─────────────────────────────────────────────
                # Get latest camera frame
                # ─────────────────────────────────────────────

                frame = self.source.get_latest_frame()

                if frame is not None:

                    with self.lock:
                        self.latest_frame = frame

                # ─────────────────────────────────────────────
                # Gaze point
                # ─────────────────────────────────────────────

                if type(event).__name__ == "GazePoint":

                    with self.lock:

                        self.gaze_x = float(
                            event.x
                        )

                        self.gaze_y = float(
                            event.y
                        )

                        self.confidence = float(
                            event.confidence
                        )

                        self.tracking_active = (
                            self.confidence > 0.0
                        )

                # ─────────────────────────────────────────────
                # Tracking state
                # ─────────────────────────────────────────────

                elif type(event).__name__ == "TrackingState":

                    status = (
                        getattr(event, "status", None)
                        or getattr(event, "state", None)
                    )

                    if status is not None:

                        with self.lock:

                            self.tracking_active = (
                                str(status).lower()
                                == "active"
                            )

        except Exception as e:

            if self.running:

                print(
                    f"[Fovea] Runtime error: {e}"
                )

    def get_gaze(self):

        with self.lock:

            return (
                self.gaze_x,
                self.gaze_y,
                self.confidence,
                self.tracking_active,
            )

    def get_frame(self):

        with self.lock:

            if self.latest_frame is None:
                return None

            return self.latest_frame.copy()

    def stop(self):

        self.running = False

        try:

            if self.source is not None:
                self.source.close()

        except Exception:
            pass

        self.source = None