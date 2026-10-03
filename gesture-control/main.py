import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import speech_recognition as sr

from collections import deque
from threading import Thread, Lock

# ── AirSwipe modules ──────────────────────────────────────────────────────────

from config import (
    MEDIAPIPE_OPTS,
    CAM_W,
    CAM_H,
    HISTORY_LEN
)

from gesture_classifier import (
    to_px,
    classify
)

from gesture_action import (
    make_state,
    execute_action
)

from display import (
    draw_skeleton,
    draw_hud
)

# ── Fovea ─────────────────────────────────────────────────────────────────────

from eye.fovea_bridge import FoveaBridge

from mediapipe.tasks.python import vision as mp_vision


# ═════════════════════════════════════════════════════════════════════════════
# BLINK DETECTION
# ═════════════════════════════════════════════════════════════════════════════

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, eye_indices):

    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical_1 = np.linalg.norm(
        np.array([p2.x, p2.y]) -
        np.array([p6.x, p6.y])
    )

    vertical_2 = np.linalg.norm(
        np.array([p3.x, p3.y]) -
        np.array([p5.x, p5.y])
    )

    horizontal = np.linalg.norm(
        np.array([p1.x, p1.y]) -
        np.array([p4.x, p4.y])
    )

    if horizontal == 0:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )


class DoubleBlinkDetector:

    def __init__(self):

        self.EAR_THRESHOLD = 0.20
        self.blinking = False
        self.blinks = deque(maxlen=4)

        self.DOUBLE_BLINK_WINDOW = 1.2
        self.CLICK_COOLDOWN = 1.0
        self.last_click_time = 0

    def update(self, face_landmarks):

        if face_landmarks is None:
            return False

        left_ear = eye_aspect_ratio(
            face_landmarks,
            LEFT_EYE
        )

        right_ear = eye_aspect_ratio(
            face_landmarks,
            RIGHT_EYE
        )

        ear = (
            left_ear +
            right_ear
        ) / 2.0

        # Eye closed
        if ear < self.EAR_THRESHOLD:

            self.blinking = True

            return False

        # Eye opened
        if self.blinking:

            self.blinking = False

            now = time.time()

            self.blinks.append(now)

            while (
                self.blinks
                and now - self.blinks[0]
                > self.DOUBLE_BLINK_WINDOW
            ):
                self.blinks.popleft()

            if len(self.blinks) >= 2:

                if (
                    now - self.last_click_time
                    > self.CLICK_COOLDOWN
                ):

                    self.last_click_time = now
                    self.blinks.clear()

                    return True

                self.blinks.clear()

        return False


# ═════════════════════════════════════════════════════════════════════════════
# VOICE CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════

class VoiceController:

    def __init__(self):

        self.running = False

        self.lock = Lock()

        self.command = None

        self.last_command = ""

        self.recognizer = sr.Recognizer()

        self.recognizer.energy_threshold = 300

        self.recognizer.dynamic_energy_threshold = True

        self.recognizer.pause_threshold = 0.6

        self.microphone = None

        try:

            self.microphone = sr.Microphone()

            print("[Voice] Microphone detected.")

        except Exception as e:

            print("[Voice] Microphone error:", e)

    def start(self):

        if self.microphone is None:

            print("[Voice] Voice control disabled.")

            return

        self.running = True

        self.thread = Thread(
            target=self._listen,
            daemon=True
        )

        self.thread.start()

        print("[Voice] Voice control started.")

    def _listen(self):

        try:

            with self.microphone as source:

                print("[Voice] Adjusting microphone...")

                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1
                )

                print("[Voice] Listening for commands...")

                while self.running:

                    try:

                        audio = self.recognizer.listen(
                            source,
                            timeout=1,
                            phrase_time_limit=3
                        )

                        text = self.recognizer.recognize_google(
                            audio
                        )

                        text = text.lower().strip()

                        print(
                            f"[Voice] Heard: {text}"
                        )

                        command = self._parse_command(
                            text
                        )

                        if command:

                            with self.lock:

                                self.command = command
                                self.last_command = command

                    except sr.WaitTimeoutError:

                        continue

                    except sr.UnknownValueError:

                        continue

                    except sr.RequestError as e:

                        print(
                            "[Voice] Speech service error:",
                            e
                        )

                        time.sleep(1)

                    except Exception as e:

                        if self.running:

                            print(
                                "[Voice] Error:",
                                e
                            )

                        time.sleep(0.2)

        except Exception as e:

            print(
                "[Voice] Microphone thread stopped:",
                e
            )

    def _parse_command(self, text):

        # ── Click ─────────────────────────────────────────────────────────────

        if (
            text == "click"
            or text == "left click"
            or text == "select"
        ):
            return "click"

        # ── Double click ──────────────────────────────────────────────────────

        if (
            "double click" in text
            or "double-click" in text
        ):
            return "double_click"

        # ── Right click ───────────────────────────────────────────────────────

        if (
            "right click" in text
            or "right-click" in text
        ):
            return "right_click"

        # ── Scroll ────────────────────────────────────────────────────────────

        if (
            "scroll up" in text
            or "scroll upward" in text
        ):
            return "scroll_up"

        if (
            "scroll down" in text
            or "scroll downward" in text
        ):
            return "scroll_down"

        # ── Gaze ──────────────────────────────────────────────────────────────

        if (
            "pause gaze" in text
            or "stop gaze" in text
            or "gaze pause" in text
        ):
            return "pause_gaze"

        if (
            "resume gaze" in text
            or "start gaze" in text
            or "gaze resume" in text
        ):
            return "resume_gaze"

        # ── Browser navigation ────────────────────────────────────────────────

        if (
            "go back" in text
            or text == "back"
        ):
            return "go_back"

        # ── Escape ────────────────────────────────────────────────────────────

        if (
            text == "escape"
            or text == "esc"
            or "press escape" in text
        ):
            return "escape"

        return None

    def get_command(self):

        with self.lock:

            command = self.command

            self.command = None

            return command

    def stop(self):

        self.running = False
# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():

    print("=" * 70)
    print("             AIRSWIPE MULTIMODAL CONTROL SYSTEM")
    print("=" * 70)

    print()
    print("CONTROLS")
    print("----------------------------------------")
    print("G                 = Pause / Resume Gaze")
    print("Q                 = Quit")
    print()
    print("LOOK              = Move Cursor")
    print("DOUBLE BLINK      = Left Click")
    print("INDEX + PINKY     = Left Click")
    print("PEACE             = Right Click")
    print("THREE FINGERS     = Double Click")
    print("OPEN PALM         = Scroll Up")
    print("FOUR FINGERS      = Scroll Down")
    print("FIST              = Pause / Resume AirSwipe")
    print()
    print("VOICE COMMANDS")
    print("----------------------------------------")
    print('"click"')
    print('"double click"')
    print('"right click"')
    print('"scroll up"')
    print('"scroll down"')
    print('"pause gaze"')
    print('"resume gaze"')
    print('"go back"')
    print('"escape"')
    print("----------------------------------------")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # FOVEA
    # ═════════════════════════════════════════════════════════════════════════

    print("[1/5] Starting Fovea gaze engine...")

    gaze = FoveaBridge()

    gaze.start()

    # ═════════════════════════════════════════════════════════════════════════
    # VOICE
    # ═════════════════════════════════════════════════════════════════════════

    print("[2/5] Starting voice controller...")

    voice = VoiceController()

    voice.start()

    # ═════════════════════════════════════════════════════════════════════════
    # AIRSWIPE STATE
    # ═════════════════════════════════════════════════════════════════════════

    state = make_state()

    history = deque(
        maxlen=HISTORY_LEN
    )

    prev_frame_t = time.time()

    p = np.zeros(
        (21, 3),
        dtype=np.float32
    )

    # ═════════════════════════════════════════════════════════════════════════
    # GAZE
    # ═════════════════════════════════════════════════════════════════════════

    gaze_enabled = True

    smooth_x = None
    smooth_y = None

    SMOOTHING = 0.25

    # ═════════════════════════════════════════════════════════════════════════
    # BLINK
    # ═════════════════════════════════════════════════════════════════════════

    blink_detector = DoubleBlinkDetector()

    print("[3/5] Loading hand detector...")

    # ═════════════════════════════════════════════════════════════════════════
    # FACE MODEL
    # ═════════════════════════════════════════════════════════════════════════

    print("[4/5] Loading face detector...")

    face_model_path = (
        r"C:\Users\myacc\AppData\Local\fovea\fovea"
        r"\Cache\models\face_landmarker.task"
    )

    try:

        face_options = mp_vision.FaceLandmarkerOptions(

            base_options=mp.tasks.BaseOptions(
                model_asset_path=face_model_path
            ),

            running_mode=mp_vision.RunningMode.VIDEO,

            num_faces=1,

            min_face_detection_confidence=0.5,

            min_face_presence_confidence=0.5,

            min_tracking_confidence=0.5,

            output_face_blendshapes=False,

            output_facial_transformation_matrixes=False
        )

        face_detector = (
            mp_vision.FaceLandmarker
            .create_from_options(
                face_options
            )
        )

    except Exception as e:

        print()
        print("[WARNING] Face detector could not start.")
        print(e)
        print("Double-blink clicking is disabled.")
        print()

        face_detector = None

    # ═════════════════════════════════════════════════════════════════════════
    # HAND DETECTOR
    # ═════════════════════════════════════════════════════════════════════════

    with mp_vision.HandLandmarker.create_from_options(
        MEDIAPIPE_OPTS
    ) as detector:

        print("[5/5] AirSwipe ready.")
        print()
        print("Look at the screen to move the cursor.")
        print("Speak a command such as: CLICK")
        print("Press G to pause/resume gaze.")
        print("Press Q to quit.")
        print("-" * 70)

        try:

            while True:

                # ═════════════════════════════════════════════════════════════
                # GET FOVEA FRAME
                # ═════════════════════════════════════════════════════════════

                frame = gaze.get_frame()

                if frame is None:

                    time.sleep(0.01)

                    continue

                frame = frame.copy()

                ih, iw = frame.shape[:2]

                frame_ts_ms = int(
                    time.time() * 1000
                )

                # ═════════════════════════════════════════════════════════════
                # RGB
                # ═════════════════════════════════════════════════════════════

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                # ═════════════════════════════════════════════════════════════
                # HAND TRACKING
                # ═════════════════════════════════════════════════════════════

                result = detector.detect_for_video(
                    mp_img,
                    frame_ts_ms
                )

                raw_gesture = "none"

                if result.hand_landmarks:

                    lm_list = result.hand_landmarks[0]

                    p = to_px(
                        lm_list,
                        iw,
                        ih
                    )

                    draw_skeleton(
                        frame,
                        p
                    )

                    raw_gesture = classify(p)

                history.append(
                    raw_gesture
                )

                if history:

                    gesture = max(
                        set(history),
                        key=history.count
                    )

                else:

                    gesture = "none"

                # ═════════════════════════════════════════════════════════════
                # HAND ACTION
                # ═════════════════════════════════════════════════════════════

                if state["click_flash"] > 0:

                    state["click_flash"] -= 1

                state = execute_action(
                    gesture,
                    p,
                    iw,
                    ih,
                    state
                )

                # ═════════════════════════════════════════════════════════════
                # GAZE
                # ═════════════════════════════════════════════════════════════

                (
                    gaze_x,
                    gaze_y,
                    gaze_confidence,
                    gaze_tracking
                ) = gaze.get_gaze()

                # ═════════════════════════════════════════════════════════════
                # GAZE → CURSOR
                # ═════════════════════════════════════════════════════════════

                if (
                    gaze_enabled
                    and gaze_tracking
                    and gaze_confidence > 0.50
                    
                ):

                    screen_w, screen_h = (
                        pyautogui.size()
                    )

                    target_x = (
                        gaze_x *
                        screen_w
                    )

                    target_y = (
                        gaze_y *
                        screen_h
                    )

                    if smooth_x is None:

                        smooth_x = target_x
                        smooth_y = target_y

                    else:

                        smooth_x = (
                            smooth_x *
                            (1 - SMOOTHING)
                            +
                            target_x *
                            SMOOTHING
                        )

                        smooth_y = (
                            smooth_y *
                            (1 - SMOOTHING)
                            +
                            target_y *
                            SMOOTHING
                        )

                    pyautogui.moveTo(
                        int(smooth_x),
                        int(smooth_y),
                        duration=0
                    )

                # ═════════════════════════════════════════════════════════════
                # DOUBLE BLINK
                # ═════════════════════════════════════════════════════════════

                double_blink = False

                if face_detector is not None:

                    face_result = (
                        face_detector.detect_for_video(
                            mp_img,
                            frame_ts_ms
                        )
                    )

                    if face_result.face_landmarks:

                        face_landmarks = (
                            face_result.face_landmarks[0]
                        )

                        double_blink = (
                            blink_detector.update(
                                face_landmarks
                            )
                        )

                if (
                    double_blink
                    and gaze_enabled
                    and not state["paused"]
                ):

                    pyautogui.click()

                    state["click_flash"] = 8

                    print(
                        "[Blink] Double blink -> LEFT CLICK"
                    )

                # ═════════════════════════════════════════════════════════════
                # VOICE COMMAND
                # ═════════════════════════════════════════════════════════════
                command = voice.get_command()

                if command:

                    print(
                        f"[Voice Command] {command}"
                    )

                    # ── Click ────────────────────────────────────────────────

                    if command == "click":

                        if not state["paused"]:

                            print(
                                "[Voice Action] Performing LEFT CLICK"
                            )

                            pyautogui.click(
                                button="left"
                            )

                            state["click_flash"] = 8

                    # ── Double click ─────────────────────────────────────────

                    elif command == "double_click":

                        if not state["paused"]:

                            print(
                                "[Voice Action] Performing DOUBLE CLICK"
                            )

                            pyautogui.doubleClick(
                                interval=0.1
                            )

                            state["click_flash"] = 8

                    # ── Right click ──────────────────────────────────────────

                    elif command == "right_click":

                        if not state["paused"]:

                            print(
                                "[Voice Action] Performing RIGHT CLICK"
                            )

                            pyautogui.click(
                                button="right"
                            )

                            state["click_flash"] = 8

                    # ── Scroll up ────────────────────────────────────────────

                    elif command == "scroll_up ":    
                        print("[Voice Action] SCROLL UP")

                        pyautogui.scroll(12)

                    elif command == "scroll_down":

                        print("[Voice Action] SCROLL DOWN")

                        pyautogui.scroll(-12)

                    # ── Scroll down ──────────────────────────────────────────

                    

                    # ── Pause gaze ───────────────────────────────────────────

                    elif command == "pause_gaze":

                        gaze_enabled = False

                        smooth_x = None
                        smooth_y = None

                        print(
                            "[Voice] GAZE PAUSED"
                        )

                    # ── Resume gaze ─────────────────────────────────────────

                    elif command == "resume_gaze":

                        gaze_enabled = True

                        smooth_x = None
                        smooth_y = None

                        print(
                            "[Voice] GAZE RESUMED"
                        )

                    # ── Go back ──────────────────────────────────────────────

                    elif command == "go_back":

                        pyautogui.hotkey(
                            "alt",
                            "left"
                        )

                    # ── Escape ───────────────────────────────────────────────

                    elif command == "escape":

                        pyautogui.press(
                            "esc"
                        )

                # ═════════════════════════════════════════════════════════════
                # FPS
                # ═════════════════════════════════════════════════════════════

                now = time.time()

                fps = 1.0 / max(
                    now - prev_frame_t,
                    1e-6
                )

                prev_frame_t = now

                # ═════════════════════════════════════════════════════════════
                # HUD
                # ═════════════════════════════════════════════════════════════

                draw_hud(
                    frame,
                    gesture,
                    state["paused"],
                    fps,
                    state["click_flash"]
                )

                # Gaze status

                if gaze_tracking:

                    gaze_text = (
                        f"Gaze: "
                        f"({gaze_x:.2f}, "
                        f"{gaze_y:.2f}) "
                        f"Conf: "
                        f"{gaze_confidence:.2f}"
                    )

                    cv2.putText(
                        frame,
                        gaze_text,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                else:

                    cv2.putText(
                        frame,
                        "Gaze: Waiting...",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                # Multimodal status

                cv2.putText(
                    frame,
                    "MULTIMODAL: GAZE + HAND + VOICE + BLINK",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2
                )

                # Gaze status

                if gaze_enabled:

                    gaze_status = "GAZE: ACTIVE"

                    gaze_color = (
                        0,
                        255,
                        0
                    )

                else:

                    gaze_status = "GAZE: PAUSED"

                    gaze_color = (
                        0,
                        0,
                        255
                    )

                cv2.putText(
                    frame,
                    gaze_status,
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    gaze_color,
                    2
                )

                # Voice status

                cv2.putText(
                    frame,
                    "VOICE: LISTENING",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )

                # Blink feedback

                if double_blink:

                    cv2.putText(
                        frame,
                        "DOUBLE BLINK -> LEFT CLICK",
                        (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                # ═════════════════════════════════════════════════════════════
                # DISPLAY
                # ═════════════════════════════════════════════════════════════

                cv2.imshow(
                    "AirSwipe - Multimodal Control",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                # Q → quit

                if key == ord("q"):

                    break

                # G → pause/resume gaze

                elif key == ord("g"):

                    gaze_enabled = not gaze_enabled

                    smooth_x = None
                    smooth_y = None

                    if gaze_enabled:

                        print(
                            "[Gaze] RESUMED"
                        )

                    else:

                        print(
                            "[Gaze] PAUSED"
                        )

        except KeyboardInterrupt:

            print()
            print("Stopping AirSwipe...")

        finally:

            print()
            print("Stopping AirSwipe...")

            voice.stop()

            gaze.stop()

            if face_detector is not None:

                try:

                    face_detector.close()

                except Exception:

                    pass

            cv2.destroyAllWindows()

            print("AirSwipe stopped.")


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    main()        