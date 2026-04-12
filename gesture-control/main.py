import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ── Import all modules ────────────────────────────────────────────────────────
from config           import MEDIAPIPE_OPTS, CAM_W, CAM_H, CAM_FPS, HISTORY_LEN
from gesture_classifier import to_px, classify
from gesture_action   import make_state, execute_action
from display          import draw_skeleton, draw_hud

from mediapipe.tasks.python import vision as mp_vision


def main():
    """
    Main loop — runs once per frame until the user presses Q.

    Frame pipeline per iteration:
        1. Read frame from webcam
        2. Flip (mirror for natural feel)
        3. Convert BGR → RGB for MediaPipe
        4. Run hand landmark detection
        5. Convert landmarks to pixels
        6. Classify raw gesture from landmarks
        7. Majority-vote over last HISTORY_LEN frames (smoothing)
        8. Execute confirmed gesture action
        9. Draw skeleton + HUD
       10. Show window / check for Q key
    """

    # ── Webcam setup ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check camera permissions.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)

    print("Gesture Control System started. Press Q in camera window to quit.")

    # ── State initialisation ──────────────────────────────────────────────────
    state        = make_state()                              # From gesture_action.py
    history      = deque(maxlen=HISTORY_LEN)                # Rolling gesture history
    prev_frame_t = time.time()                              # FPS timer
    frame_ts_ms  = 0                                        # MediaPipe VIDEO timestamp
    p            = np.zeros((21, 3), dtype=np.float32)     # Default landmarks (no hand)

    # ── Main loop ─────────────────────────────────────────────────────────────
    with mp_vision.HandLandmarker.create_from_options(MEDIAPIPE_OPTS) as detector:
        while True:

            # Step 1 — Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Step 2 — Mirror frame (so left hand = left on screen)
            frame  = cv2.flip(frame, 1)
            ih, iw = frame.shape[:2]

            # Step 3 — Convert colour space for MediaPipe
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Step 4 — Run hand detection
            # Timestamp must strictly increase — VIDEO mode requirement
            now_ms      = int(time.time() * 1000)
            frame_ts_ms = max(frame_ts_ms + 1, now_ms)
            result      = detector.detect_for_video(mp_img, frame_ts_ms)

            # Step 5+6 — Extract landmarks and classify gesture
            raw_gesture = 'none'
            if result.hand_landmarks:
                lm_list     = result.hand_landmarks[0]
                p           = to_px(lm_list, iw, ih)       # gesture_classifier.py
                draw_skeleton(frame, p)                     # display.py
                raw_gesture = classify(p)                   # gesture_classifier.py

            # Step 7 — Majority vote: pick the most common gesture in last N frames
            # This smooths out single-frame misdetections and gesture flickers
            history.append(raw_gesture)
            gesture = max(set(history), key=history.count)

            # Decrement click flash ring timer each frame
            if state['click_flash'] > 0:
                state['click_flash'] -= 1

            # Step 8 — Execute action for confirmed gesture
            state = execute_action(gesture, p, iw, ih, state)  # gesture_action.py

            # Step 9 — Compute FPS from wall-clock time between frames
            now          = time.time()
            fps          = 1.0 / max(now - prev_frame_t, 1e-6)
            prev_frame_t = now

            # Step 10 — Draw HUD and show window
            draw_hud(frame, gesture, state['paused'], fps, state['click_flash'])  # display.py
            cv2.imshow('Hand Gesture Control', frame)

            # Quit on Q key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == '__main__':
    main()
