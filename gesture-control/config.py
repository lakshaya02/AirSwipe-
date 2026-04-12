"""
config.py : Configuration & Constants
=================================================
RESPONSIBILITY: All settings, thresholds, model setup, and constants.
To tune the system behaviour,  edit THIS file.
"""

import os
import urllib.request
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────
# SCREEN RESOLUTION
# Auto-detected from the OS at startup.
# Used by gesture_action.py to map camera coords → screen coords.
# ─────────────────────────────────────────────
pyautogui.FAILSAFE = True   # Move mouse to top-left corner to emergency stop
pyautogui.PAUSE    = 0      # No built-in delay — timing is handled manually

SCREEN_W, SCREEN_H = pyautogui.size()

# ─────────────────────────────────────────────
# ACTION TIMING THRESHOLDS
# Adjust these to change how fast or slow actions trigger.
# ─────────────────────────────────────────────
SCROLL_COOLDOWN = 0.03    # Seconds between each scroll tick (lower = faster scroll)
SCROLL_SPEED    = 6       # Scroll units fired per tick
CLICK_COOLDOWN  = 0.50    # Minimum seconds between consecutive clicks
HOTKEY_COOLDOWN = 0.50    # Minimum seconds between consecutive hotkey presses

# ─────────────────────────────────────────────
# MOUSE MOVEMENT
# ─────────────────────────────────────────────
MOUSE_SMOOTH    = 0.22    # Exponential smoothing: 0 = instant snap, 1 = never moves
MARGIN          = 0.12    # Edge margin (fraction) to ignore at camera borders
                          # Increase if cursor can't reach screen edges

# ─────────────────────────────────────────────
# GESTURE DETECTION STABILITY
# ─────────────────────────────────────────────
HISTORY_LEN     = 3       # Frames used for majority-vote smoothing
                          # Higher = more stable but slightly more lag

# ─────────────────────────────────────────────
# WEBCAM RESOLUTION
# ─────────────────────────────────────────────
CAM_W  = 640
CAM_H  = 480
CAM_FPS = 30

# ─────────────────────────────────────────────
# MEDIAPIPE MODEL — AUTO DOWNLOAD
# Downloads the hand landmark model on first run (~6 MB).
# Saved locally so future runs are instant.
# ─────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand model (~6 MB, first run only) ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model saved as hand_landmarker.task")

# ─────────────────────────────────────────────
# MEDIAPIPE MODEL OPTIONS
# Configures detection confidence, tracking mode, and hand count.
# VIDEO mode requires monotonically increasing timestamps per frame.
# ─────────────────────────────────────────────
MEDIAPIPE_OPTS = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,                          # Track only one hand at a time
    min_hand_detection_confidence=0.8,    # Confidence needed to first detect a hand
    min_hand_presence_confidence=0.8,     # Confidence needed to keep tracking it
    min_tracking_confidence=0.6,          # Confidence needed to track between frames
)

# ─────────────────────────────────────────────
# HUD DISPLAY METADATA
# Maps gesture name → (display label, BGR colour) for the HUD overlay.
# Add a new entry here whenever a new gesture is added to gesture_classifier.py.
# ─────────────────────────────────────────────
GESTURE_META = {
    'pause':        ('PAUSED',        (60,  60, 220)),
    'scroll_up':    ('Scroll Up',     (50, 220,  50)),
    'scroll_down':  ('Scroll Down',   (50, 220,  50)),
    'mouse_move':   ('Mouse Move',    (50, 220,  50)),
    'left_click':   ('Left Click',    (50, 200, 255)),
    'right_click':  ('Right Click',   (50, 200, 255)),
    'double_click': ('Double Click',  (50, 200, 255)),
    'go_back':      ('Go Back',       (180, 80, 255)),
    'escape':       ('Escape',        (180, 80, 255)),
    'none':         ('No Gesture',    (120, 120, 120)),
}

# ─────────────────────────────────────────────
# HUD HINT TEXT
# Maps gesture name → human-readable hand shape shown next to the label.
# Add a new entry here whenever a new gesture is added to gesture_classifier.py.
# ─────────────────────────────────────────────
GESTURE_HINT = {
    'pause':        'Closed fist',
    'scroll_up':    'Open palm',
    'scroll_down':  '4 fingers',
    'mouse_move':   'Index finger',
    'left_click':   'Middle finger',
    'right_click':  'Peace sign',
    'double_click': '3 fingers',
    'go_back':      'Pinky only',
    'escape':       'Shaka',
    'none':         '---',
}

# ─────────────────────────────────────────────
# HAND SKELETON CONNECTIONS
# Pairs of landmark indices forming the bones of the hand.
# Used by display.py to draw lines between joints.
# ─────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index finger
    (5,9),(9,10),(10,11),(11,12),   # Middle finger
    (9,13),(13,14),(14,15),(15,16), # Ring finger
    (13,17),(17,18),(18,19),(19,20),# Pinky finger
    (0,17),(5,17),                  # Palm base
]
