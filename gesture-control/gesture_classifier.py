"""
gesture_classifier.py : Hand Detection & Gesture Classification
==========================================================================
RESPONSIBILITY: Converting raw landmarks into finger states, then into
                named gestures. All gesture logic lives here.

To ADD a new gesture:
  1. Add a new `if` block inside classify() with the finger pattern
  2. Add the gesture name to GESTURE_META and GESTURE_HINT in config.py
  3. Add an action block for it in gesture_action.py

To CHANGE a gesture mapping:
  → edit this file.
"""

import numpy as np


# ─────────────────────────────────────────────
# LANDMARK CONVERSION
# ─────────────────────────────────────────────

def to_px(lm_list, w, h):
    """
    Convert normalized MediaPipe landmarks (0.0–1.0) to pixel coordinates.

    MediaPipe returns all coordinates normalized between 0 and 1.
    This function scales them to actual pixel positions on the camera frame.

    Args:
        lm_list : list of MediaPipe landmark objects (21 points)
        w       : frame width in pixels
        h       : frame height in pixels

    Returns:
        np.ndarray of shape (21, 3) → columns are [x_px, y_px, z_norm]
    """
    return np.array([[lm.x * w, lm.y * h, lm.z] for lm in lm_list],
                    dtype=np.float32)


# ─────────────────────────────────────────────
# FINGER STATE DETECTION
# ─────────────────────────────────────────────

def finger_states(p):
    """
    Determine which fingers are extended (1) or curled (0).

    Method:
      - Fingers (index–pinky): fingertip y < PIP knuckle y → extended
        (y=0 is top of frame, so lower y = higher on screen = finger up)
      - Thumb: uses x-axis comparison to handle left/right hand mirroring.
        On a right hand (mirrored), tip x < joint x means extended.

    Landmark indices used:
      Thumb:  tip=4,  pip=3
      Index:  tip=8,  pip=6
      Middle: tip=12, pip=10
      Ring:   tip=16, pip=14
      Pinky:  tip=20, pip=18

    Args:
        p : np.ndarray (21, 3) pixel landmarks from to_px()

    Returns:
        tuple (thumb, index, middle, ring, pinky) — each 0 or 1
    """
    right_hand = p[0, 0] < p[9, 0]   # Wrist x < middle MCP x → right hand
    thumb  = int(p[4,0] < p[3,0]) if right_hand else int(p[4,0] > p[3,0])
    index  = int(p[8,1]  < p[6,1])
    middle = int(p[12,1] < p[10,1])
    ring   = int(p[16,1] < p[14,1])
    pinky  = int(p[20,1] < p[18,1])
    return thumb, index, middle, ring, pinky


# ─────────────────────────────────────────────
# GESTURE CLASSIFIER
# ─────────────────────────────────────────────

def classify(p):
    """
    Map current hand landmark positions to a gesture string.

    Checks are ordered by priority — more specific patterns come first
    to avoid ambiguous matches (e.g. fist must be before pinky-only).

    Finger state key:
        t  = thumb   (1 = extended)
        i  = index   (1 = extended)
        m  = middle  (1 = extended)
        r  = ring    (1 = extended)
        pk = pinky   (1 = extended)

    Args:
        p : np.ndarray (21, 3) pixel landmarks from to_px()

    Returns:
        str — gesture name matching a key in GESTURE_META (config.py)

    ── GESTURE TABLE ──────────────────────────────────────────────────
     Priority │ t  i  m  r  pk │ Gesture
    ──────────┼────────────────┼──────────────────────────────────────
         1    │ 0  0  0  0  0  │ pause        (closed fist)
         2    │ 1  0  0  0  1  │ escape       (shaka: thumb + pinky)
         3    │ 0  0  0  0  1  │ go_back      (pinky only)
         4    │ 1  1  1  1  1  │ scroll_up    (open palm)
         5    │ 0  1  1  1  1  │ scroll_down  (4 fingers, no thumb)
         6    │ *  1  1  1  0  │ double_click (index+middle+ring)
         7    │ *  1  1  0  0  │ right_click  (peace sign)
         8    │ 0  1  0  0  1  │ left_click   (index+pinky — horn)
         9    │ *  1  0  0  0  │ mouse_move   (index only)
    ──────────────────────────────────────────────────────────────────
    """
    t, i, m, r, pk = finger_states(p)

    # 1. Closed fist — ALL five fingers including thumb must be down
    if t == 0 and i == 0 and m == 0 and r == 0 and pk == 0:
        return 'pause'

    # 2. Shaka (thumb + pinky up, all others down) → Escape
    if t == 1 and i == 0 and m == 0 and r == 0 and pk == 1:
        return 'escape'

    # 3. Pinky only (thumb down) → Go Back
    if t == 0 and i == 0 and m == 0 and r == 0 and pk == 1:
        return 'go_back'

    # 4. Open palm (all 5 up) → Scroll Up
    if t == 1 and i == 1 and m == 1 and r == 1 and pk == 1:
        return 'scroll_up'

    # 5. Four fingers no thumb → Scroll Down
    if t == 0 and i == 1 and m == 1 and r == 1 and pk == 1:
        return 'scroll_down'

    # 6. Three fingers (index + middle + ring) → Double Click
    if i == 1 and m == 1 and r == 1 and pk == 0:
        return 'double_click'

    # 7. Peace sign (index + middle) → Right Click
    #    Must come BEFORE index-only to avoid mis-classification
    if i == 1 and m == 1 and r == 0 and pk == 0:
        return 'right_click'

    # 8. Index + pinky (horn/rock gesture) → Left Click
    if i == 1 and m == 0 and r == 0 and pk == 1:
        return 'left_click'

    # 9. Index finger only → Mouse Move
    if i == 1 and m == 0 and r == 0 and pk == 0:
        return 'mouse_move'

    return 'none'


# ─────────────────────────────────────────────
# COORDINATE MAPPING
# ─────────────────────────────────────────────

def frame_to_screen(px, py, img_w, img_h, screen_w, screen_h, margin):
    """
    Map a pixel position in the camera frame to a screen coordinate.

    Applies a margin crop so the cursor can reach screen edges without
    the hand needing to go all the way to the camera border.

    Formula:
        normalized = (raw - margin) / (1 - 2*margin)   → clipped to [0, 1]
        screen_pos = normalized * screen_dimension

    Args:
        px, py          : pixel position in camera frame
        img_w, img_h    : camera frame dimensions
        screen_w, screen_h : full screen dimensions
        margin          : fraction of frame edge to ignore (e.g. 0.12 = 12%)

    Returns:
        (x, y) tuple of integer screen coordinates
    """
    rx = np.clip((px / img_w - margin) / (1 - 2 * margin), 0.0, 1.0)
    ry = np.clip((py / img_h - margin) / (1 - 2 * margin), 0.0, 1.0)
    return int(rx * screen_w), int(ry * screen_h)
