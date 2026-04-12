"""
display.py : Camera Display & HUD Overlay
=====================================================OWNER: Team Member 4
RESPONSIBILITY: Everything visible in the camera window.
                Drawing the hand skeleton, HUD panel, labels,
                FPS counter, click flash, and paused indicator.

To CHANGE how things look:
  → Edit this file.

To ADD a new gesture label or colour:
  → Edit GESTURE_META and GESTURE_HINT in config.pya
  → This file reads from those dicts automatically.
"""

import cv2
from config import CONNECTIONS, GESTURE_META, GESTURE_HINT


# ─────────────────────────────────────────────
# HAND SKELETON
# ─────────────────────────────────────────────

def draw_skeleton(frame, p):
    """
    Draw the hand skeleton (bones + joints) on the camera frame.

    Bones: green lines drawn between each landmark pair in CONNECTIONS.
    Joints: white filled circles with a dark green border.
    Fingertips (landmarks 4, 8, 12, 16, 20) use larger circles (r=7)
    to make them easy to see; all other joints use r=4.

    Args:
        frame : np.ndarray — BGR camera frame to draw on (modified in-place)
        p     : np.ndarray (21, 3) — pixel landmark positions from to_px()
    """
    # Draw bones — lines connecting landmark pairs
    for a, b in CONNECTIONS:
        cv2.line(
            frame,
            (int(p[a, 0]), int(p[a, 1])),
            (int(p[b, 0]), int(p[b, 1])),
            (30, 200, 70), 2, cv2.LINE_AA        # Green line, anti-aliased
        )

    # Draw joints — circles at each of the 21 landmarks
    for idx, pt in enumerate(p):
        r = 7 if idx in (4, 8, 12, 16, 20) else 4   # Bigger circle for fingertips
        cv2.circle(frame, (int(pt[0]), int(pt[1])), r,
                   (255, 255, 255), -1, cv2.LINE_AA)  # White fill
        cv2.circle(frame, (int(pt[0]), int(pt[1])), r,
                   (20, 130, 40), 1, cv2.LINE_AA)     # Dark green border


# ─────────────────────────────────────────────
# HUD OVERLAY
# ─────────────────────────────────────────────

def draw_hud(frame, gesture, paused, fps, click_flash):
    """
    Draw the full HUD (Heads-Up Display) overlay on the camera frame.

    HUD elements:
      ┌─ Semi-transparent dark top bar ──────────────────────────┐
      │  [Gesture Label]   [Hand Shape Hint]          [FPS]      │
      └───────────────────────────────────────────────────────────┘
      [Click flash ring at centre — shown briefly after a click]
      [Large PAUSED text at centre — shown when tracking paused]
      [Q: quit — bottom left corner]

    Args:
        frame       : np.ndarray — BGR camera frame (modified in-place)
        gesture     : str — current confirmed gesture name
        paused      : bool — whether tracking is currently paused
        fps         : float — current frames per second
        click_flash : int — frames remaining to show click ring (0 = hidden)
    """
    h, w = frame.shape[:2]

    # ── Top panel ─────────────────────────────────────────────────────────────
    # Semi-transparent dark bar behind gesture label for readability.
    # Uses addWeighted to blend a solid rectangle onto the frame.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)   # 55% dark, 45% frame

    # ── Gesture label ─────────────────────────────────────────────────────────
    # Large coloured label (e.g. "Mouse Move") + small grey hint (e.g. "[Index finger]")
    label, color = GESTURE_META.get(gesture, ('---', (120, 120, 120)))
    hint         = GESTURE_HINT.get(gesture, '')

    cv2.putText(frame, label, (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f'[{hint}]', (12 + len(label) * 18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Click flash ring ──────────────────────────────────────────────────────
    # Cyan ring drawn at screen centre for 5 frames after a click fires.
    # click_flash is decremented each frame in main.py.
    if click_flash > 0:
        cv2.circle(frame, (w // 2, h // 2), 28,
                   (50, 200, 255), 3, cv2.LINE_AA)

    # ── Paused indicator ──────────────────────────────────────────────────────
    # Large blue PAUSED text drawn at centre when tracking is paused.
    if paused:
        cv2.putText(frame, 'PAUSED', (w // 2 - 55, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 220), 3, cv2.LINE_AA)

    # ── Footer info ───────────────────────────────────────────────────────────
    # Bottom-left: quit hint. Bottom-right: live FPS counter.
    cv2.putText(frame, 'Q: quit', (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f'{fps:.0f} fps', (w - 80, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
