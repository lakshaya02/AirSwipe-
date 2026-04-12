"""
gesture_action.py : Action Execution Engine
======================================================

RESPONSIBILITY: Translating confirmed gesture names into actual system
                actions (mouse movement, clicks, scrolling, hotkeys).
                Manages all cooldowns and action state.

To ADD a new action:
  1. Add a new `if gesture == 'your_gesture':` block in execute_action()
  2. Decide if it's edge-triggered (fires once) or continuous (fires every frame)
  3. Use the appropriate cooldown pattern shown below

To CHANGE cooldown values:
  → Edit the constants in config.py (CLICK_COOLDOWN, SCROLL_COOLDOWN etc.)
  → Do NOT hardcode numbers here — always reference config.py
"""

import time
import pyautogui
from gesture_classifier import frame_to_screen
from config import (
    SCREEN_W, SCREEN_H,
    SCROLL_COOLDOWN, SCROLL_SPEED,
    CLICK_COOLDOWN, HOTKEY_COOLDOWN,
    MOUSE_SMOOTH, MARGIN,
)


# ─────────────────────────────────────────────
# INITIAL STATE FACTORY
# ─────────────────────────────────────────────

def make_state():
    """
    Create and return the initial action state dictionary.

    This dict is passed into execute_action() every frame and returned
    with any updates applied. It is the single source of truth for all
    timing, position, and mode data across frames.

    State keys:
        mouse_pos      : (x, y) last known smoothed cursor position
        last_click_t   : timestamp of most recent click action
        last_hotkey_t  : timestamp of most recent hotkey action
        last_scroll_t  : timestamp of most recent scroll tick
        last_pause_t   : timestamp of most recent pause toggle
        prev_gesture   : gesture string from the previous frame
        paused         : bool — whether all actions are currently paused
        click_flash    : int — frames remaining for click flash ring display
    """
    return {
        'mouse_pos':     (SCREEN_W // 2, SCREEN_H // 2),
        'last_click_t':  0.0,
        'last_hotkey_t': 0.0,
        'last_scroll_t': 0.0,
        'last_pause_t':  0.0,
        'prev_gesture':  'none',
        'paused':        False,
        'click_flash':   0,
    }


# ─────────────────────────────────────────────
# ACTION EXECUTION
# ─────────────────────────────────────────────

def execute_action(gesture, p, img_w, img_h, state):
    """
    Translate a confirmed gesture name into a system action.

    Two action patterns are used:

    EDGE-TRIGGERED (fires ONCE per gesture, on first frame only):
        Used for: clicks, hotkeys, pause toggle
        Pattern:  `if is_new and (now - state['last_X_t']) >= COOLDOWN:`

    CONTINUOUS (fires EVERY frame the gesture is held):
        Used for: mouse move, scroll
        Pattern:  `if (now - state['last_X_t']) >= COOLDOWN:`

    Args:
        gesture  : str — confirmed gesture name from gesture_classifier.py
        p        : np.ndarray (21,3) — current hand landmarks in pixels
        img_w    : int — camera frame width
        img_h    : int — camera frame height
        state    : dict — mutable action state (returned with updates)

    Returns:
        Updated state dict
    """
    now       = time.time()
    prev      = state['prev_gesture']
    paused    = state['paused']
    mouse_pos = state['mouse_pos']
    is_new    = (gesture != prev)   # True only on the FIRST frame of a new gesture

    # ── PAUSE ─────────────────────────────────────────────────────────────────
    # Edge-triggered toggle. Has its own cooldown to prevent rapid toggling
    # when the fist gesture is held continuously.
    if gesture == 'pause':
        if is_new and (now - state['last_pause_t']) >= 0.5:
            state['paused']       = not paused
            state['last_pause_t'] = now
        state['prev_gesture'] = gesture
        return state

    # ── GLOBAL PAUSE GATE ─────────────────────────────────────────────────────
    # All gestures below this line are blocked while paused.
    # Only the pause gesture above can toggle out of paused mode.
    if paused:
        state['prev_gesture'] = gesture
        return state

    # ── MOUSE MOVE ────────────────────────────────────────────────────────────
    # Continuous — runs every frame while index finger is up.
    # Exponential smoothing blends the current position toward the target
    # to eliminate jitter. Lower MOUSE_SMOOTH = faster/snappier cursor.
    if gesture == 'mouse_move':
        tx, ty = frame_to_screen(
            int(p[8, 0]), int(p[8, 1]),   # Index fingertip pixel position
            img_w, img_h,
            SCREEN_W, SCREEN_H, MARGIN
        )
        # Exponential smoothing: new_pos = old_pos + (target - old_pos) * factor
        sx = int(mouse_pos[0] + (tx - mouse_pos[0]) * (1 - MOUSE_SMOOTH))
        sy = int(mouse_pos[1] + (ty - mouse_pos[1]) * (1 - MOUSE_SMOOTH))
        pyautogui.moveTo(sx, sy, duration=0)
        state['mouse_pos']    = (sx, sy)
        state['prev_gesture'] = gesture
        return state

    # ── SCROLL UP ─────────────────────────────────────────────────────────────
    # Continuous — fires a scroll tick every SCROLL_COOLDOWN seconds
    # while the open palm gesture is held.
    if gesture == 'scroll_up':
        if (now - state['last_scroll_t']) >= SCROLL_COOLDOWN:
            pyautogui.scroll(SCROLL_SPEED)
            state['last_scroll_t'] = now
        state['prev_gesture'] = gesture
        return state

    # ── SCROLL DOWN ───────────────────────────────────────────────────────────
    # Continuous — same as scroll up but negative scroll direction.
    if gesture == 'scroll_down':
        if (now - state['last_scroll_t']) >= SCROLL_COOLDOWN:
            pyautogui.scroll(-SCROLL_SPEED)
            state['last_scroll_t'] = now
        state['prev_gesture'] = gesture
        return state

    # ── LEFT CLICK ────────────────────────────────────────────────────────────
    # Edge-triggered — fires once when gesture first appears.
    # click_flash starts a 5-frame visual ring indicator in the HUD.
    if gesture == 'left_click':
        if is_new and (now - state['last_click_t']) >= CLICK_COOLDOWN:
            pyautogui.click()
            state['last_click_t'] = now
            state['click_flash']  = 5   # Triggers ring flash in display.py
        state['prev_gesture'] = gesture
        return state

    # ── RIGHT CLICK ───────────────────────────────────────────────────────────
    # Edge-triggered — peace sign gesture.
    if gesture == 'right_click':
        if is_new and (now - state['last_click_t']) >= CLICK_COOLDOWN:
            pyautogui.rightClick()
            state['last_click_t'] = now
            state['click_flash']  = 5
        state['prev_gesture'] = gesture
        return state

    # ── DOUBLE CLICK ──────────────────────────────────────────────────────────
    # Edge-triggered — three fingers up gesture.
    if gesture == 'double_click':
        if is_new and (now - state['last_click_t']) >= CLICK_COOLDOWN:
            pyautogui.doubleClick()
            state['last_click_t'] = now
            state['click_flash']  = 5
        state['prev_gesture'] = gesture
        return state

    # ── HOTKEYS ───────────────────────────────────────────────────────────────
    # Edge-triggered — all hotkeys share a single cooldown timer.
    # Add new hotkeys by inserting into this dict.
    hotkeys = {
        'go_back': lambda: pyautogui.hotkey('alt', 'left'),  # Browser back
        'escape':  lambda: pyautogui.press('escape'),         # ESC key
    }

    if gesture in hotkeys:
        if is_new and (now - state['last_hotkey_t']) >= HOTKEY_COOLDOWN:
            hotkeys[gesture]()
            state['last_hotkey_t'] = now
        state['prev_gesture'] = gesture
        return state

    state['prev_gesture'] = gesture
    return state
