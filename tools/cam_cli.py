#!/usr/bin/env python3
"""Interactive camera editor for the live Agora stream.

Run on staging (or SSH in). Arrow keys orbit the external camera around
its look-at target; WASD pans the target on the ground plane, Q/E raises
or lowers it, [/] tweaks FOV, +/- zoom in/out. Each keystroke rewrites
telearms/cam_override.yaml, which TelearmsTeleop picks up on the next
frame — no restart, no Agora interruption.

Camera state is kept in spherical coords around the target (azimuth,
elevation, distance) so orbiting feels natural. pos for the YAML is
derived on write; target/fovy are stored directly.
"""
import curses
import math
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pyyaml required — activate playground-venv before running.", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parents[1]
OVERRIDE = REPO / "telearms" / "cam_override.yaml"

# Matches the pose baked into world.xml. Used when no override file
# exists and as the target of the reset key.
DEFAULTS = {
    "target": [0.25, -0.05, 1.40],
    "azimuth": 180.0,    # degrees; 180 = camera on -X side of target
    "elevation": 43.0,   # degrees above the horizontal plane
    "distance": 0.37,    # meters
    "fovy": 90.0,
}

# Per-keystroke step sizes. Tuned so a few taps makes a visible difference
# without overshooting: azim/elev 5°/3°, target 5 cm, distance 10 cm.
STEP_AZIM = 5.0
STEP_ELEV = 3.0
STEP_DIST = 0.10
STEP_TARGET = 0.05
STEP_FOVY = 2.0


def compute_pos(state):
    """Derive world-space cam position from spherical state around target."""
    tx, ty, tz = state["target"]
    r = state["distance"]
    az = math.radians(state["azimuth"])
    el = math.radians(state["elevation"])
    px = tx + r * math.cos(el) * math.cos(az)
    py = ty + r * math.cos(el) * math.sin(az)
    pz = tz + r * math.sin(el)
    return [px, py, pz]


def load_state():
    """Start from override file if present, else DEFAULTS."""
    state = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    if not OVERRIDE.exists():
        return state
    try:
        data = yaml.safe_load(OVERRIDE.read_text()) or {}
    except Exception:
        return state
    if "target" in data and len(data["target"]) == 3:
        state["target"] = [float(x) for x in data["target"]]
    if "fovy" in data:
        state["fovy"] = float(data["fovy"])
    if "pos" in data and len(data["pos"]) == 3:
        px, py, pz = (float(x) for x in data["pos"])
        tx, ty, tz = state["target"]
        dx, dy, dz = px - tx, py - ty, pz - tz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > 1e-6:
            state["distance"] = dist
            state["elevation"] = math.degrees(math.asin(max(-1.0, min(1.0, dz / dist))))
            state["azimuth"] = math.degrees(math.atan2(dy, dx))
    return state


def write_yaml(state):
    pos = compute_pos(state)
    body = (
        f"pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
        f"target: [{state['target'][0]:.3f}, {state['target'][1]:.3f}, {state['target'][2]:.3f}]\n"
        f"fovy: {state['fovy']:.1f}\n"
    )
    OVERRIDE.write_text(body)


HELP_LINES = [
    "↑ ↓           elevation (pitch)",
    "← →           azimuth (orbit)",
    "+ / -         zoom (distance)",
    "w a s d       pan target on X/Y",
    "q / e         raise / lower target (Z)",
    "[ / ]         fovy -/+",
    "r             reset to defaults",
    "c             clear override and quit",
    "Esc / x       quit (keep current override)",
]


def ui(stdscr):
    state = load_state()
    write_yaml(state)
    status = f"wrote {OVERRIDE}"

    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    while True:
        stdscr.erase()
        pos = compute_pos(state)
        stdscr.addstr(0, 0, "Live camera editor — Agora external cam", curses.A_BOLD)
        stdscr.addstr(1, 0, f"writing: {OVERRIDE}", curses.A_DIM)

        stdscr.addstr(3, 0, "state", curses.A_UNDERLINE)
        stdscr.addstr(4, 2, f"target   [{state['target'][0]:+.2f}, {state['target'][1]:+.2f}, {state['target'][2]:+.2f}]")
        stdscr.addstr(5, 2, f"azimuth  {state['azimuth']:+7.1f} deg")
        stdscr.addstr(6, 2, f"elev     {state['elevation']:+7.1f} deg")
        stdscr.addstr(7, 2, f"distance {state['distance']:7.2f} m")
        stdscr.addstr(8, 2, f"fovy     {state['fovy']:7.1f} deg")
        stdscr.addstr(9, 2, f"=> pos   [{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}]")

        stdscr.addstr(11, 0, "controls", curses.A_UNDERLINE)
        for i, line in enumerate(HELP_LINES):
            stdscr.addstr(12 + i, 2, line)

        stdscr.addstr(12 + len(HELP_LINES) + 1, 0, f"status: {status}")
        stdscr.refresh()

        key = stdscr.getch()
        changed = True

        if key == curses.KEY_LEFT:
            state["azimuth"] = (state["azimuth"] - STEP_AZIM) % 360.0
        elif key == curses.KEY_RIGHT:
            state["azimuth"] = (state["azimuth"] + STEP_AZIM) % 360.0
        elif key == curses.KEY_UP:
            state["elevation"] = min(85.0, state["elevation"] + STEP_ELEV)
        elif key == curses.KEY_DOWN:
            state["elevation"] = max(-85.0, state["elevation"] - STEP_ELEV)
        elif key in (ord('+'), ord('=')):
            state["distance"] = max(0.30, state["distance"] - STEP_DIST)
        elif key in (ord('-'), ord('_')):
            state["distance"] += STEP_DIST
        elif key == ord('w'):
            state["target"][0] += STEP_TARGET
        elif key == ord('s'):
            state["target"][0] -= STEP_TARGET
        elif key == ord('a'):
            state["target"][1] += STEP_TARGET
        elif key == ord('d'):
            state["target"][1] -= STEP_TARGET
        elif key == ord('q'):
            state["target"][2] += STEP_TARGET
        elif key == ord('e'):
            state["target"][2] -= STEP_TARGET
        elif key == ord('['):
            state["fovy"] = max(20.0, state["fovy"] - STEP_FOVY)
        elif key == ord(']'):
            state["fovy"] = min(90.0, state["fovy"] + STEP_FOVY)
        elif key == ord('r'):
            state = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
        elif key == ord('c'):
            try:
                OVERRIDE.unlink()
                status = f"cleared {OVERRIDE} and exiting"
            except FileNotFoundError:
                status = "no override file to clear"
            stdscr.addstr(12 + len(HELP_LINES) + 1, 0, f"status: {status}")
            stdscr.refresh()
            return
        elif key in (27, ord('x')):  # 27 = ESC
            return
        else:
            changed = False

        if changed:
            write_yaml(state)
            status = "applied"


def main():
    curses.wrapper(lambda stdscr: ui(stdscr))


if __name__ == "__main__":
    main()
