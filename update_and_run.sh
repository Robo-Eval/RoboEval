#!/usr/bin/env bash
# update_and_run.sh — pull latest, start Xvfb on :100, launch the sim in Telearms mode.
#
# Modeled after RoboEval/update_and_run.sh but isolated from the production
# instance so both can coexist on the same host:
#   - repo:   ~/telearms-roboplayground  (RoboEval lives at ~/RoboEval)
#   - venv:   ~/playground-venv          (RoboEval uses ~/roboeval-venv)
#   - display: :100                      (RoboEval uses :99)
#   - env file: <repo>/.env              (RoboEval uses ~/roboeval.env)
set -euo pipefail

REPO="${REPO:-/home/ubuntu/telearms-roboplayground}"
VENV="${VENV:-/home/ubuntu/playground-venv}"
DISPLAY_NUM="${DISPLAY_NUM:-100}"
ENV_NAME="${ENV_NAME:-Rotate Valve}"
ROBOT="${ROBOT:-Bimanual Panda}"

cd "$REPO"
git pull

source "$VENV/bin/activate"

# Kill any previous playground sim (match on repo path so we don't touch RoboEval)
pkill -f "$REPO/roboeval/data_collection/demo_recorder.py" 2>/dev/null || true

# Clean just OUR Xvfb (:$DISPLAY_NUM), leave the RoboEval one alone
pkill -f "Xvfb :$DISPLAY_NUM" 2>/dev/null || true
rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}"
mkdir -p /tmp/.X11-unix

# Load env (API keys, INSTANCE_ID, etc.)
set -o allexport
# shellcheck disable=SC1091
source "$REPO/.env"
set +o allexport

export DISPLAY=":$DISPLAY_NUM"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO:$REPO/roboeval${PYTHONPATH:+:$PYTHONPATH}"

# Start Xvfb on our display
Xvfb ":$DISPLAY_NUM" -screen 0 1280x720x24 &
XPID=$!
sleep 2

# Trap: if the sim exits (clean or crash), take Xvfb down with it.
trap 'kill $XPID 2>/dev/null || true' EXIT

cd "$REPO/roboeval"
python data_collection/demo_recorder.py \
    input_mode=Telearms \
    robot="$ROBOT" \
    env="$ENV_NAME"
