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
PIDFILE="${PIDFILE:-/tmp/playground-sim.pid}"
XVFB_PIDFILE="${XVFB_PIDFILE:-/tmp/playground-xvfb.pid}"

cd "$REPO"
git pull

source "$VENV/bin/activate"

# Kill previous playground sim + Xvfb using pidfiles (can't match by cmdline
# since venv python resolves to /usr/bin/python3.12 same as RoboEval prod).
for pf in "$PIDFILE" "$XVFB_PIDFILE"; do
    if [[ -f "$pf" ]]; then
        old_pid=$(cat "$pf" 2>/dev/null || true)
        if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "Stopping previous process $old_pid (from $pf)"
            kill "$old_pid" 2>/dev/null || true
            # Give it up to 10s to terminate cleanly before SIGKILL
            for _ in $(seq 1 10); do
                kill -0 "$old_pid" 2>/dev/null || break
                sleep 1
            done
            kill -9 "$old_pid" 2>/dev/null || true
        fi
        rm -f "$pf"
    fi
done

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
echo "$XPID" > "$XVFB_PIDFILE"
sleep 2

# Trap: if the sim exits (clean or crash), take Xvfb down with it.
trap 'kill $XPID 2>/dev/null || true; rm -f "$PIDFILE" "$XVFB_PIDFILE"' EXIT

cd "$REPO/roboeval"
python data_collection/demo_recorder.py \
    input_mode=Telearms \
    robot="$ROBOT" \
    env="$ENV_NAME" &
SIM_PID=$!
echo "$SIM_PID" > "$PIDFILE"
wait "$SIM_PID"
