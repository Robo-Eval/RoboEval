#!/usr/bin/env bash
# cam_cli.sh — launch the interactive camera editor on staging over SSH.
#
# Pulls latest on staging (so you're running the code you just pushed),
# activates the sim's venv for pyyaml, then opens tools/cam_cli.py under
# a real TTY. Arrow keys orbit the live external camera; the sim picks
# up each keystroke on the next frame.
#
# Exit: Esc or x (keeps override) / c (clears override).
#
set -euo pipefail

HOST="${TELEARMS_STAGING_HOST:-thorsim-staging}"
REPO="/home/ubuntu/telearms-roboplayground"
VENV="/home/ubuntu/playground-venv"

exec ssh -t -o UpdateHostKeys=no "$HOST" \
    "cd '$REPO' && git pull --ff-only >/dev/null && source '$VENV/bin/activate' && python tools/cam_cli.py"
