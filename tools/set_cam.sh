#!/usr/bin/env bash
# set_cam.sh — set (or clear) the live external camera pose on staging.
#
# Writes telearms/cam_override.yaml on the staging host over SSH. The running
# TelearmsTeleop process polls that file every tick and re-applies cam_pos /
# cam_quat / fovy on the next frame — no restart, no interruption to the
# Agora stream.
#
# Usage:
#   tools/set_cam.sh -p "-0.3 0 2.5" -t "0.4 0 0.8" -f 60
#   tools/set_cam.sh --clear              # drop the override, back to world.xml
#   tools/set_cam.sh --show               # print the current override on staging
#
set -euo pipefail

HOST="${TELEARMS_STAGING_HOST:-thorsim-staging}"
REMOTE_PATH="/home/ubuntu/telearms-roboplayground/telearms/cam_override.yaml"

POS=""
TARGET=""
QUAT=""
FOVY=""
MODE="set"

usage() {
    cat <<'EOF'
Usage: set_cam.sh [options]

Live-edit the external camera on the staging sim.

Options:
  -p, --pos "x y z"         Camera position (world space, meters).
  -t, --target "x y z"      Look-at target (meters). Quat is computed from
                            pos + target using the same convention as
                            tools/calibrate_external.py.
  -q, --quat "w x y z"      Direct quaternion override (alternative to --target).
  -f, --fovy <degrees>      Vertical field of view.
      --clear               Remove the override file (falls back to world.xml).
      --show                Print the current override file.
      --host <alias>        SSH host alias (default: $TELEARMS_STAGING_HOST or
                            thorsim-staging).
  -h, --help                This message.

Notes:
  Commas inside the triplets are OK: "-0.3, 0, 2.5" and "-0.3 0 2.5" both work.

Examples:
  tools/set_cam.sh -p "-0.3 0 2.5" -t "0.4 0 0.8" -f 60
  tools/set_cam.sh --clear
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--pos)    POS="$2"; shift 2 ;;
        -t|--target) TARGET="$2"; shift 2 ;;
        -q|--quat)   QUAT="$2"; shift 2 ;;
        -f|--fovy)   FOVY="$2"; shift 2 ;;
        --clear)     MODE="clear"; shift ;;
        --show)      MODE="show"; shift ;;
        --host)      HOST="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

ssh_run() {
    ssh -o UpdateHostKeys=no "$HOST" "$@"
}

if [[ "$MODE" == "clear" ]]; then
    ssh_run "rm -f '$REMOTE_PATH'"
    echo "[set_cam] cleared $HOST:$REMOTE_PATH"
    exit 0
fi

if [[ "$MODE" == "show" ]]; then
    ssh_run "cat '$REMOTE_PATH' 2>/dev/null || echo '(no override)'"
    exit 0
fi

# Split a "a b c[ d]" or "a, b, c[, d]" string into an array.
split_triplet() {
    local raw="${1//,/ }"
    # shellcheck disable=SC2206
    echo ${raw}
}

build_yaml() {
    if [[ -n "$POS" ]]; then
        read -r -a c <<<"$(split_triplet "$POS")"
        [[ ${#c[@]} -eq 3 ]] || { echo "--pos needs 3 numbers, got: $POS" >&2; exit 2; }
        echo "pos: [${c[0]}, ${c[1]}, ${c[2]}]"
    fi
    if [[ -n "$TARGET" ]]; then
        read -r -a c <<<"$(split_triplet "$TARGET")"
        [[ ${#c[@]} -eq 3 ]] || { echo "--target needs 3 numbers, got: $TARGET" >&2; exit 2; }
        echo "target: [${c[0]}, ${c[1]}, ${c[2]}]"
    fi
    if [[ -n "$QUAT" ]]; then
        read -r -a c <<<"$(split_triplet "$QUAT")"
        [[ ${#c[@]} -eq 4 ]] || { echo "--quat needs 4 numbers (w x y z), got: $QUAT" >&2; exit 2; }
        echo "quat: [${c[0]}, ${c[1]}, ${c[2]}, ${c[3]}]"
    fi
    if [[ -n "$FOVY" ]]; then
        echo "fovy: $FOVY"
    fi
}

YAML="$(build_yaml)"
if [[ -z "$YAML" ]]; then
    echo "[set_cam] nothing to set. Pass at least one of --pos/--target/--quat/--fovy, or --clear." >&2
    usage >&2
    exit 2
fi

# Pipe YAML into a remote cat — avoids heredoc quoting gotchas with negatives.
printf '%s\n' "$YAML" | ssh_run "cat > '$REMOTE_PATH'"

echo "[set_cam] wrote $HOST:$REMOTE_PATH:"
printf '%s\n' "$YAML" | sed 's/^/  /'
