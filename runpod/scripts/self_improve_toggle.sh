#!/usr/bin/env bash
set -euo pipefail

# Toggle the self-improve loop on/off without restarting the pod.
#
# The loop in entrypoint.sh polls `gs://$GCS_BUCKET/deepsculpt/control/self_improve.enabled`
# every $POLL_INTERVAL seconds (default 60). This script writes "on" or "off"
# to that object, so you can flip the switch from anywhere with `gsutil` access.
#
# Usage:
#   self_improve_toggle.sh on
#   self_improve_toggle.sh off
#   self_improve_toggle.sh status

state="${1:-status}"
GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET is required}"
GCS_CONTROL="gs://${GCS_BUCKET}/deepsculpt/control/self_improve.enabled"

write_state() {
    local val="$1"
    local tmp
    tmp="$(mktemp)"
    echo "$val" > "$tmp"
    gsutil -q cp "$tmp" "$GCS_CONTROL"
    rm "$tmp"
}

case "$state" in
    on|ON|enable|start)
        write_state "on"
        echo "=== self-improve ENABLED — next poll cycle (<=POLL_INTERVAL) will pick this up ==="
        ;;
    off|OFF|disable|stop)
        write_state "off"
        echo "=== self-improve DISABLED — current iteration (if any) finishes, no new ones launch ==="
        ;;
    status)
        if gsutil -q cat "$GCS_CONTROL" 2>/dev/null; then
            :
        else
            echo "(no toggle object yet — defaults to OFF)"
        fi
        ;;
    *)
        echo "Usage: $0 {on|off|status}" >&2
        exit 1
        ;;
esac
