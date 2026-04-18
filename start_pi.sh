#!/bin/bash
# Run everything on the Pi (headless).
# Usage: bash start_pi.sh
# Stop:  Ctrl+C

cd "$(dirname "$0")"

get_host_ip() {
  local ip=""

  if command -v hostname >/dev/null 2>&1; then
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi

  if [ -z "$ip" ] && command -v ipconfig >/dev/null 2>&1; then
    ip="$(ipconfig getifaddr en0 2>/dev/null)"
  fi

  if [ -z "$ip" ] && command -v ipconfig >/dev/null 2>&1; then
    ip="$(ipconfig getifaddr en1 2>/dev/null)"
  fi

  if [ -z "$ip" ]; then
    ip="localhost"
  fi

  printf '%s\n' "$ip"
}

echo "Starting stream server on port 8000..."
python3 -m uvicorn stream:app --host 0.0.0.0 --port 8000 &
STREAM_PID=$!

# Give the server a moment to bind
sleep 3

echo "Starting alarm scheduler (alarm time: $(python3 -c 'from config import ALARM_TIME; print(ALARM_TIME)'))..."
python3 -m alarm.scheduler &
SCHED_PID=$!

echo ""
echo "Running. Open http://$(get_host_ip):8000 on your phone/laptop."
echo "  Stream PID:    $STREAM_PID"
echo "  Scheduler PID: $SCHED_PID"
echo ""
echo "Press Ctrl+C to stop everything."

trap "echo 'Stopping...'; kill $STREAM_PID $SCHED_PID; exit 0" SIGINT SIGTERM
wait
