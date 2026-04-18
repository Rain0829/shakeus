#!/bin/bash
# Run everything on the Pi.
# Usage: bash start_pi.sh
# Stop:  Ctrl+C  (kills both processes)

cd "$(dirname "$0")"

echo "Starting Flask server (song server + alarm trigger)..."
python3 main.py &
FLASK_PID=$!

# Give Flask a moment to bind the port
sleep 2

echo "Starting alarm scheduler (alarm time: $(python3 -c 'from config import ALARM_TIME; print(ALARM_TIME)'))..."
python3 alarm/scheduler.py &
SCHED_PID=$!

echo ""
echo "Both processes running."
echo "  Flask PID:     $FLASK_PID"
echo "  Scheduler PID: $SCHED_PID"
echo ""
echo "Press Ctrl+C to stop everything."

# Forward Ctrl+C to both child processes
trap "echo 'Stopping...'; kill $FLASK_PID $SCHED_PID; exit 0" SIGINT SIGTERM
wait
