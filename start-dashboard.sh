#!/bin/bash
cd /home/ubuntu/dashboard
pkill -f 'serve.*3000' || true
sleep 2
npx serve -l 3000 -s dist > /tmp/dashboard.log 2>&1 &
PID=$!
echo $PID > /tmp/dashboard.pid
echo "Dashboard PID: $PID"
sleep 3
ps -p $PID && echo "Dashboard is running"
curl -s http://localhost:3000 | head -5
