#!/bin/bash

echo "Rebuilding dashboard with fixes..."
cd /home/ubuntu/dashboard

# Check if components were updated
echo "Checking updated files:"
ls -lh src/components/TalkToKingAI.jsx src/components/CommandCenter.jsx src/components/Analytics.jsx

echo ""
echo "Building dashboard..."
npm run build

if [ $? -eq 0 ]; then
    echo "✓ Build successful"
    
    # Restart dashboard service
    echo "Restarting dashboard service..."
    pkill -f 'serve.*3000'
    sleep 2
    
    BUILD_DIR="dist"
    nohup serve -l 3000 -s $BUILD_DIR > /tmp/dashboard.log 2>&1 &
    echo $! > /tmp/dashboard.pid
    
    sleep 3
    
    if kill -0 $(cat /tmp/dashboard.pid) 2>/dev/null; then
        echo "✓ Dashboard restarted successfully (PID: $(cat /tmp/dashboard.pid))"
    else
        echo "✗ Dashboard failed to start"
    fi
else
    echo "✗ Build failed"
    exit 1
fi
