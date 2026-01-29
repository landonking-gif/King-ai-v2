cd /home/ubuntu/dashboard/src/components

# Backup
cp TalkToKingAI.jsx TalkToKingAI.jsx.backup

# Fix the WebSocket import and API call using sed
sed -i "s/import { io } from 'socket.io-client';//g" TalkToKingAI.jsx
sed -i "s/const \[socket, setSocket\] = useState(null);//g" TalkToKingAI.jsx
sed -i '/useEffect(() => {/,/return () => newSocket.close();/{ /Initialize WebSocket/,/return () => newSocket.close();/d; }' TalkToKingAI.jsx

# Rebuild
cd /home/ubuntu/dashboard
npm run build

# Restart
pkill -f 'serve.*3000'
sleep 2
nohup serve -l 3000 -s dist > /tmp/dashboard.log 2>&1 &
echo $! > /tmp/dashboard.pid

echo "Dashboard fixed and restarted"
