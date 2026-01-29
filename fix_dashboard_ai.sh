#!/bin/bash

echo "=== Dashboard AI Fix Script ==="
echo ""

# Define paths
DASHBOARD_DIR="/home/ubuntu/dashboard"
BACKUP_DIR="/tmp/dashboard_backup_$(date +%s)"

# Backup current components
echo "1. Creating backup..."
mkdir -p $BACKUP_DIR
cp $DASHBOARD_DIR/src/components/TalkToKingAI.jsx $BACKUP_DIR/ 2>/dev/null || true
cp $DASHBOARD_DIR/src/components/CommandCenter.jsx $BACKUP_DIR/ 2>/dev/null || true
cp $DASHBOARD_DIR/src/components/Analytics.jsx $BACKUP_DIR/ 2>/dev/null || true
echo "   Backup created at: $BACKUP_DIR"

# Fix TalkToKingAI.jsx - Remove WebSocket and fix API call
echo "2. Fixing TalkToKingAI.jsx..."
cat > $DASHBOARD_DIR/src/components/TalkToKingAI.jsx.new << 'EOFFILE'
import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Workflow, Play, CheckCircle, AlertCircle } from 'lucide-react';

const TalkToKingAI = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Hello! I\'m King AI. How can I help you today? You can ask me to create workflows, analyze data, or execute tasks.',
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentWorkflow, setCurrentWorkflow] = useState(null);
  const [executionStatus, setExecutionStatus] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date(),
      type: 'text'
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Send to backend
    try {
      const response = await fetch('/api/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          text: userMessage.content,
          user_id: 'dashboard-user',
          business_id: 'default-business',
          agent_id: 'primary'
        })
      });

      if (!response.ok) {
        throw new Error(\`API Error: \${response.status}\`);
      }

      const data = await response.json();
      
      // Add AI response to messages
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        type: 'text'
      }]);
      setIsLoading(false);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'system',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        type: 'error'
      }]);
      setIsLoading(false);
    }
  };
EOFFILE

# Append the rest of the file content (keeping existing UI code)
tail -n +100 $DASHBOARD_DIR/src/components/TalkToKingAI.jsx >> $DASHBOARD_DIR/src/components/TalkToKingAI.jsx.new 2>/dev/null || true
mv $DASHBOARD_DIR/src/components/TalkToKingAI.jsx.new $DASHBOARD_DIR/src/components/TalkToKingAI.jsx

echo "   ✓ TalkToKingAI.jsx updated"

# Rebuild dashboard
echo "3. Rebuilding dashboard..."
cd $DASHBOARD_DIR
npm run build 2>&1 | tail -10

if [ $? -eq 0 ]; then
    echo "   ✓ Build successful"
    
    # Restart dashboard service
    echo "4. Restarting dashboard service..."
    pkill -f 'serve.*3000'
    sleep 2
    
    BUILD_DIR="dist"
    nohup serve -l 3000 -s $BUILD_DIR > /tmp/dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    echo $DASHBOARD_PID > /tmp/dashboard.pid
    
    sleep 3
    
    if kill -0 $DASHBOARD_PID 2>/dev/null; then
        echo "   ✓ Dashboard restarted successfully (PID: $DASHBOARD_PID)"
        echo ""
        echo "=== Dashboard Fixed! ==="
        echo "Access at: http://52.90.242.99/"
        echo "The AI chat should now work properly."
    else
        echo "   ✗ Dashboard failed to start - check /tmp/dashboard.log"
    fi
else
    echo "   ✗ Build failed - restoring backup"
    cp $BACKUP_DIR/* $DASHBOARD_DIR/src/components/
    exit 1
fi
