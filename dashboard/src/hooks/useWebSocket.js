import { useEffect, useRef, useState, useCallback } from 'react';

const RECONNECT_DELAY = 3000;
const HEARTBEAT_INTERVAL = 30000;

export function useWebSocket(url, options = {}) {
  const { userId, businessId, onEvent, autoReconnect = true } = options;
  
  const wsRef = useRef(null);
  const heartbeatRef = useRef(null);
  const reconnectRef = useRef(null);
  
  const [connected, setConnected] = useState(false);
  const [connectionId, setConnectionId] = useState(null);
  const [lastEvent, setLastEvent] = useState(null);

  const connect = useCallback(() => {
    const params = new URLSearchParams();
    if (userId) params.set('user_id', userId);
    if (businessId) params.set('business_id', businessId);
    
    const wsUrl = `${url}?${params.toString()}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      
      // Start heartbeat
      heartbeatRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, HEARTBEAT_INTERVAL);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'connected') {
          setConnectionId(data.connection_id);
        } else if (data.type === 'event') {
          setLastEvent(data);
          onEvent?.(data.event, data.data, data.timestamp);
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      setConnectionId(null);
      
      if (heartbeatRef.current) {
        clearInterval(heartbeatRef.current);
      }
      
      // Auto reconnect
      if (autoReconnect) {
        reconnectRef.current = setTimeout(connect, RECONNECT_DELAY);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    wsRef.current = ws;
  }, [url, userId, businessId, onEvent, autoReconnect]);

  const disconnect = useCallback(() => {
    if (reconnectRef.current) {
      clearTimeout(reconnectRef.current);
    }
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
  }, []);

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const subscribe = useCallback((channel) => {
    send({ type: 'subscribe', channel });
  }, [send]);

  const unsubscribe = useCallback((channel) => {
    send({ type: 'unsubscribe', channel });
  }, [send]);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    connected,
    connectionId,
    lastEvent,
    send,
    subscribe,
    unsubscribe,
    reconnect: connect,
  };
}
