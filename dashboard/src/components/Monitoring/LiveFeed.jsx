import { useState, useEffect, useRef } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import './Monitoring.css';

export function LiveFeed({ businessId, maxItems = 50 }) {
  const [events, setEvents] = useState([]);
  const feedRef = useRef(null);

  // WebSocket disabled - using polling instead
  const connected = false;

  const eventIcons = {
    'business.updated': '🏢',
    'task.started': '▶️',
    'task.completed': '✅',
    'task.failed': '❌',
    'playbook.started': '📋',
    'playbook.completed': '🎉',
    'approval.required': '⚠️',
    'approval.granted': '✓',
    'system.alert': '🚨',
    'analytics.kpi_update': '📊',
  };

  return (
    <div className="live-feed">
      <div className="feed-header">
        <h3>Live Activity</h3>
        <span className={`live-indicator ${connected ? 'active' : ''}`}>
          {connected ? '● LIVE' : '○ OFFLINE'}
        </span>
      </div>

      <div className="feed-list" ref={feedRef}>
        {events.length === 0 ? (
          <div className="feed-empty">
            <p>Waiting for events...</p>
          </div>
        ) : (
          events.map((e) => (
            <div key={e.id} className="feed-item">
              <span className="feed-icon">
                {eventIcons[e.event] || '📌'}
              </span>
              <div className="feed-content">
                <span className="feed-event">{e.event}</span>
                <span className="feed-time">
                  {new Date(e.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
