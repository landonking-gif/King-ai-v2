import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import './Monitoring.css';

export function AlertBanner() {
  const [alerts, setAlerts] = useState([]);

  useWebSocket(
    `ws://${window.location.host}/api/monitoring/ws`,
    {
      onEvent: (event, data) => {
        if (event === 'system.alert') {
          setAlerts((prev) => [...prev, { ...data, id: Date.now() }]);
        }
      },
    }
  );

  const dismissAlert = (id) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  };

  if (alerts.length === 0) return null;

  return (
    <div className="alert-banner-container">
      {alerts.map((alert) => (
        <div
          key={alert.id}
          className={`alert-banner alert-${alert.level}`}
        >
          <span className="alert-icon">
            {alert.level === 'critical' ? '🚨' : '⚠️'}
          </span>
          <span className="alert-message">{alert.message}</span>
          <button
            className="alert-dismiss"
            onClick={() => dismissAlert(alert.id)}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
