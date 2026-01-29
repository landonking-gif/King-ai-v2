import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import './Monitoring.css';

export function SystemStatus() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  // WebSocket disabled - using polling instead
  const connected = false;

  useEffect(() => {
    fetchHealth();
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/monitoring/health');
      const data = await res.json();
      setHealth(data);
    } catch (err) {
      console.error('Failed to fetch health:', err);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/api/monitoring/metrics');
      const data = await res.json();
      setMetrics(data);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
    setLoading(false);
  };

  if (loading) {
    return <div className="loading">Loading system status...</div>;
  }

  const statusColor = {
    healthy: '#27ae60',
    degraded: '#f39c12',
    unhealthy: '#e74c3c',
  };

  return (
    <div className="system-status">
      <div className="status-header">
        <h2>System Status</h2>
        <span className={`connection-indicator ${connected ? 'connected' : ''}`}>
          {connected ? '● Live' : '○ Disconnected'}
        </span>
      </div>

      {health && (
        <div className="health-section">
          <div
            className="overall-status"
            style={{ backgroundColor: statusColor[health.status] }}
          >
            {health.status.toUpperCase()}
          </div>

          <div className="health-checks">
            {Object.entries(health.checks).map(([name, check]) => (
              <div key={name} className="health-check">
                <span
                  className="check-indicator"
                  style={{ backgroundColor: statusColor[check.status] }}
                />
                <span className="check-name">{name}</span>
                <span className="check-latency">{check.latency_ms}ms</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {metrics && (
        <div className="metrics-section">
          <h3>Resources</h3>
          <div className="metrics-grid">
            <MetricGauge
              label="CPU"
              value={metrics.cpu_percent}
              max={100}
              unit="%"
            />
            <MetricGauge
              label="Memory"
              value={metrics.memory_percent}
              max={100}
              unit="%"
            />
            <MetricGauge
              label="Disk"
              value={metrics.disk_percent}
              max={100}
              unit="%"
            />
            <div className="metric-box">
              <span className="metric-label">Connections</span>
              <span className="metric-value">{metrics.active_connections}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricGauge({ label, value, max, unit }) {
  const percent = (value / max) * 100;
  const color = percent > 90 ? '#e74c3c' : percent > 75 ? '#f39c12' : '#27ae60';

  return (
    <div className="metric-gauge">
      <span className="gauge-label">{label}</span>
      <div className="gauge-bar">
        <div
          className="gauge-fill"
          style={{ width: `${percent}%`, backgroundColor: color }}
        />
      </div>
      <span className="gauge-value">
        {value.toFixed(1)}{unit}
      </span>
    </div>
  );
}
