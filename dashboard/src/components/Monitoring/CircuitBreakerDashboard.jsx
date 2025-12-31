import { useState, useEffect, useCallback, useRef } from 'react';
import { MetricCard } from '../Charts/MetricCard';
import './Monitoring.css';

/**
 * CircuitBreakerDashboard - Real-time circuit breaker monitoring
 * 
 * Shows:
 * - Status of all circuit breakers (open, closed, half-open)
 * - Request counts and failure rates
 * - Recent failures with timestamps
 * - Manual reset controls
 */
export function CircuitBreakerDashboard({ 
  apiBaseUrl = '/api/v1/system',
  refreshInterval = 5000,
  onReset,
}) {
  const [circuits, setCircuits] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resetting, setResetting] = useState(null);
  const intervalRef = useRef(null);
  
  const fetchCircuits = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/circuit-breakers`);
      if (!response.ok) throw new Error('Failed to fetch circuit breakers');
      const data = await response.json();
      setCircuits(data.circuits || {});
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);
  
  useEffect(() => {
    fetchCircuits();
    intervalRef.current = setInterval(fetchCircuits, refreshInterval);
    return () => clearInterval(intervalRef.current);
  }, [fetchCircuits, refreshInterval]);
  
  const handleReset = async (name) => {
    setResetting(name);
    try {
      const response = await fetch(`${apiBaseUrl}/circuit-breakers/${name}/reset`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to reset circuit breaker');
      await fetchCircuits();
      if (onReset) onReset(name);
    } catch (err) {
      setError(err.message);
    } finally {
      setResetting(null);
    }
  };
  
  const getStateColor = (state) => {
    switch (state) {
      case 'closed': return 'var(--success)';
      case 'open': return 'var(--danger)';
      case 'half_open': return 'var(--warning)';
      default: return 'var(--text-tertiary)';
    }
  };
  
  const getStateIcon = (state) => {
    switch (state) {
      case 'closed': return '✓';
      case 'open': return '✕';
      case 'half_open': return '⟳';
      default: return '?';
    }
  };
  
  if (loading) {
    return (
      <div className="circuit-dashboard loading">
        <div className="loading-spinner" />
        <p>Loading circuit breakers...</p>
      </div>
    );
  }
  
  const circuitList = Object.entries(circuits);
  const openCount = circuitList.filter(([, c]) => c.state === 'open').length;
  const halfOpenCount = circuitList.filter(([, c]) => c.state === 'half_open').length;
  
  return (
    <div className="circuit-dashboard">
      <div className="circuit-header">
        <h2>Circuit Breakers</h2>
        <div className="circuit-summary">
          <span className="summary-badge healthy">
            {circuitList.length - openCount - halfOpenCount} Healthy
          </span>
          {halfOpenCount > 0 && (
            <span className="summary-badge warning">{halfOpenCount} Recovering</span>
          )}
          {openCount > 0 && (
            <span className="summary-badge danger">{openCount} Open</span>
          )}
        </div>
      </div>
      
      {error && (
        <div className="circuit-error">
          <span>⚠️ {error}</span>
          <button onClick={fetchCircuits}>Retry</button>
        </div>
      )}
      
      <div className="circuit-grid">
        {circuitList.map(([name, circuit]) => (
          <CircuitCard
            key={name}
            name={name}
            circuit={circuit}
            stateColor={getStateColor(circuit.state)}
            stateIcon={getStateIcon(circuit.state)}
            onReset={() => handleReset(name)}
            resetting={resetting === name}
          />
        ))}
      </div>
      
      {circuitList.length === 0 && (
        <div className="circuit-empty">
          <p>No circuit breakers registered</p>
        </div>
      )}
    </div>
  );
}

function CircuitCard({ name, circuit, stateColor, stateIcon, onReset, resetting }) {
  const [showDetails, setShowDetails] = useState(false);
  
  const failureRate = circuit.total_calls > 0
    ? ((circuit.failed_calls / circuit.total_calls) * 100).toFixed(1)
    : '0.0';
  
  return (
    <div className={`circuit-card state-${circuit.state}`}>
      <div className="circuit-card-header">
        <div className="circuit-name">
          <span className="state-indicator" style={{ color: stateColor }}>
            {stateIcon}
          </span>
          <h3>{name}</h3>
        </div>
        <span className="circuit-state" style={{ color: stateColor }}>
          {circuit.state.replace('_', ' ')}
        </span>
      </div>
      
      <div className="circuit-metrics">
        <div className="metric">
          <span className="metric-value">{circuit.total_calls}</span>
          <span className="metric-label">Total</span>
        </div>
        <div className="metric">
          <span className="metric-value success">{circuit.successful_calls}</span>
          <span className="metric-label">Success</span>
        </div>
        <div className="metric">
          <span className="metric-value danger">{circuit.failed_calls}</span>
          <span className="metric-label">Failed</span>
        </div>
        <div className="metric">
          <span className="metric-value warning">{circuit.rejected_calls}</span>
          <span className="metric-label">Rejected</span>
        </div>
      </div>
      
      <div className="circuit-progress">
        <div className="progress-bar">
          <div 
            className="progress-fill success"
            style={{ width: `${100 - parseFloat(failureRate)}%` }}
          />
          <div 
            className="progress-fill danger"
            style={{ width: `${failureRate}%` }}
          />
        </div>
        <span className="progress-label">{failureRate}% failure rate</span>
      </div>
      
      <div className="circuit-actions">
        <button 
          className="details-btn"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? 'Hide Details' : 'Show Details'}
        </button>
        
        {circuit.state === 'open' && (
          <button 
            className="reset-btn"
            onClick={onReset}
            disabled={resetting}
          >
            {resetting ? 'Resetting...' : 'Reset'}
          </button>
        )}
      </div>
      
      {showDetails && (
        <div className="circuit-details">
          <div className="detail-row">
            <span>Consecutive Failures:</span>
            <span>{circuit.consecutive_failures}</span>
          </div>
          <div className="detail-row">
            <span>State Changes:</span>
            <span>{circuit.state_changes}</span>
          </div>
          <div className="detail-row">
            <span>Failure Threshold:</span>
            <span>{circuit.config?.failure_threshold || '-'}</span>
          </div>
          <div className="detail-row">
            <span>Timeout:</span>
            <span>{circuit.config?.timeout || '-'}s</span>
          </div>
          
          {circuit.last_failure && (
            <div className="detail-row">
              <span>Last Failure:</span>
              <span>{new Date(circuit.last_failure).toLocaleString()}</span>
            </div>
          )}
          
          {circuit.recent_failures?.length > 0 && (
            <div className="recent-failures">
              <h4>Recent Failures</h4>
              <ul>
                {circuit.recent_failures.map((failure, i) => (
                  <li key={i}>
                    <span className="failure-time">
                      {new Date(failure.time).toLocaleTimeString()}
                    </span>
                    <span className="failure-type">{failure.type}</span>
                    <span className="failure-error">{failure.error}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * SystemHealthDashboard - Overall system health monitoring
 */
export function SystemHealthDashboard({ apiBaseUrl = '/api/v1/system' }) {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/health`);
        const data = await response.json();
        setHealth(data);
      } catch (err) {
        setHealth({ status: 'unhealthy', error: err.message });
      } finally {
        setLoading(false);
      }
    };
    
    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, [apiBaseUrl]);
  
  if (loading) {
    return <div className="health-loading">Checking system health...</div>;
  }
  
  const statusColor = {
    healthy: 'var(--success)',
    degraded: 'var(--warning)',
    unhealthy: 'var(--danger)',
  }[health?.status] || 'var(--text-tertiary)';
  
  return (
    <div className="health-dashboard">
      <div className="health-header">
        <div className="health-status" style={{ borderColor: statusColor }}>
          <span className="status-icon" style={{ color: statusColor }}>
            {health?.status === 'healthy' ? '✓' : health?.status === 'degraded' ? '⚠' : '✕'}
          </span>
          <div>
            <h2>System Status</h2>
            <span style={{ color: statusColor }}>{health?.status?.toUpperCase()}</span>
          </div>
        </div>
        
        <div className="health-meta">
          <span>Version: {health?.version}</span>
          <span>Uptime: {formatUptime(health?.uptime_seconds)}</span>
        </div>
      </div>
      
      <div className="health-checks">
        {Object.entries(health?.checks || {}).map(([name, check]) => (
          <div 
            key={name}
            className={`health-check ${check.status}`}
          >
            <span className="check-icon">
              {check.status === 'healthy' ? '✓' : '✕'}
            </span>
            <span className="check-name">{name}</span>
            {check.error && (
              <span className="check-error">{check.error}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function formatUptime(seconds) {
  if (!seconds) return '-';
  
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

export default CircuitBreakerDashboard;
