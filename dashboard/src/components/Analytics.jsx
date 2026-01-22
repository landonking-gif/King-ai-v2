import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import './Analytics.css';

const Analytics = () => {
  const [metrics, setMetrics] = useState({
    systemHealth: 95,
    agentPerformance: 87,
    activityLevel: 72,
    throughput: 150
  });

  const [charts, setCharts] = useState([]);
  const [activityFeed, setActivityFeed] = useState([
    { time: '2 min ago', desc: 'Workflow execution completed' },
    { time: '5 min ago', desc: 'Agent spawned successfully' },
    { time: '10 min ago', desc: 'Approval request processed' }
  ]);

  // WebSocket connection for real-time updates
  const { connected, subscribe } = useWebSocket('ws://localhost:8100/ws/activity-feed', {
    onEvent: (eventType, data, timestamp) => {
      handleWebSocketEvent(eventType, data, timestamp);
    }
  });

  const handleWebSocketEvent = (eventType, data, timestamp) => {
    switch (eventType) {
      case 'analytics.kpi_update':
        setMetrics(prev => ({ ...prev, ...data.kpis }));
        break;
      case 'task.completed':
      case 'task.started':
      case 'task.failed':
        addActivityItem(`${data.status}: ${data.task_id}`, timestamp);
        break;
      case 'approval.required':
        addActivityItem(`Approval required: ${data.title}`, timestamp);
        break;
      case 'system.alert':
        addActivityItem(`System alert: ${data.message}`, timestamp);
        break;
      default:
        break;
    }
  };

  const addActivityItem = (description, timestamp) => {
    const timeAgo = getTimeAgo(new Date(timestamp));
    setActivityFeed(prev => [
      { time: timeAgo, desc: description },
      ...prev.slice(0, 9) // Keep only last 10 items
    ]);
  };

  const getTimeAgo = (date) => {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'Just now';
    if (diffMins === 1) return '1 min ago';
    return `${diffMins} min ago`;
  };

  useEffect(() => {
    // Fetch initial analytics data
    fetchAnalyticsData();

    // Subscribe to analytics channel
    if (connected) {
      subscribe('analytics');
    }
  }, [connected, subscribe]);

  const fetchAnalyticsData = async () => {
    try {
      // Mock data for now - in production this would fetch from API
      setCharts([
        { id: 1, title: 'System Health Over Time', data: [90, 92, 95, 93, 95] },
        { id: 2, title: 'Agent Performance Metrics', data: [85, 87, 89, 87, 87] },
        { id: 3, title: 'Activity Monitoring', data: [60, 70, 75, 72, 72] }
      ]);
    } catch (error) {
      console.error('Failed to fetch analytics data:', error);
    }
  };

  return (
    <div className="analytics-dashboard">
      <div className="analytics-header">
        <h2>Analytics Dashboard</h2>
        <p>Performance metrics, system health, and activity monitoring</p>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>System Health</h3>
          <div className="metric-value">{metrics.systemHealth}%</div>
          <div className="metric-bar">
            <div className="metric-fill" style={{ width: `${metrics.systemHealth}%` }}></div>
          </div>
        </div>

        <div className="metric-card">
          <h3>Agent Performance</h3>
          <div className="metric-value">{metrics.agentPerformance}%</div>
          <div className="metric-bar">
            <div className="metric-fill" style={{ width: `${metrics.agentPerformance}%` }}></div>
          </div>
        </div>

        <div className="metric-card">
          <h3>Activity Level</h3>
          <div className="metric-value">{metrics.activityLevel}%</div>
          <div className="metric-bar">
            <div className="metric-fill" style={{ width: `${metrics.activityLevel}%` }}></div>
          </div>
        </div>

        <div className="metric-card">
          <h3>Throughput</h3>
          <div className="metric-value">{metrics.throughput}</div>
          <span className="metric-unit">req/min</span>
        </div>
      </div>

      <div className="charts-section">
        <h3>Performance Charts</h3>
        <div className="charts-grid">
          {charts.map(chart => (
            <div key={chart.id} className="chart-card">
              <h4>{chart.title}</h4>
              <div className="chart-placeholder">
                {/* Placeholder for actual chart */}
                <div className="chart-data">
                  {chart.data.join(', ')}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="activity-monitor">
        <h3>Recent Activity</h3>
        <div className="connection-status">
          <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></span>
          {connected ? 'Live Updates Active' : 'Connecting...'}
        </div>
        <div className="activity-list">
          {activityFeed.map((item, index) => (
            <div key={index} className="activity-item">
              <span className="activity-time">{item.time}</span>
              <span className="activity-desc">{item.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Analytics;