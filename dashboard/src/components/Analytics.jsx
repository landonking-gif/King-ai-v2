import React, { useState, useEffect } from 'react';
import './Analytics.css';

const Analytics = () => {
  const [metrics, setMetrics] = useState({
    systemHealth: 95,
    agentPerformance: 87,
    activityLevel: 72,
    throughput: 150
  });

  const [charts, setCharts] = useState([]);

  useEffect(() => {
    // Fetch analytics data
    fetchAnalyticsData();
  }, []);

  const fetchAnalyticsData = async () => {
    try {
      // Mock data for now
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
        <div className="activity-list">
          <div className="activity-item">
            <span className="activity-time">2 min ago</span>
            <span className="activity-desc">Workflow execution completed</span>
          </div>
          <div className="activity-item">
            <span className="activity-time">5 min ago</span>
            <span className="activity-desc">Agent spawned successfully</span>
          </div>
          <div className="activity-item">
            <span className="activity-time">10 min ago</span>
            <span className="activity-desc">Approval request processed</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;