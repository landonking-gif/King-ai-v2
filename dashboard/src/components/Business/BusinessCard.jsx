import { HealthBadge } from '../Common/HealthBadge';
import { ProgressBar } from '../Common/ProgressBar';
import './BusinessCard.css';

export function BusinessCard({ business, onClick }) {
  const {
    id,
    name,
    type,
    stage,
    health_score,
    revenue,
    growth_rate,
    active_tasks,
  } = business;
  
  const stageColors = {
    ideation: '#9b59b6',
    validation: '#3498db',
    launch: '#e67e22',
    growth: '#27ae60',
    scaling: '#2ecc71',
    maturity: '#1abc9c',
  };
  
  return (
    <div className="business-card" onClick={() => onClick?.(id)}>
      <div className="card-header">
        <h3 className="business-name">{name}</h3>
        <span
          className="stage-badge"
          style={{ backgroundColor: stageColors[stage] || '#95a5a6' }}
        >
          {stage}
        </span>
      </div>
      
      <div className="card-body">
        <div className="metric-row">
          <span className="metric-label">Health</span>
          <HealthBadge score={health_score} />
        </div>
        
        <div className="metric-row">
          <span className="metric-label">Revenue</span>
          <span className="metric-value">${revenue?.toLocaleString()}</span>
        </div>
        
        <div className="metric-row">
          <span className="metric-label">Growth</span>
          <span className={`metric-value ${growth_rate >= 0 ? 'positive' : 'negative'}`}>
            {growth_rate >= 0 ? '+' : ''}{growth_rate}%
          </span>
        </div>
        
        {active_tasks > 0 && (
          <div className="active-tasks">
            <span>🔄 {active_tasks} active tasks</span>
          </div>
        )}
      </div>
      
      <div className="card-footer">
        <span className="business-type">{type}</span>
      </div>
    </div>
  );
}
