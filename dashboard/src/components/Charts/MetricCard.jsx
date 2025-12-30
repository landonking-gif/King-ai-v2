import './Charts.css';

export function MetricCard({
  title,
  value,
  change,
  changeLabel,
  icon,
  trend,
}) {
  const isPositive = change >= 0;
  const trendIcon = trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→';
  
  return (
    <div className="metric-card">
      <div className="metric-header">
        {icon && <span className="metric-icon">{icon}</span>}
        <span className="metric-title">{title}</span>
      </div>
      
      <div className="metric-value-large">{value}</div>
      
      {change !== undefined && (
        <div className={`metric-change ${isPositive ? 'positive' : 'negative'}`}>
          <span className="trend-icon">{trendIcon}</span>
          <span>{isPositive ? '+' : ''}{change}%</span>
          {changeLabel && <span className="change-label">{changeLabel}</span>}
        </div>
      )}
    </div>
  );
}
