import './Approvals.css';

const riskConfig = {
  critical: { color: '#c0392b', icon: '🔴', label: 'Critical' },
  high: { color: '#e74c3c', icon: '🟠', label: 'High' },
  medium: { color: '#f39c12', icon: '🟡', label: 'Medium' },
  low: { color: '#27ae60', icon: '🟢', label: 'Low' },
};

export function RiskBadge({ level, showLabel = true }) {
  const config = riskConfig[level] || riskConfig.medium;
  
  return (
    <span
      className="risk-badge"
      style={{ '--risk-color': config.color }}
    >
      <span className="risk-icon">{config.icon}</span>
      {showLabel && <span className="risk-label">{config.label}</span>}
    </span>
  );
}
