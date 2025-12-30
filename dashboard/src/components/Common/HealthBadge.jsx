import './Common.css';

export function HealthBadge({ score, showLabel = true }) {
  const getColor = (score) => {
    if (score >= 80) return '#27ae60';
    if (score >= 60) return '#f39c12';
    if (score >= 40) return '#e67e22';
    return '#e74c3c';
  };
  
  const getLabel = (score) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Poor';
  };
  
  return (
    <div className="health-badge" style={{ '--health-color': getColor(score) }}>
      <span className="health-score">{score}</span>
      {showLabel && <span className="health-label">{getLabel(score)}</span>}
    </div>
  );
}
