import './Common.css';

export function ProgressBar({ value, max = 100, label, showPercent = true, color }) {
  const percent = Math.min(100, (value / max) * 100);
  
  const getAutoColor = (pct) => {
    if (pct >= 75) return '#27ae60';
    if (pct >= 50) return '#3498db';
    if (pct >= 25) return '#f39c12';
    return '#e74c3c';
  };
  
  const barColor = color || getAutoColor(percent);
  
  return (
    <div className="progress-container">
      {label && <span className="progress-label">{label}</span>}
      
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${percent}%`,
            backgroundColor: barColor,
          }}
        />
      </div>
      
      {showPercent && (
        <span className="progress-percent">{percent.toFixed(0)}%</span>
      )}
    </div>
  );
}
