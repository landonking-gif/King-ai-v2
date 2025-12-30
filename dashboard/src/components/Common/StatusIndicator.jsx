import './Common.css';

const statusConfig = {
  running: { color: '#3498db', icon: '🔄', label: 'Running' },
  completed: { color: '#27ae60', icon: '✓', label: 'Completed' },
  failed: { color: '#e74c3c', icon: '✗', label: 'Failed' },
  pending: { color: '#f39c12', icon: '⏳', label: 'Pending' },
  paused: { color: '#95a5a6', icon: '⏸', label: 'Paused' },
  active: { color: '#27ae60', icon: '●', label: 'Active' },
  inactive: { color: '#95a5a6', icon: '○', label: 'Inactive' },
};

export function StatusIndicator({ status, showLabel = true, size = 'medium' }) {
  const config = statusConfig[status] || statusConfig.pending;
  
  return (
    <span className={`status-indicator ${size}`} style={{ '--status-color': config.color }}>
      <span className="status-icon">{config.icon}</span>
      {showLabel && <span className="status-label">{config.label}</span>}
    </span>
  );
}
