import './Common.css';

export function StatusIndicator({ status }) {
  const statusConfig = {
    pending: { color: '#f39c12', label: 'Pending', icon: '⏳' },
    approved: { color: '#27ae60', label: 'Approved', icon: '✓' },
    rejected: { color: '#e74c3c', label: 'Rejected', icon: '✗' },
    expired: { color: '#95a5a6', label: 'Expired', icon: '⏰' },
    modified: { color: '#3498db', label: 'Modified', icon: '✎' },
  };

  const config = statusConfig[status] || statusConfig.pending;

  return (
    <span
      className="status-indicator"
      style={{ '--status-color': config.color }}
    >
      <span className="status-icon">{config.icon}</span>
      <span className="status-label">{config.label}</span>
    </span>
  );
}
