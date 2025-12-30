import { useState } from 'react';
import { RiskBadge } from './RiskBadge';
import './Approvals.css';

export function ApprovalCard({ request, onAction }) {
  const [expanded, setExpanded] = useState(false);
  const [notes, setNotes] = useState('');
  const [processing, setProcessing] = useState(false);
  
  const handleApprove = async () => {
    setProcessing(true);
    await onAction(request.id, 'approve', { notes });
    setProcessing(false);
  };
  
  const handleReject = async () => {
    setProcessing(true);
    await onAction(request.id, 'reject', { notes });
    setProcessing(false);
  };
  
  const typeIcons = {
    financial: '💰',
    legal: '⚖️',
    operational: '⚙️',
    strategic: '🎯',
    technical: '🔧',
    external: '🌐',
  };
  
  return (
    <div className={`approval-card risk-${request.risk_level}`}>
      <div className="card-main" onClick={() => setExpanded(!expanded)}>
        <div className="card-icon">{typeIcons[request.type] || '📋'}</div>
        
        <div className="card-content">
          <h3 className="card-title">{request.title}</h3>
          <p className="card-desc">{request.description}</p>
          
          <div className="card-meta">
            <span className="meta-type">{request.type}</span>
            <span className="meta-time">
              ⏱ {request.waiting_hours.toFixed(1)}h waiting
            </span>
          </div>
        </div>
        
        <div className="card-risk">
          <RiskBadge level={request.risk_level} />
        </div>
        
        <span className="expand-icon">{expanded ? '▲' : '▼'}</span>
      </div>
      
      {expanded && (
        <div className="card-details">
          {request.risk_factors?.length > 0 && (
            <div className="risk-factors">
              <h4>Risk Factors</h4>
              {request.risk_factors.map((factor, i) => (
                <div key={i} className={`risk-factor severity-${factor.severity}`}>
                  <span className="factor-category">{factor.category}</span>
                  <p>{factor.description}</p>
                </div>
              ))}
            </div>
          )}
          
          <div className="payload-preview">
            <h4>Action Details</h4>
            <pre>{JSON.stringify(request.payload, null, 2)}</pre>
          </div>
          
          <div className="action-form">
            <textarea
              placeholder="Add notes (optional)..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
            />
            
            <div className="action-buttons">
              <button
                className="btn btn-reject"
                onClick={handleReject}
                disabled={processing}
              >
                Reject
              </button>
              <button
                className="btn btn-approve"
                onClick={handleApprove}
                disabled={processing}
              >
                Approve
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
