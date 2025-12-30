import { useState, useEffect } from 'react';
import { ApprovalCard } from './ApprovalCard';
import './Approvals.css';

export function ApprovalQueue({ businessId, onApprove, onReject }) {
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  
  useEffect(() => {
    fetchPending();
  }, [businessId]);
  
  const fetchPending = async () => {
    setLoading(true);
    try {
      const url = businessId
        ? `/api/approvals/pending?business_id=${businessId}`
        : '/api/approvals/pending';
      const res = await fetch(url);
      const data = await res.json();
      setRequests(data.requests || []);
    } catch (err) {
      console.error('Failed to fetch approvals:', err);
    }
    setLoading(false);
  };
  
  const filteredRequests = requests.filter(r => {
    if (filter === 'all') return true;
    return r.risk_level === filter;
  });
  
  const handleAction = async (id, action, data) => {
    if (action === 'approve') {
      await onApprove?.(id, data);
    } else {
      await onReject?.(id, data);
    }
    fetchPending();
  };
  
  if (loading) {
    return <div className="loading">Loading approvals...</div>;
  }
  
  return (
    <div className="approval-queue">
      <div className="queue-header">
        <h2>Pending Approvals</h2>
        <span className="count-badge">{requests.length}</span>
        
        <div className="filter-tabs">
          {['all', 'critical', 'high', 'medium', 'low'].map(level => (
            <button
              key={level}
              className={`filter-tab ${filter === level ? 'active' : ''}`}
              onClick={() => setFilter(level)}
            >
              {level}
            </button>
          ))}
        </div>
      </div>
      
      <div className="queue-list">
        {filteredRequests.length === 0 ? (
          <div className="empty-queue">
            <span className="empty-icon">✓</span>
            <p>No pending approvals</p>
          </div>
        ) : (
          filteredRequests.map(req => (
            <ApprovalCard
              key={req.id}
              request={req}
              onAction={handleAction}
            />
          ))
        )}
      </div>
    </div>
  );
}
