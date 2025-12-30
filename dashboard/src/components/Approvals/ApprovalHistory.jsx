import { useState, useEffect } from 'react';
import { DataTable } from '../Common/DataTable';
import { RiskBadge } from './RiskBadge';
import { StatusIndicator } from '../Common/StatusIndicator';
import './Approvals.css';

export function ApprovalHistory({ businessId }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchHistory();
  }, [businessId]);
  
  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/approvals/history/${businessId}`);
      const data = await res.json();
      setHistory(data.requests || []);
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
    setLoading(false);
  };
  
  const columns = [
    { key: 'title', label: 'Action' },
    { key: 'type', label: 'Type' },
    {
      key: 'risk_level',
      label: 'Risk',
      render: (val) => <RiskBadge level={val} showLabel={false} />,
    },
    {
      key: 'status',
      label: 'Status',
      render: (val) => <StatusIndicator status={val} />,
    },
    {
      key: 'created_at',
      label: 'Created',
      render: (val) => new Date(val).toLocaleDateString(),
    },
    { key: 'reviewed_by', label: 'Reviewer' },
  ];
  
  if (loading) {
    return <div className="loading">Loading history...</div>;
  }
  
  return (
    <div className="approval-history">
      <h2>Approval History</h2>
      <DataTable columns={columns} data={history} />
    </div>
  );
}
