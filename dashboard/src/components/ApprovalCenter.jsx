import React, { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Clock, AlertTriangle, Eye } from 'lucide-react';

const ApprovalCenter = () => {
  const [approvals, setApprovals] = useState([
    {
      id: '1',
      title: 'High-risk workflow execution',
      description: 'Deploy new business unit with $50K budget',
      priority: 'high',
      risk_level: 'high',
      ai_thinking: 'This workflow involves significant financial commitment. Risk factors include market volatility and execution complexity.',
      provenance_chain: ['User Request', 'AI Analysis', 'Risk Assessment', 'Approval Required'],
      expires_at: new Date(Date.now() + 3600000).toISOString(), // 1 hour from now
      status: 'pending'
    },
    {
      id: '2',
      title: 'Medium-risk agent spawn',
      description: 'Create new marketing agent for social media campaign',
      priority: 'medium',
      risk_level: 'medium',
      ai_thinking: 'Moderate risk due to content generation capabilities. Standard safety protocols should mitigate concerns.',
      provenance_chain: ['Campaign Request', 'Agent Selection', 'Capability Check'],
      expires_at: new Date(Date.now() + 7200000).toISOString(), // 2 hours from now
      status: 'pending'
    }
  ]);

  const [selectedApprovals, setSelectedApprovals] = useState([]);

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high': return 'text-red-500';
      case 'medium': return 'text-yellow-500';
      case 'low': return 'text-green-500';
      default: return 'text-gray-500';
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'high': return 'bg-red-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const calculateTimeRemaining = (expiresAt) => {
    const now = new Date();
    const expiry = new Date(expiresAt);
    const diff = expiry - now;

    if (diff <= 0) return 'Expired';

    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));

    return `${hours}h ${minutes}m`;
  };

  const handleApprove = async (id) => {
    // In real app, call API
    setApprovals(prev => prev.map(app =>
      app.id === id ? { ...app, status: 'approved' } : app
    ));
  };

  const handleReject = async (id) => {
    setApprovals(prev => prev.map(app =>
      app.id === id ? { ...app, status: 'rejected' } : app
    ));
  };

  const handleBulkApprove = () => {
    selectedApprovals.forEach(id => handleApprove(id));
    setSelectedApprovals([]);
  };

  const handleBulkReject = () => {
    selectedApprovals.forEach(id => handleReject(id));
    setSelectedApprovals([]);
  };

  const toggleSelection = (id) => {
    setSelectedApprovals(prev =>
      prev.includes(id)
        ? prev.filter(item => item !== id)
        : [...prev, id]
    );
  };

  return (
    <div className="approval-center">
      {/* Bulk Actions */}
      {selectedApprovals.length > 0 && (
        <div className="card glass mb-6">
          <div className="flex items-center justify-between">
            <span>{selectedApprovals.length} approval(s) selected</span>
            <div className="flex gap-4">
              <button
                className="btn-primary flex items-center gap-2"
                onClick={handleBulkApprove}
              >
                <CheckCircle className="w-4 h-4" />
                Bulk Approve
              </button>
              <button
                className="btn-secondary flex items-center gap-2"
                onClick={handleBulkReject}
              >
                <XCircle className="w-4 h-4" />
                Bulk Reject
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Approvals List */}
      <div className="space-y-6">
        {approvals.filter(app => app.status === 'pending').map((approval) => (
          <div key={approval.id} className="card glass">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-start gap-4">
                <input
                  type="checkbox"
                  checked={selectedApprovals.includes(approval.id)}
                  onChange={() => toggleSelection(approval.id)}
                  className="mt-1"
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-semibold">{approval.title}</h3>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getPriorityColor(approval.priority)} bg-opacity-20`}>
                      {approval.priority.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-gray-400 mb-3">{approval.description}</p>

                  {/* Risk Assessment */}
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle className="w-4 h-4 text-yellow-500" />
                    <span className="text-sm">Risk Level:</span>
                    <div className={`w-3 h-3 rounded-full ${getRiskColor(approval.risk_level)}`} />
                    <span className="text-sm capitalize">{approval.risk_level}</span>
                  </div>

                  {/* AI Thinking Trace */}
                  <div className="bg-gray-800 p-3 rounded mb-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Eye className="w-4 h-4 text-blue-500" />
                      <span className="text-sm font-medium">AI Analysis</span>
                    </div>
                    <p className="text-sm text-gray-300">{approval.ai_thinking}</p>
                  </div>

                  {/* Provenance Chain */}
                  <div className="mb-3">
                    <span className="text-sm font-medium mb-2 block">Provenance Chain:</span>
                    <div className="flex gap-2">
                      {approval.provenance_chain.map((step, index) => (
                        <React.Fragment key={index}>
                          <span className="px-2 py-1 bg-gray-700 rounded text-xs">{step}</span>
                          {index < approval.provenance_chain.length - 1 && (
                            <span className="text-gray-500">→</span>
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="text-right">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-400">
                    Expires in: {calculateTimeRemaining(approval.expires_at)}
                  </span>
                </div>
                <div className="flex gap-2">
                  <button
                    className="btn-primary flex items-center gap-1"
                    onClick={() => handleApprove(approval.id)}
                  >
                    <CheckCircle className="w-4 h-4" />
                    Approve
                  </button>
                  <button
                    className="btn-secondary flex items-center gap-1"
                    onClick={() => handleReject(approval.id)}
                  >
                    <XCircle className="w-4 h-4" />
                    Reject
                  </button>
                </div>
              </div>
            </div>

            {/* Audit Trail */}
            <div className="border-t border-gray-700 pt-3">
              <span className="text-sm font-medium mb-2 block">Audit Trail:</span>
              <div className="text-xs text-gray-400">
                Created: {new Date().toLocaleString()} | Last reviewed: Never | Actions: 0
              </div>
            </div>
          </div>
        ))}
      </div>

      {approvals.filter(app => app.status === 'pending').length === 0 && (
        <div className="card glass text-center py-12">
          <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">All Caught Up!</h3>
          <p className="text-gray-400">No pending approvals at this time.</p>
        </div>
      )}
    </div>
  );
};

export default ApprovalCenter;