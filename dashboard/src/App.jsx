
import React, { useState, useEffect } from 'react';
import './App.css';
import { PLDashboard } from './components/Charts/PLDashboard';
import { CircuitBreakerDashboard } from './components/Monitoring/CircuitBreakerDashboard';

// Dynamic API base to handle port 80 (proxy) or port 8000 (direct)
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? `http://${window.location.hostname}:8000/api`
  : `http://${window.location.hostname}/api`;

function App() {
  const [activeTab, setActiveTab] = useState('empire');
  const [stats, setStats] = useState({
    total_revenue: 0,
    total_profit: 0,
    active_units: 0
  });
  const [businesses, setBusinesses] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [evolutions, setEvolutions] = useState([]);
  const [approvals, setApprovals] = useState([]);
  const [schedulerStatus, setSchedulerStatus] = useState(null);
  const [plData, setPlData] = useState(null);
  const [plPeriod, setPlPeriod] = useState('30d');

  // Fetch Empire Stats and Businesses
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [bizRes, healthRes, evoRes, approvalRes, schedRes, plRes] = await Promise.all([
          fetch(`${API_BASE}/businesses/`),
          fetch(`${API_BASE}/health`),
          fetch(`${API_BASE}/evolution/proposals`).catch(() => ({ ok: false })),
          fetch(`${API_BASE}/approvals/pending`).catch(() => ({ ok: false })),
          fetch(`${API_BASE}/scheduler/status`).catch(() => ({ ok: false })),
          fetch(`${API_BASE}/analytics/pl?period=${plPeriod}`).catch(() => ({ ok: false }))
        ]);

        if (bizRes.ok) {
          const data = await bizRes.json();
          setBusinesses(data);

          // Calculate stats locally from business data
          const rev = data.reduce((acc, b) => acc + (b.total_revenue || 0), 0);
          const exp = data.reduce((acc, b) => acc + (b.total_expenses || 0), 0);
          setStats({
            total_revenue: rev,
            total_profit: rev - exp,
            active_units: data.filter(b => b.status !== 'sunset').length
          });
        }

        if (evoRes.ok) {
          const evoData = await evoRes.json();
          setEvolutions(Array.isArray(evoData) ? evoData : []);
        }

        if (approvalRes.ok) {
          const approvalData = await approvalRes.json();
          // Handle both array format and {requests: [...]} format
          if (Array.isArray(approvalData)) {
            setApprovals(approvalData);
          } else if (approvalData.requests) {
            setApprovals(approvalData.requests);
          } else {
            setApprovals([]);
          }
        }

        if (schedRes.ok) {
          const schedData = await schedRes.json();
          setSchedulerStatus(schedData);
        }

        if (plRes.ok) {
          const plData = await plRes.json();
          setPlData(plData);
        }
      } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000); // Polling every 10s
    return () => clearInterval(interval);
  }, [plPeriod]);

  // Fetch Chat History on mount
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_BASE}/chat/history`);
        if (res.ok) {
          const history = await res.json();
          setMessages(history);
        }
      } catch (err) {
        console.error("Failed to fetch chat history:", err);
      }
    };
    fetchHistory();
  }, []);

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;

    const userMsg = { role: 'user', content: chatInput };
    setMessages(prev => [...prev, userMsg]);
    setChatInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: chatInput })
      });

      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'system', content: '⚠️ Connection to King AI failed.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Evolution handlers
  const handleApproveEvolution = async (id) => {
    try {
      await fetch(`${API_BASE}/evolution/approve/${id}`, { method: 'POST' });
      const evoRes = await fetch(`${API_BASE}/evolution/proposals`);
      if (evoRes.ok) setEvolutions(await evoRes.json());
    } catch (err) {
      console.error('Failed to approve evolution:', err);
    }
  };

  const handleRejectEvolution = async (id) => {
    try {
      await fetch(`${API_BASE}/evolution/reject/${id}`, { method: 'POST' });
      const evoRes = await fetch(`${API_BASE}/evolution/proposals`);
      if (evoRes.ok) setEvolutions(await evoRes.json());
    } catch (err) {
      console.error('Failed to reject evolution:', err);
    }
  };

  // Approval handlers
  const handleApprove = async (id) => {
    try {
      await fetch(`${API_BASE}/approvals/${id}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'dashboard_user' })
      });
      const approvalRes = await fetch(`${API_BASE}/approvals/pending`);
      if (approvalRes.ok) {
        const data = await approvalRes.json();
        setApprovals(Array.isArray(data) ? data : (data.requests || []));
      }
    } catch (err) {
      console.error('Failed to approve request:', err);
    }
  };

  const handleReject = async (id) => {
    try {
      await fetch(`${API_BASE}/approvals/${id}/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'dashboard_user', reason: 'Rejected via dashboard' })
      });
      const approvalRes = await fetch(`${API_BASE}/approvals/pending`);
      if (approvalRes.ok) {
        const data = await approvalRes.json();
        setApprovals(Array.isArray(data) ? data : (data.requests || []));
      }
    } catch (err) {
      console.error('Failed to reject request:', err);
    }
  };

  return (
    <div className="dashboard-container">
      <aside className="sidebar glass">
        <div className="brand">
          <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>🤴 King AI</h1>
          <p style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>Autonomous Empire v2</p>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button
            className={`btn-nav ${activeTab === 'empire' ? 'active' : ''}`}
            onClick={() => setActiveTab('empire')}
          >
            🏰 My Empire
          </button>
          <button
            className={`btn-nav ${activeTab === 'pl' ? 'active' : ''}`}
            onClick={() => setActiveTab('pl')}
          >
            📊 P&L Dashboard
          </button>
          <button
            className={`btn-nav ${activeTab === 'ceo' ? 'active' : ''}`}
            onClick={() => setActiveTab('ceo')}
          >
            🗣️ Talk to CEO
          </button>
          <button
            className={`btn-nav ${activeTab === 'approvals' ? 'active' : ''}`}
            onClick={() => setActiveTab('approvals')}
          >
            ⚖️ Approvals {approvals.length > 0 && <span className="badge">{approvals.length}</span>}
          </button>
          <button
            className={`btn-nav ${activeTab === 'evolution' ? 'active' : ''}`}
            onClick={() => setActiveTab('evolution')}
          >
            🧬 Evolution {evolutions.filter(e => e.status === 'pending').length > 0 &&
              <span className="badge">{evolutions.filter(e => e.status === 'pending').length}</span>}
          </button>
          <button
            className={`btn-nav ${activeTab === 'system' ? 'active' : ''}`}
            onClick={() => setActiveTab('system')}
          >
            🔧 System Health
          </button>
          <button
            className={`btn-nav ${activeTab === 'scheduler' ? 'active' : ''}`}
            onClick={() => setActiveTab('scheduler')}
          >
            ⏰ Scheduler
          </button>
        </nav>

        <div style={{ marginTop: 'auto' }}>
          <div className="status-badge">
            <span className="dot pulse"></span>
            System Online
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header style={{ marginBottom: '40px' }}>
          <h2 style={{ fontSize: '2rem' }}>
            {activeTab === 'empire' && 'Empire Overview'}
            {activeTab === 'pl' && 'Profit & Loss Dashboard'}
            {activeTab === 'ceo' && 'CEO Briefing'}
            {activeTab === 'approvals' && 'Approval Queue'}
            {activeTab === 'evolution' && 'Evolution Proposals'}
            {activeTab === 'system' && 'System Health & Circuit Breakers'}
            {activeTab === 'scheduler' && 'Autonomous Scheduler'}
          </h2>
          <p style={{ color: 'var(--text-dim)' }}>Welcome back, Your Majesty.</p>
        </header>

        {activeTab === 'empire' && (
          <div className="content-fade-in">
            <div className="stat-grid">
              <div className="card glass">
                <h3 style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '8px' }}>Total Revenue</h3>
                <p className="stat-value">${stats.total_revenue.toLocaleString()}</p>
              </div>
              <div className="card glass">
                <h3 style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '8px' }}>Total Profit</h3>
                <p className="stat-value" style={{ color: stats.total_profit >= 0 ? '#10b981' : '#ef4444' }}>
                  ${stats.total_profit.toLocaleString()}
                </p>
              </div>
              <div className="card glass">
                <h3 style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '8px' }}>Active Units</h3>
                <p className="stat-value">{stats.active_units}</p>
              </div>
            </div>

            <div className="card glass" style={{ marginTop: '32px' }}>
              <h3 style={{ marginBottom: '20px' }}>Managed Businesses</h3>
              {businesses.length === 0 ? (
                <p style={{ color: 'var(--text-dim)' }}>No businesses found. Deploy a unit to begin.</p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {businesses.map(b => (
                    <div key={b.id} className="glass" style={{ padding: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <strong>{b.name}</strong>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>{b.type} • {b.status}</div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ color: '#10b981' }}>+${(b.total_revenue || 0).toLocaleString()}</div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Revenue</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'pl' && (
          <div className="content-fade-in">
            <PLDashboard
              data={plData}
              period={plPeriod}
              onPeriodChange={setPlPeriod}
              showComparison={true}
            />
          </div>
        )}

        {activeTab === 'system' && (
          <div className="content-fade-in">
            <CircuitBreakerDashboard
              apiBaseUrl={`${API_BASE}/v1/system`}
              refreshInterval={5000}
            />
          </div>
        )}

        {activeTab === 'ceo' && (
          <div className="card glass" style={{ height: 'calc(100% - 140px)', display: 'flex', flexDirection: 'column', padding: '0' }}>
            <div style={{ flex: 1, padding: '24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {messages.length === 0 && (
                <p style={{ color: 'var(--text-dim)', textAlign: 'center', marginTop: '40px' }}>
                  Start a conversation with your Master AI Brain.
                </p>
              )}
              {messages.map((msg, i) => (
                <div key={i} style={{
                  alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  backgroundColor: msg.role === 'user' ? 'var(--primary)' : 'rgba(255,255,255,0.05)',
                  padding: '12px 18px',
                  borderRadius: '16px',
                  maxWidth: '80%',
                  border: msg.role === 'user' ? 'none' : '1px solid var(--glass-border)'
                }}>
                  {msg.content}
                </div>
              ))}
              {isLoading && <div style={{ color: 'var(--text-dim)', fontStyle: 'italic' }}>King AI is thinking...</div>}
            </div>
            <div style={{ padding: '20px', borderTop: '1px solid var(--glass-border)', display: 'flex', gap: '12px' }}>
              <input
                type="text"
                placeholder="Ask about your empire or issue a command..."
                className="glass"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                style={{ flex: 1, padding: '14px', border: '1px solid var(--glass-border)', outline: 'none', color: 'white' }}
              />
              <button
                className="btn-primary"
                onClick={handleSendMessage}
                disabled={isLoading}
              >
                Send
              </button>
            </div>
          </div>
        )}

        {activeTab === 'approvals' && (
          <div className="content-fade-in">
            <div className="card glass">
              <h3 style={{ marginBottom: '20px' }}>Pending Approvals</h3>
              {approvals.length === 0 ? (
                <p style={{ color: 'var(--text-dim)' }}>No pending approvals. The empire runs smoothly.</p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {approvals.map(a => (
                    <div key={a.id} className="glass" style={{ padding: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                          <strong>{a.title}</strong>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                            {a.action_type} • Risk: {a.risk_level}
                          </div>
                          <p style={{ marginTop: '8px', fontSize: '0.9rem' }}>{a.description}</p>
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <button
                            className="btn-primary"
                            onClick={() => handleApprove(a.id)}
                            style={{ padding: '8px 16px', fontSize: '0.8rem' }}
                          >
                            ✓ Approve
                          </button>
                          <button
                            className="btn-secondary"
                            onClick={() => handleReject(a.id)}
                            style={{ padding: '8px 16px', fontSize: '0.8rem', background: 'rgba(239, 68, 68, 0.2)', border: '1px solid rgba(239, 68, 68, 0.5)' }}
                          >
                            ✗ Reject
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'evolution' && (
          <div className="content-fade-in">
            <div className="card glass">
              <h3 style={{ marginBottom: '20px' }}>🧬 Self-Modification Proposals</h3>
              <p style={{ color: 'var(--text-dim)', marginBottom: '20px', fontSize: '0.9rem' }}>
                The Master AI proposes improvements to its own code. Review and approve changes before they are applied.
              </p>
              {evolutions.length === 0 ? (
                <p style={{ color: 'var(--text-dim)' }}>No evolution proposals pending. The system is optimal.</p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {evolutions.map(evo => (
                    <div key={evo.id} className="glass" style={{ padding: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <strong>{evo.description?.substring(0, 80) || 'System Improvement'}</strong>
                            <span style={{
                              padding: '2px 8px',
                              borderRadius: '12px',
                              fontSize: '0.7rem',
                              background: evo.status === 'pending' ? 'rgba(251, 191, 36, 0.2)' :
                                evo.status === 'approved' ? 'rgba(16, 185, 129, 0.2)' :
                                  'rgba(239, 68, 68, 0.2)',
                              color: evo.status === 'pending' ? '#fbbf24' :
                                evo.status === 'approved' ? '#10b981' : '#ef4444'
                            }}>
                              {evo.status?.toUpperCase() || 'PENDING'}
                            </span>
                          </div>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                            Type: {evo.type} • Confidence: {((evo.confidence_score || 0) * 100).toFixed(1)}%
                          </div>
                          <p style={{ marginTop: '8px', fontSize: '0.9rem', color: 'var(--text-dim)' }}>
                            {evo.rationale || evo.expected_impact || 'No rationale provided'}
                          </p>
                        </div>
                        <div style={{ display: 'flex', gap: '8px', marginLeft: '16px' }}>
                          {(evo.status === 'pending' || evo.status === 'PENDING') && (
                            <>
                              <button
                                className="btn-primary"
                                onClick={() => handleApproveEvolution(evo.id)}
                                style={{ padding: '8px 16px', fontSize: '0.8rem' }}
                              >
                                ✓ Approve
                              </button>
                              <button
                                className="btn-secondary"
                                onClick={() => handleRejectEvolution(evo.id)}
                                style={{ padding: '8px 16px', fontSize: '0.8rem', background: 'rgba(239, 68, 68, 0.2)', border: '1px solid rgba(239, 68, 68, 0.5)' }}
                              >
                                ✗ Reject
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'scheduler' && (
          <div className="content-fade-in">
            <div className="stat-grid">
              <div className="card glass">
                <h3 style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '8px' }}>Scheduler Status</h3>
                <p className="stat-value" style={{ color: schedulerStatus?.running ? '#10b981' : '#ef4444' }}>
                  {schedulerStatus?.running ? '● Running' : '○ Stopped'}
                </p>
              </div>
              <div className="card glass">
                <h3 style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '8px' }}>Active Tasks</h3>
                <p className="stat-value">{schedulerStatus?.active_tasks || 0}</p>
              </div>
              <div className="card glass">
                <h3 style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '8px' }}>Total Executions</h3>
                <p className="stat-value">{schedulerStatus?.total_executions || 0}</p>
              </div>
            </div>

            <div className="card glass" style={{ marginTop: '32px' }}>
              <h3 style={{ marginBottom: '20px' }}>Scheduled Tasks</h3>
              {!schedulerStatus?.tasks || schedulerStatus.tasks.length === 0 ? (
                <p style={{ color: 'var(--text-dim)' }}>No scheduled tasks registered.</p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {schedulerStatus.tasks.map(task => (
                    <div key={task.id} className="glass" style={{ padding: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <strong>{task.name}</strong>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                          {task.frequency} • Runs: {task.run_count} • Avg: {task.avg_duration_ms}ms
                        </div>
                        {task.last_run && (
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                            Last: {new Date(task.last_run).toLocaleString()}
                          </div>
                        )}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <span style={{
                          padding: '4px 12px',
                          borderRadius: '12px',
                          fontSize: '0.8rem',
                          background: task.enabled ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                          color: task.enabled ? '#10b981' : '#ef4444'
                        }}>
                          {task.status}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
