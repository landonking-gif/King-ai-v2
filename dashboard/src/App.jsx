
import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = `http://${window.location.hostname}:8000/api`;

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

  // Fetch Empire Stats and Businesses
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [bizRes, healthRes] = await Promise.all([
          fetch(`${API_BASE}/businesses/`),
          fetch(`${API_BASE}/health`)
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
      } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000); // Polling every 10s
    return () => clearInterval(interval);
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
            className={`btn-nav ${activeTab === 'ceo' ? 'active' : ''}`}
            onClick={() => setActiveTab('ceo')}
          >
            🗣️ Talk to CEO
          </button>
          <button
            className={`btn-nav ${activeTab === 'approvals' ? 'active' : ''}`}
            onClick={() => setActiveTab('approvals')}
          >
            ⚖️ Approvals
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
            {activeTab === 'ceo' && 'CEO Briefing'}
            {activeTab === 'approvals' && 'Approval Queue'}
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
      </main>
    </div>
  );
}

export default App;
