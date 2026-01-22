
import React, { useState } from 'react';
import './App.css';
import CommandCenter from './components/CommandCenter';
import ApprovalCenter from './components/ApprovalCenter';
import BusinessPLTracker from './components/BusinessPLTracker';
import AgentControlCenter from './components/AgentControlCenter';
import WorkflowStudio from './components/WorkflowStudio';
import TalkToKingAI from './components/TalkToKingAI';

// Dashboard Components (to be implemented)
const MemoryExplorer = () => <div>Memory Explorer</div>;
const ModelHub = () => <div>Model Hub</div>;
const ToolCatalog = () => <div>Tool Catalog</div>;
const SkillManager = () => <div>Skill Manager</div>;
const ProvenanceViewer = () => <div>Provenance Viewer</div>;
const SummaryHub = () => <div>Summary Hub</div>;
const ActivityMonitor = () => <div>Activity Monitor</div>;
const HistoryArchive = () => <div>History Archive</div>;
const SettingsConfig = () => <div>Settings & Configuration</div>;

const dashboards = [
  { id: 'command-center', name: 'Command Center', component: CommandCenter },
  { id: 'workflow-studio', name: 'Workflow Studio', component: WorkflowStudio },
  { id: 'approval-center', name: 'Approval Center', component: ApprovalCenter },
  { id: 'business-pl', name: 'Business P&L Tracker', component: BusinessPLTracker },
  { id: 'agent-control', name: 'Agent Control Center', component: AgentControlCenter },
  { id: 'memory-explorer', name: 'Memory Explorer', component: MemoryExplorer },
  { id: 'model-hub', name: 'Model Hub', component: ModelHub },
  { id: 'tool-catalog', name: 'Tool Catalog', component: ToolCatalog },
  { id: 'skill-manager', name: 'Skill Manager', component: SkillManager },
  { id: 'provenance-viewer', name: 'Provenance Viewer', component: ProvenanceViewer },
  { id: 'talk-to-king', name: 'Talk to King AI', component: TalkToKingAI },
  { id: 'summary-hub', name: 'Summary Hub', component: SummaryHub },
  { id: 'activity-monitor', name: 'Activity Monitor', component: ActivityMonitor },
  { id: 'history-archive', name: 'History Archive', component: HistoryArchive },
  { id: 'settings', name: 'Settings & Configuration', component: SettingsConfig },
];

function App() {
  const [activeDashboard, setActiveDashboard] = useState('command-center');

  const ActiveComponent = dashboards.find(d => d.id === activeDashboard)?.component || CommandCenter;

  return (
    <div className="dashboard-container">
      <aside className="sidebar glass">
        <div className="brand">
          <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>🤴 King AI v3</h1>
          <p style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>Agentic Framework Control Panel</p>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {dashboards.map(dashboard => (
            <button
              key={dashboard.id}
              className={`btn-nav ${activeDashboard === dashboard.id ? 'active' : ''}`}
              onClick={() => setActiveDashboard(dashboard.id)}
            >
              {dashboard.name}
            </button>
          ))}
        </nav>

        <div style={{ marginTop: 'auto' }}>
          <div className="status-badge">
            <span className="dot pulse"></span>
            Framework Online
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header style={{ marginBottom: '40px' }}>
          <h2 style={{ fontSize: '2rem' }}>
            {dashboards.find(d => d.id === activeDashboard)?.name}
          </h2>
          <p style={{ color: 'var(--text-dim)' }}>Master Control Panel</p>
        </header>

        <div className="content-fade-in">
          <ActiveComponent />
        </div>
      </main>
    </div>
  );
}

export default App;
