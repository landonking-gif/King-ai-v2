import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Activity, Zap, CheckCircle, AlertTriangle, DollarSign, Users, Cpu, TrendingUp } from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';

const CommandCenter = () => {
  // ... existing state ...

  const [socket, setSocket] = useState(null);

  // WebSocket for real-time updates
  const { connected } = useWebSocket('ws://localhost:8100/ws', {
    onEvent: (eventType, data, timestamp) => {
      handleWebSocketEvent(eventType, data);
    }
  });

  const handleWebSocketEvent = (eventType, data) => {
    switch (eventType) {
      case 'task.completed':
      case 'task.started':
      case 'task.failed':
        // Update activity feed
        const activityDesc = `${data.status}: ${data.task_id}`;
        setActivityFeed(prev => [{
          id: Date.now(),
          type: data.status,
          message: activityDesc,
          timestamp: new Date().toLocaleTimeString()
        }, ...prev.slice(0, 9)]);
        break;
      case 'approval.required':
        setKpis(prev => ({ ...prev, pendingApprovals: (prev.pendingApprovals || 0) + 1 }));
        break;
      case 'system.alert':
        // Update system health based on alerts
        if (data.level === 'error') {
          setSystemHealth(prev => ({
            ...prev,
            orchestrator: { ...prev.orchestrator, status: 'unhealthy' }
          }));
        }
        break;
      case 'analytics.kpi_update':
        setKpis(prev => ({ ...prev, ...data.kpis }));
        break;
      default:
        break;
    }
  };

  useEffect(() => {
    // Initialize WebSocket connection is now handled by useWebSocket hook
    setSocket({ connected }); // For backward compatibility
  }, [connected]);

  // ... rest of component ...
  const [systemHealth, setSystemHealth] = useState({
    orchestrator: { status: 'healthy', port: 8000 },
    subagent_manager: { status: 'healthy', port: 8001 },
    memory_service: { status: 'healthy', port: 8002 },
    mcp_gateway: { status: 'healthy', port: 8080 },
    code_executor: { status: 'healthy', port: 8004 }
  });

  const [kpis, setKpis] = useState({
    activeWorkflows: 5,
    pendingApprovals: 3,
    tokenUsage: 125000
  });

  const [workflowThroughput, setWorkflowThroughput] = useState([
    { hour: '00', completed: 5 },
    { hour: '01', completed: 3 },
    { hour: '02', completed: 7 },
    { hour: '03', completed: 4 },
    { hour: '04', completed: 8 },
    { hour: '05', completed: 6 }
  ]);

  const [plSummary, setPlSummary] = useState({
    totalRevenue: 10000,
    totalExpenses: 7000,
    netProfit: 3000,
    marginPercent: 30
  });

  const [agentUtilization, setAgentUtilization] = useState([
    { agent: 'orchestrator', utilization: 85 },
    { agent: 'subagent', utilization: 70 },
    { agent: 'memory', utilization: 60 },
    { agent: 'mcp', utilization: 45 },
    { agent: 'executor', utilization: 55 }
  ]);

  const [modelCosts, setModelCosts] = useState([
    { model: 'gpt-4', cost: 50.0 },
    { model: 'claude-3', cost: 40.0 },
    { model: 'llama-3', cost: 10.0 }
  ]);

  const [activityFeed, setActivityFeed] = useState([
    { type: 'workflow', message: 'Workflow completed successfully', timestamp: new Date().toISOString() },
    { type: 'approval', message: 'New approval request pending', timestamp: new Date().toISOString() }
  ]);

  useEffect(() => {
    // Fetch data from backend
    const fetchData = async () => {
      try {
        const [healthRes, overviewRes] = await Promise.all([
          fetch('/api/dashboard/health'),
          fetch('/api/dashboard/overview')
        ]);

        if (healthRes.ok) {
          setSystemHealth(await healthRes.json());
        }

        if (overviewRes.ok) {
          const data = await overviewRes.json();
          setKpis({
            activeWorkflows: data.active_workflows || 0,
            pendingApprovals: data.pending_approvals || 0,
            tokenUsage: data.total_tokens || 0
          });
        }
      } catch (err) {
        console.error('Failed to fetch command center data:', err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Update every 5s
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    return status === 'healthy' ? '#10b981' : '#ef4444';
  };

  return (
    <div className="command-center">
      {/* System Health Bar */}
      <div className="card glass mb-6">
        <h3 className="text-lg font-semibold mb-4">System Health</h3>
        <div className="grid grid-cols-5 gap-4">
          {Object.entries(systemHealth).map(([service, info]) => (
            <div key={service} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: getStatusColor(info.status) }}
              />
              <div>
                <div className="text-sm font-medium capitalize">{service.replace('_', ' ')}</div>
                <div className="text-xs text-gray-400">:{info.port}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="card glass">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-500" />
            <div>
              <div className="text-2xl font-bold">{kpis.activeWorkflows}</div>
              <div className="text-sm text-gray-400">Active Workflows</div>
            </div>
          </div>
        </div>

        <div className="card glass">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-8 h-8 text-yellow-500" />
            <div>
              <div className="text-2xl font-bold">{kpis.pendingApprovals}</div>
              <div className="text-sm text-gray-400">Pending Approvals</div>
            </div>
          </div>
        </div>

        <div className="card glass">
          <div className="flex items-center gap-3">
            <Zap className="w-8 h-8 text-purple-500" />
            <div>
              <div className="text-2xl font-bold">{kpis.tokenUsage.toLocaleString()}</div>
              <div className="text-sm text-gray-400">Token Usage</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Workflow Throughput Chart */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">Workflow Throughput (24h)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={workflowThroughput}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="completed" stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* P&L Summary */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">P&L Summary</h3>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span>Total Revenue</span>
              <span className="text-green-500">${plSummary.totalRevenue.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span>Total Expenses</span>
              <span className="text-red-500">${plSummary.totalExpenses.toLocaleString()}</span>
            </div>
            <div className="flex justify-between font-semibold">
              <span>Net Profit</span>
              <span className={plSummary.netProfit >= 0 ? 'text-green-500' : 'text-red-500'}>
                ${plSummary.netProfit.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Margin %</span>
              <span>{plSummary.marginPercent}%</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Agent Utilization */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">Agent Utilization</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={agentUtilization}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="agent" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="utilization" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model Costs */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">Model Costs</h3>
          <div className="space-y-3">
            {modelCosts.map((model) => (
              <div key={model.model} className="flex justify-between items-center">
                <span>{model.model}</span>
                <span className="text-red-500">${model.cost.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Live Activity Feed */}
      <div className="card glass">
        <h3 className="text-lg font-semibold mb-4">Live Activity Feed</h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {activityFeed.map((activity, index) => (
            <div key={index} className="flex items-center gap-3 p-2 bg-gray-800 rounded">
              <Activity className="w-4 h-4 text-blue-500" />
              <span className="text-sm">{activity.message}</span>
              <span className="text-xs text-gray-400 ml-auto">
                {new Date(activity.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card glass mt-6">
        <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button className="btn-primary">New Workflow</button>
          <button className="btn-secondary">View Approvals</button>
          <button className="btn-secondary">System Health</button>
          <button className="btn-secondary">Agent Control</button>
        </div>
      </div>
    </div>
  );
};

export default CommandCenter;