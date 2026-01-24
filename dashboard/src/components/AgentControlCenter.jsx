import React, { useState, useEffect } from 'react';
import { Play, Pause, Square, Plus, Settings, Activity, Cpu, Zap } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { useWebSocket } from '../hooks/useWebSocket';

const AgentControlCenter = () => {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showSpawnDialog, setShowSpawnDialog] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState(null);

  // WebSocket for real-time updates
  const { connected } = useWebSocket('/api/ws/agents', {
    onEvent: (event, data) => {
      if (event === 'agent_spawned' || event === 'agent_destroyed' || event === 'agent_updated') {
        fetchAgents(); // Refresh the list
      }
    }
  });

  useEffect(() => {
    fetchAgents();
  }, []);

  const fetchAgents = async () => {
    try {
      const response = await fetch('/api/agents');
      const data = await response.json();
      // Transform API response to component format
      const transformedAgents = (data.agents || []).map(agent => ({
        id: agent.id,
        name: agent.name,
        type: agent.capabilities.length > 0 ? agent.capabilities[0] : 'general',
        status: agent.status === 'available' ? 'running' : agent.status === 'busy' ? 'running' : 'stopped',
        capabilities: agent.capabilities,
        performance: {
          cpu: 45, // Mock for now - would come from metrics endpoint
          memory: 67,
          throughput: 120,
          uptime: '99.8%'
        },
        utilization: Array.from({ length: 6 }, (_, i) => ({
          time: `${i * 4}:00`,
          value: Math.floor(Math.random() * 100) // Mock data
        }))
      }));
      setAgents(transformedAgents);
    } catch (error) {
      console.error('Failed to fetch agents:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'text-green-500';
      case 'paused': return 'text-yellow-500';
      case 'stopped': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <Play className="w-4 h-4" />;
      case 'paused': return <Pause className="w-4 h-4" />;
      case 'stopped': return <Square className="w-4 h-4" />;
      default: return <Square className="w-4 h-4" />;
    }
  };

  const handleAgentAction = async (agentId, action) => {
    try {
      let status;
      if (action === 'start') status = 'available';
      else if (action === 'pause') status = 'disabled';
      else if (action === 'stop') status = 'disabled';

      const response = await fetch(`/api/agents/${agentId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status })
      });

      if (response.ok) {
        // Update local state
        setAgents(prev => prev.map(agent =>
          agent.id === agentId
            ? {
                ...agent,
                status: action === 'start' ? 'running' :
                       action === 'pause' ? 'paused' :
                       action === 'stop' ? 'stopped' : agent.status
              }
            : agent
        ));
      } else {
        console.error('Failed to update agent status');
      }
    } catch (error) {
      console.error('Error updating agent:', error);
    }
  };

  const spawnNewAgent = async () => {
    try {
      const response = await fetch('/api/agents', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: 'New Agent',
          capabilities: ['basic-processing'],
          config: {}
        })
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Agent spawned:', data);
        setShowSpawnDialog(false);
        fetchAgents(); // Refresh the list
      } else {
        console.error('Failed to spawn agent');
      }
    } catch (error) {
      console.error('Error spawning agent:', error);
    }
  };

  return (
    <div className="agent-control-center">
      {/* Header with Spawn Button */}
      <div className="card glass mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">Agent Control Center</h3>
            <p className="text-sm text-gray-400">Manage and monitor active agents</p>
          </div>
          <button
            className="btn-primary flex items-center gap-2"
            onClick={() => setShowSpawnDialog(true)}
          >
            <Plus className="w-4 h-4" />
            Spawn New Agent
          </button>
        </div>
      </div>

      {/* Agent List */}
      <div className="space-y-6">
        {loading ? (
          <div className="card glass text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-400">Loading agents...</p>
          </div>
        ) : agents.length === 0 ? (
          <div className="card glass text-center py-12">
            <Activity className="w-12 h-12 text-gray-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Agents Running</h3>
            <p className="text-gray-400 mb-4">Spawn your first agent to get started.</p>
            <button
              className="btn-primary"
              onClick={() => setShowSpawnDialog(true)}
            >
              Spawn New Agent
            </button>
          </div>
        ) : (
          agents.map((agent) => (
            <div key={agent.id} className="card glass">
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h4 className="text-lg font-semibold">{agent.name}</h4>
                  <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${getStatusColor(agent.status)} bg-opacity-20`}>
                    {getStatusIcon(agent.status)}
                    {agent.status}
                  </span>
                  <span className="px-2 py-1 bg-gray-700 rounded text-xs capitalize">
                    {agent.type}
                  </span>
                </div>

                {/* Capabilities */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {agent.capabilities.map((cap, index) => (
                    <span key={index} className="px-2 py-1 bg-blue-500 bg-opacity-20 text-blue-400 rounded text-xs">
                      {cap.replace('-', ' ')}
                    </span>
                  ))}
                </div>

                {/* Performance Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-blue-500" />
                    <div>
                      <div className="text-sm font-medium">{agent.performance.cpu}%</div>
                      <div className="text-xs text-gray-400">CPU</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-green-500" />
                    <div>
                      <div className="text-sm font-medium">{agent.performance.memory}%</div>
                      <div className="text-xs text-gray-400">Memory</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-purple-500" />
                    <div>
                      <div className="text-sm font-medium">{agent.performance.throughput}</div>
                      <div className="text-xs text-gray-400">Throughput</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Settings className="w-4 h-4 text-yellow-500" />
                    <div>
                      <div className="text-sm font-medium">{agent.performance.uptime}</div>
                      <div className="text-xs text-gray-400">Uptime</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                {agent.status !== 'running' && (
                  <button
                    className="btn-primary flex items-center gap-1"
                    onClick={() => handleAgentAction(agent.id, 'start')}
                  >
                    <Play className="w-4 h-4" />
                    Start
                  </button>
                )}
                {agent.status === 'running' && (
                  <button
                    className="btn-secondary flex items-center gap-1"
                    onClick={() => handleAgentAction(agent.id, 'pause')}
                  >
                    <Pause className="w-4 h-4" />
                    Pause
                  </button>
                )}
                <button
                  className="btn-secondary flex items-center gap-1"
                  onClick={() => handleAgentAction(agent.id, 'stop')}
                >
                  <Square className="w-4 h-4" />
                  Stop
                </button>
              </div>
            </div>

            {/* Utilization Chart */}
            <div className="border-t border-gray-700 pt-4">
              <h5 className="text-sm font-medium mb-2">Utilization (24h)</h5>
              <ResponsiveContainer width="100%" height={100}>
                <LineChart data={agent.utilization}>
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke={agent.status === 'running' ? '#22c55e' : '#6b7280'}
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          ))
        )}
      </div>

      {/* Overall Metrics */}
      <div className="card glass mt-6">
        <h3 className="text-lg font-semibold mb-4">System Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-500">
              {agents.filter(a => a.status === 'running').length}
            </div>
            <div className="text-sm text-gray-400">Active Agents</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-500">
              {Math.round(agents.reduce((acc, a) => acc + a.performance.cpu, 0) / agents.length)}%
            </div>
            <div className="text-sm text-gray-400">Avg CPU Usage</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-500">
              {agents.reduce((acc, a) => acc + a.performance.throughput, 0)}
            </div>
            <div className="text-sm text-gray-400">Total Throughput</div>
          </div>
        </div>
      </div>

      {/* Spawn Agent Dialog */}
      {showSpawnDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="card glass p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Spawn New Agent</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Agent Name</label>
                <input
                  type="text"
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2"
                  placeholder="Enter agent name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Agent Type</label>
                <select className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2">
                  <option value="creative">Creative</option>
                  <option value="analytical">Analytical</option>
                  <option value="automation">Automation</option>
                  <option value="general">General</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Capabilities</label>
                <div className="flex flex-wrap gap-2">
                  {['text-generation', 'data-processing', 'automation'].map(cap => (
                    <label key={cap} className="flex items-center gap-1">
                      <input type="checkbox" className="rounded" />
                      <span className="text-sm">{cap.replace('-', ' ')}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                className="btn-primary flex-1"
                onClick={spawnNewAgent}
              >
                Spawn Agent
              </button>
              <button
                className="btn-secondary flex-1"
                onClick={() => setShowSpawnDialog(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentControlCenter;