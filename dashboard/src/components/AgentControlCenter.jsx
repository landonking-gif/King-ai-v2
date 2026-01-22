import React, { useState, useEffect } from 'react';
import { Play, Pause, Square, Plus, Settings, Activity, Cpu, Zap } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

const AgentControlCenter = () => {
  const [agents, setAgents] = useState([
    {
      id: 'agent-001',
      name: 'Content Generator',
      type: 'creative',
      status: 'running',
      capabilities: ['text-generation', 'content-creation', 'seo-optimization'],
      performance: {
        cpu: 45,
        memory: 67,
        throughput: 120,
        uptime: '99.8%'
      },
      utilization: [
        { time: '00:00', value: 30 },
        { time: '04:00', value: 25 },
        { time: '08:00', value: 80 },
        { time: '12:00', value: 95 },
        { time: '16:00', value: 85 },
        { time: '20:00', value: 45 }
      ]
    },
    {
      id: 'agent-002',
      name: 'Data Analyzer',
      type: 'analytical',
      status: 'running',
      capabilities: ['data-processing', 'analytics', 'reporting'],
      performance: {
        cpu: 62,
        memory: 78,
        throughput: 85,
        uptime: '99.5%'
      },
      utilization: [
        { time: '00:00', value: 20 },
        { time: '04:00', value: 15 },
        { time: '08:00', value: 70 },
        { time: '12:00', value: 88 },
        { time: '16:00', value: 75 },
        { time: '20:00', value: 35 }
      ]
    },
    {
      id: 'agent-003',
      name: 'Task Automator',
      type: 'automation',
      status: 'paused',
      capabilities: ['workflow-automation', 'task-scheduling', 'integration'],
      performance: {
        cpu: 12,
        memory: 34,
        throughput: 0,
        uptime: '98.2%'
      },
      utilization: [
        { time: '00:00', value: 0 },
        { time: '04:00', value: 0 },
        { time: '08:00', value: 0 },
        { time: '12:00', value: 0 },
        { time: '16:00', value: 0 },
        { time: '20:00', value: 0 }
      ]
    }
  ]);

  const [showSpawnDialog, setShowSpawnDialog] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState(null);

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

  const handleAgentAction = (agentId, action) => {
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
  };

  const spawnNewAgent = () => {
    const newAgent = {
      id: `agent-${Date.now()}`,
      name: 'New Agent',
      type: 'general',
      status: 'running',
      capabilities: ['basic-processing'],
      performance: {
        cpu: 0,
        memory: 0,
        throughput: 0,
        uptime: '100%'
      },
      utilization: Array.from({ length: 6 }, (_, i) => ({
        time: `${i * 4}:00`,
        value: 0
      }))
    };
    setAgents(prev => [...prev, newAgent]);
    setShowSpawnDialog(false);
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
        {agents.map((agent) => (
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
        ))}
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