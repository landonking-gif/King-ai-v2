import React, { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

interface Agent {
  id: string;
  name: string;
  status: 'running' | 'stopped' | 'paused' | 'error';
  type: string;
  capability?: string;
  utilization: number;
  last_activity: string;
  created_at: string;
  memory_usage?: number;
  cpu_usage?: number;
}

interface UtilizationMetric {
  timestamp: string;
  cpu: number;
  memory: number;
}

export const AgentControlCenter: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [metrics, setMetrics] = useState<UtilizationMetric[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSpawnModal, setShowSpawnModal] = useState(false);
  const [spawnFormData, setSpawnFormData] = useState({
    name: '',
    type: 'generic',
    capability: '',
  });

  useEffect(() => {
    fetchAgents();
    const interval = setInterval(fetchAgents, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchAgents = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/subagent-manager/agents', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setAgents(data.agents || []);
        setError(null);
      } else if (response.status === 401) {
        setError('Unauthorized - please login');
      } else {
        setError('Failed to fetch agents');
      }
    } catch (err) {
      setError('Connection error: ' + String(err));
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'running':
        return 'bg-green-100 text-green-800';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800';
      case 'stopped':
        return 'bg-gray-100 text-gray-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const handleSpawnAgent = async () => {
    if (!spawnFormData.name.trim()) {
      setError('Agent name is required');
      return;
    }

    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/subagent-manager/agents', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(spawnFormData),
      });

      if (response.ok) {
        setShowSpawnModal(false);
        setSpawnFormData({ name: '', type: 'generic', capability: '' });
        fetchAgents();
      } else {
        setError('Failed to spawn agent');
      }
    } catch (err) {
      setError('Error spawning agent: ' + String(err));
    }
  };

  const handleAgentControl = async (agentId: string, action: 'pause' | 'resume' | 'restart' | 'destroy') => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/subagent-manager/agents/${agentId}/${action}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        fetchAgents();
      } else {
        setError(`Failed to ${action} agent`);
      }
    } catch (err) {
      setError(`Error performing ${action}: ` + String(err));
    }
  };

  const UtilizationChart = ({ agent }: { agent: Agent }) => (
    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm font-medium text-gray-600">CPU Usage</div>
          <div className="mt-2 bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full"
              style={{ width: `${agent.cpu_usage || 0}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 mt-1">{agent.cpu_usage || 0}%</div>
        </div>
        <div>
          <div className="text-sm font-medium text-gray-600">Memory Usage</div>
          <div className="mt-2 bg-gray-200 rounded-full h-2">
            <div
              className="bg-purple-600 h-2 rounded-full"
              style={{ width: `${agent.memory_usage || 0}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 mt-1">{agent.memory_usage || 0}%</div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="p-6 bg-white">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Agent Control Center</h1>
          <p className="text-gray-600 mt-2">Manage and monitor active agents</p>
        </div>
        <Button onClick={() => setShowSpawnModal(true)}>Spawn New Agent</Button>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {loading && (
        <div className="text-center p-8">
          <p className="text-gray-500">Loading agents...</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Agents List */}
        <div className="lg:col-span-2">
          <Card className="p-4">
            <h2 className="text-lg font-semibold mb-4">Active Agents ({agents.length})</h2>
            {agents.length === 0 ? (
              <div className="text-center p-8 text-gray-500">
                No agents currently running
              </div>
            ) : (
              <div className="space-y-3">
                {agents.map((agent) => (
                  <div
                    key={agent.id}
                    onClick={() => setSelectedAgent(selectedAgent?.id === agent.id ? null : agent)}
                    className="p-4 border rounded-lg hover:bg-gray-50 cursor-pointer transition"
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <h3 className="font-medium text-gray-900">{agent.name}</h3>
                          <Badge className={getStatusColor(agent.status)}>
                            {agent.status.toUpperCase()}
                          </Badge>
                        </div>
                        <div className="mt-2 text-sm text-gray-600">
                          <p>Type: {agent.type}</p>
                          <p>Capability: {agent.capability || 'General'}</p>
                          <p>Created: {new Date(agent.created_at).toLocaleString()}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm">
                          <p className="text-gray-600">Utilization</p>
                          <p className="text-lg font-semibold text-gray-900">
                            {agent.utilization}%
                          </p>
                        </div>
                      </div>
                    </div>

                    {selectedAgent?.id === agent.id && (
                      <>
                        <UtilizationChart agent={agent} />
                        <div className="mt-4 flex gap-2">
                          {agent.status === 'running' && (
                            <>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleAgentControl(agent.id, 'pause');
                                }}
                              >
                                Pause
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleAgentControl(agent.id, 'restart');
                                }}
                              >
                                Restart
                              </Button>
                            </>
                          )}
                          {agent.status === 'paused' && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleAgentControl(agent.id, 'resume');
                              }}
                            >
                              Resume
                            </Button>
                          )}
                          <Button
                            size="sm"
                            variant="outline"
                            className="text-red-600 hover:bg-red-50"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleAgentControl(agent.id, 'destroy');
                            }}
                          >
                            Destroy
                          </Button>
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* Statistics Panel */}
        <div>
          <Card className="p-4">
            <h2 className="text-lg font-semibold mb-4">Statistics</h2>
            <div className="space-y-4">
              <div className="p-3 bg-blue-50 rounded-lg">
                <p className="text-sm text-gray-600">Total Agents</p>
                <p className="text-2xl font-bold text-blue-600">{agents.length}</p>
              </div>
              <div className="p-3 bg-green-50 rounded-lg">
                <p className="text-sm text-gray-600">Running</p>
                <p className="text-2xl font-bold text-green-600">
                  {agents.filter((a) => a.status === 'running').length}
                </p>
              </div>
              <div className="p-3 bg-yellow-50 rounded-lg">
                <p className="text-sm text-gray-600">Paused</p>
                <p className="text-2xl font-bold text-yellow-600">
                  {agents.filter((a) => a.status === 'paused').length}
                </p>
              </div>
              <div className="p-3 bg-red-50 rounded-lg">
                <p className="text-sm text-gray-600">Error</p>
                <p className="text-2xl font-bold text-red-600">
                  {agents.filter((a) => a.status === 'error').length}
                </p>
              </div>
              <div className="p-3 bg-purple-50 rounded-lg">
                <p className="text-sm text-gray-600">Avg Utilization</p>
                <p className="text-2xl font-bold text-purple-600">
                  {agents.length > 0
                    ? Math.round(
                        agents.reduce((sum, a) => sum + a.utilization, 0) /
                          agents.length
                      )
                    : 0}
                  %
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Spawn Modal */}
      {showSpawnModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md p-6">
            <h2 className="text-xl font-bold mb-4">Spawn New Agent</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Agent Name
                </label>
                <input
                  type="text"
                  value={spawnFormData.name}
                  onChange={(e) =>
                    setSpawnFormData({ ...spawnFormData, name: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Research Agent"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Agent Type
                </label>
                <select
                  value={spawnFormData.type}
                  onChange={(e) =>
                    setSpawnFormData({ ...spawnFormData, type: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="generic">Generic</option>
                  <option value="analyst">Analyst</option>
                  <option value="developer">Developer</option>
                  <option value="researcher">Researcher</option>
                  <option value="executor">Executor</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Capability
                </label>
                <input
                  type="text"
                  value={spawnFormData.capability}
                  onChange={(e) =>
                    setSpawnFormData({ ...spawnFormData, capability: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Market Analysis"
                />
              </div>
              <div className="flex gap-3 pt-4">
                <Button
                  variant="outline"
                  onClick={() => setShowSpawnModal(false)}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSpawnAgent}
                  className="flex-1 bg-blue-600 text-white hover:bg-blue-700"
                >
                  Spawn Agent
                </Button>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default AgentControlCenter;
