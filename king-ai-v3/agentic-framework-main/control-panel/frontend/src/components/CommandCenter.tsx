import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Play, Pause, Square, RefreshCw } from 'lucide-react'

interface SystemStatus {
  agents: number
  workflows: number
  approvals: number
  alerts: number
}

interface Agent {
  id: string
  name: string
  status: 'running' | 'paused' | 'stopped'
  type: string
  lastActivity: string
}

const CommandCenter = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    agents: 0,
    workflows: 0,
    approvals: 0,
    alerts: 0
  })

  const [agents, setAgents] = useState<Agent[]>([])

  useEffect(() => {
    // Fetch system status and agents data
    fetchSystemData()
  }, [])

  const fetchSystemData = async () => {
    try {
      const [statusRes, agentsRes] = await Promise.all([
        fetch('/api/dashboard/status'),
        fetch('/api/dashboard/agents')
      ])

      if (statusRes.ok) {
        const statusData = await statusRes.json()
        setSystemStatus(statusData)
      }

      if (agentsRes.ok) {
        const agentsData = await agentsRes.json()
        setAgents(agentsData)
      }
    } catch (error) {
      console.error('Failed to fetch system data:', error)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500'
      case 'paused': return 'bg-yellow-500'
      case 'stopped': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Play size={16} />
      case 'paused': return <Pause size={16} />
      case 'stopped': return <Square size={16} />
      default: return <RefreshCw size={16} />
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
          Command Center
        </h2>
        <Button onClick={fetchSystemData} variant="outline">
          <RefreshCw size={16} className="mr-2" />
          Refresh
        </Button>
      </div>

      {/* System Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStatus.agents}</div>
            <p className="text-xs text-muted-foreground">
              Currently running
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Workflows</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStatus.workflows}</div>
            <p className="text-xs text-muted-foreground">
              In progress
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Approvals</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStatus.approvals}</div>
            <p className="text-xs text-muted-foreground">
              Require attention
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStatus.alerts}</div>
            <p className="text-xs text-muted-foreground">
              Active alerts
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Agent Status Table */}
      <Card>
        <CardHeader>
          <CardTitle>Agent Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Agent</th>
                  <th className="text-left p-2">Type</th>
                  <th className="text-left p-2">Status</th>
                  <th className="text-left p-2">Last Activity</th>
                  <th className="text-left p-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {agents.map((agent) => (
                  <tr key={agent.id} className="border-b">
                    <td className="p-2">{agent.name}</td>
                    <td className="p-2">
                      <Badge variant="outline">{agent.type}</Badge>
                    </td>
                    <td className="p-2">
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${getStatusColor(agent.status)}`} />
                        <span className="capitalize">{agent.status}</span>
                      </div>
                    </td>
                    <td className="p-2 text-sm text-gray-500">
                      {agent.lastActivity}
                    </td>
                    <td className="p-2">
                      <Button size="sm" variant="outline">
                        {getStatusIcon(agent.status)}
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default CommandCenter