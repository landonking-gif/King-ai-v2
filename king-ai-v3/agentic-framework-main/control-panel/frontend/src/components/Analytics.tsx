import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Activity,
  Users,
  Zap,
  DollarSign,
  Clock,
  Download
} from 'lucide-react'

interface Metric {
  name: string
  value: number
  change: number
  changeType: 'increase' | 'decrease'
  period: string
}

interface ChartData {
  name: string
  value: number
  date: string
}

const Analytics = () => {
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | '90d'>('7d')
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [performanceData, setPerformanceData] = useState<ChartData[]>([])

  useEffect(() => {
    fetchAnalyticsData()
  }, [timeRange])

  const fetchAnalyticsData = async () => {
    try {
      const [metricsRes, chartRes] = await Promise.all([
        fetch('/api/analytics/metrics'),
        fetch('/api/analytics/chart-data')
      ])

      if (metricsRes.ok) {
        const metricsData = await metricsRes.json()
        setMetrics([
          {
            name: 'Total Requests',
            value: metricsData.totalRequests,
            change: 12.5,
            changeType: 'increase',
            period: 'vs last week'
          },
          {
            name: 'Active Agents',
            value: metricsData.activeAgents,
            change: -2.1,
            changeType: 'decrease',
            period: 'vs last week'
          },
          {
            name: 'Avg Response Time',
            value: metricsData.avgResponseTime,
            change: -8.3,
            changeType: 'decrease',
            period: 'vs last week'
          },
          {
            name: 'Success Rate',
            value: metricsData.successRate,
            change: 1.8,
            changeType: 'increase',
            period: 'vs last week'
          }
        ])
      }

      if (chartRes.ok) {
        const chartData = await chartRes.json()
        setPerformanceData(chartData.map((item: any) => ({
          name: new Date(item.date).toLocaleDateString('en-US', { weekday: 'short' }),
          value: item.requests,
          date: item.date
        })))
      }
    } catch (error) {
      console.error('Failed to fetch analytics data:', error)
    } finally {
      setLoading(false)
    }
  }

  const costData = [
    { category: 'LLM Calls', amount: 1250.50, percentage: 45 },
    { category: 'Compute', amount: 890.25, percentage: 32 },
    { category: 'Storage', amount: 345.75, percentage: 12 },
    { category: 'Network', amount: 280.50, percentage: 11 }
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            Analytics Dashboard
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Monitor system performance and business metrics
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="flex bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
            {['24h', '7d', '30d', '90d'].map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range as any)}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  timeRange === range
                    ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
          <Button variant="outline" size="sm">
            <Download size={16} className="mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => (
          <Card key={index}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                    {metric.name}
                  </p>
                  <p className="text-3xl font-bold text-gray-900 dark:text-white mt-1">
                    {typeof metric.value === 'number' && metric.value < 10
                      ? metric.value.toFixed(1)
                      : metric.value.toLocaleString()
                    }
                    {metric.name.includes('Rate') && '%'}
                  </p>
                  <div className="flex items-center mt-2">
                    {metric.changeType === 'increase' ? (
                      <TrendingUp size={16} className="text-green-500 mr-1" />
                    ) : (
                      <TrendingDown size={16} className="text-red-500 mr-1" />
                    )}
                    <span className={`text-sm font-medium ${
                      metric.changeType === 'increase' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {metric.change > 0 ? '+' : ''}{metric.change}%
                    </span>
                    <span className="text-sm text-gray-500 ml-1">{metric.period}</span>
                  </div>
                </div>
                <div className="h-12 w-12 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
                  {index === 0 && <Activity className="h-6 w-6 text-blue-600" />}
                  {index === 1 && <Users className="h-6 w-6 text-blue-600" />}
                  {index === 2 && <Clock className="h-6 w-6 text-blue-600" />}
                  {index === 3 && <Zap className="h-6 w-6 text-blue-600" />}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Request Volume</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="text-center">
                <BarChart3 size={48} className="mx-auto mb-4 text-gray-400" />
                <p className="text-gray-500">Chart visualization would go here</p>
                <p className="text-sm text-gray-400 mt-1">Using recharts library</p>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-7 gap-2">
              {performanceData.map((data, index) => (
                <div key={index} className="text-center">
                  <div className="text-xs text-gray-500">{data.name}</div>
                  <div className="text-sm font-medium">{data.value}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Agent Performance */}
        <Card>
          <CardHeader>
            <CardTitle>Agent Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {agentPerformance.map((agent, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <h4 className="text-sm font-medium">{agent.name}</h4>
                      <Badge variant="outline" className="text-xs">
                        {agent.requests} requests
                      </Badge>
                    </div>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>Success: {agent.success}%</span>
                      <span>Avg Time: {agent.avgTime}s</span>
                    </div>
                  </div>
                  <div className="ml-4">
                    <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-green-500 h-2 rounded-full"
                        style={{ width: `${agent.success}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Cost Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <DollarSign size={20} className="mr-2" />
            Cost Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {costData.map((item, index) => (
              <div key={index} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium">{item.category}</h4>
                  <span className="text-sm text-gray-500">{item.percentage}%</span>
                </div>
                <p className="text-xl font-bold">${item.amount.toFixed(2)}</p>
                <div className="mt-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-medium text-blue-900 dark:text-blue-100">Total Monthly Cost</h4>
                <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">$2,767.00</p>
              </div>
              <div className="text-right">
                <p className="text-sm text-blue-700 dark:text-blue-300">Budget: $3,000</p>
                <p className="text-sm text-green-600">92.2% utilized</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Health Timeline */}
      <Card>
        <CardHeader>
          <CardTitle>System Health Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div className="text-center">
              <Activity size={32} className="mx-auto mb-2 text-gray-400" />
              <p className="text-gray-500 text-sm">System uptime: 99.9%</p>
              <p className="text-gray-400 text-xs">Last 30 days</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default Analytics