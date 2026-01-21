import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import {
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  Eye,
  User,
  FileText,
  Calendar,
  Filter
} from 'lucide-react'

interface ApprovalRequest {
  id: string
  type: 'workflow_execution' | 'agent_deployment' | 'data_access' | 'code_deployment'
  title: string
  description: string
  requester: string
  priority: 'low' | 'medium' | 'high' | 'critical'
  status: 'pending' | 'approved' | 'rejected' | 'escalated'
  submittedAt: string
  dueDate?: string
  riskLevel: 'low' | 'medium' | 'high'
  category: string
}

const ApprovalCenter = () => {
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([])
  const [filter, setFilter] = useState<'all' | 'pending' | 'escalated'>('all')
  const [selectedApproval, setSelectedApproval] = useState<ApprovalRequest | null>(null)

  useEffect(() => {
    fetchApprovals()
  }, [])

  const fetchApprovals = async () => {
    try {
      const response = await fetch('/api/approvals')
      if (response.ok) {
        const data = await response.json()
        setApprovals(data)
      }
    } catch (error) {
      console.error('Failed to fetch approvals:', error)
    } finally {
      setLoading(false)
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-500'
      case 'high': return 'bg-orange-500'
      case 'medium': return 'bg-yellow-500'
      case 'low': return 'bg-green-500'
      default: return 'bg-gray-500'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-yellow-500'
      case 'approved': return 'bg-green-500'
      case 'rejected': return 'bg-red-500'
      case 'escalated': return 'bg-purple-500'
      default: return 'bg-gray-500'
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'text-red-600'
      case 'medium': return 'text-orange-600'
      case 'low': return 'text-green-600'
      default: return 'text-gray-600'
    }
  }

  const handleApproval = (id: string, action: 'approve' | 'reject') => {
    setApprovals(approvals.map(approval =>
      approval.id === id
        ? { ...approval, status: action === 'approve' ? 'approved' : 'rejected' }
        : approval
    ))
  }

  const filteredApprovals = approvals.filter(approval => {
    if (filter === 'all') return true
    return approval.status === filter
  })

  const stats = {
    total: approvals.length,
    pending: approvals.filter(a => a.status === 'pending').length,
    escalated: approvals.filter(a => a.status === 'escalated').length,
    approved: approvals.filter(a => a.status === 'approved').length
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            Approval Center
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Review and approve automated actions and deployments
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Filter size={16} className="mr-2" />
            Filters
          </Button>
          <Button variant="outline" size="sm">
            Bulk Actions
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Requests</p>
                <p className="text-2xl font-bold">{stats.total}</p>
              </div>
              <FileText className="h-8 w-8 text-gray-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Pending</p>
                <p className="text-2xl font-bold text-yellow-600">{stats.pending}</p>
              </div>
              <Clock className="h-8 w-8 text-yellow-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Escalated</p>
                <p className="text-2xl font-bold text-purple-600">{stats.escalated}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Approved Today</p>
                <p className="text-2xl font-bold text-green-600">{stats.approved}</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filter Tabs */}
      <div className="flex space-x-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
        {[
          { key: 'all', label: 'All Requests', count: stats.total },
          { key: 'pending', label: 'Pending', count: stats.pending },
          { key: 'escalated', label: 'Escalated', count: stats.escalated }
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setFilter(tab.key as any)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              filter === tab.key
                ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            {tab.label} ({tab.count})
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Approval Requests List */}
        <Card>
          <CardHeader>
            <CardTitle>Approval Requests</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {filteredApprovals.map((approval) => (
                <div
                  key={approval.id}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    selectedApproval?.id === approval.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedApproval(approval)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <h4 className="font-medium text-sm">{approval.title}</h4>
                        <div className={`w-2 h-2 rounded-full ${getPriorityColor(approval.priority)}`} />
                      </div>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                        {approval.description}
                      </p>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <div className="flex items-center space-x-1">
                          <User size={12} />
                          <span>{approval.requester}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Calendar size={12} />
                          <span>{new Date(approval.submittedAt).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-col items-end space-y-2">
                      <Badge variant="outline" className={`text-xs ${getRiskColor(approval.riskLevel)}`}>
                        {approval.riskLevel} risk
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {approval.category}
                      </Badge>
                    </div>
                  </div>

                  <div className="flex items-center justify-between mt-3">
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${getStatusColor(approval.status)}`} />
                      <span className="text-xs capitalize">{approval.status}</span>
                    </div>
                    <div className="flex space-x-2">
                      {approval.status === 'pending' && (
                        <>
                          <Button size="sm" variant="outline" className="text-green-600 border-green-600 hover:bg-green-50">
                            <CheckCircle size={12} className="mr-1" />
                            Approve
                          </Button>
                          <Button size="sm" variant="outline" className="text-red-600 border-red-600 hover:bg-red-50">
                            <XCircle size={12} className="mr-1" />
                            Reject
                          </Button>
                        </>
                      )}
                      <Button size="sm" variant="ghost">
                        <Eye size={12} />
                      </Button>
                    </div>
                  </div>

                  {approval.dueDate && (
                    <div className="mt-2 text-xs text-orange-600">
                      Due: {new Date(approval.dueDate).toLocaleString()}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Approval Details */}
        <Card>
          <CardHeader>
            <CardTitle>
              {selectedApproval ? 'Request Details' : 'Select a Request'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedApproval ? (
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">{selectedApproval.title}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    {selectedApproval.description}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Type</label>
                    <p className="text-sm mt-1">{selectedApproval.type.replace('_', ' ')}</p>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Priority</label>
                    <p className="text-sm mt-1 capitalize">{selectedApproval.priority}</p>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Requester</label>
                    <p className="text-sm mt-1">{selectedApproval.requester}</p>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Risk Level</label>
                    <p className={`text-sm mt-1 capitalize ${getRiskColor(selectedApproval.riskLevel)}`}>
                      {selectedApproval.riskLevel}
                    </p>
                  </div>
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Timeline</label>
                  <div className="mt-2 space-y-1">
                    <p className="text-sm">Submitted: {new Date(selectedApproval.submittedAt).toLocaleString()}</p>
                    {selectedApproval.dueDate && (
                      <p className="text-sm">Due: {new Date(selectedApproval.dueDate).toLocaleString()}</p>
                    )}
                  </div>
                </div>

                {selectedApproval.status === 'pending' && (
                  <div className="flex space-x-2 pt-4 border-t">
                    <Button
                      className="flex-1 bg-green-600 hover:bg-green-700"
                      onClick={() => handleApproval(selectedApproval.id, 'approve')}
                    >
                      <CheckCircle size={16} className="mr-2" />
                      Approve Request
                    </Button>
                    <Button
                      variant="outline"
                      className="flex-1 border-red-600 text-red-600 hover:bg-red-50"
                      onClick={() => handleApproval(selectedApproval.id, 'reject')}
                    >
                      <XCircle size={16} className="mr-2" />
                      Reject Request
                    </Button>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <FileText size={48} className="mx-auto mb-4 opacity-50" />
                  <p>Select an approval request to view details</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default ApprovalCenter