import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import {
  Play,
  Pause,
  Square,
  Save,
  Upload,
  Download,
  Plus,
  Settings,
  Zap,
  Database,
  Code,
  MessageSquare
} from 'lucide-react'

interface WorkflowStep {
  id: string
  type: 'llm' | 'tool' | 'data' | 'condition'
  name: string
  config: any
  position: { x: number; y: number }
}

interface Workflow {
  id: string
  name: string
  description: string
  steps: WorkflowStep[]
  status: 'draft' | 'running' | 'paused' | 'completed' | 'error'
}

const WorkflowStudio = () => {
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null)

  useEffect(() => {
    fetchWorkflows()
  }, [])

  const fetchWorkflows = async () => {
    try {
      const response = await fetch('/api/workflows')
      if (response.ok) {
        const data = await response.json()
        setWorkflows(data)
      }
    } catch (error) {
      console.error('Failed to fetch workflows:', error)
    } finally {
      setLoading(false)
    }
  }

  const stepTypes = [
    { type: 'llm', icon: MessageSquare, label: 'LLM Call', color: 'bg-blue-500' },
    { type: 'tool', icon: Settings, label: 'Tool Execution', color: 'bg-green-500' },
    { type: 'data', icon: Database, label: 'Data Processing', color: 'bg-purple-500' },
    { type: 'condition', icon: Zap, label: 'Conditional Logic', color: 'bg-orange-500' }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500'
      case 'paused': return 'bg-yellow-500'
      case 'completed': return 'bg-blue-500'
      case 'error': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const handleCreateWorkflow = () => {
    const newWorkflow: Workflow = {
      id: `wf-${Date.now()}`,
      name: 'New Workflow',
      description: 'Description of the new workflow',
      steps: [],
      status: 'draft'
    }
    setWorkflows([...workflows, newWorkflow])
    setSelectedWorkflow(newWorkflow)
    setIsCreating(true)
  }

  const handleWorkflowAction = (workflowId: string, action: 'play' | 'pause' | 'stop') => {
    setWorkflows(workflows.map(wf =>
      wf.id === workflowId
        ? { ...wf, status: action === 'play' ? 'running' : action === 'pause' ? 'paused' : 'completed' }
        : wf
    ))
  }

  const addStepToWorkflow = (stepType: string) => {
    if (!selectedWorkflow) return

    const newStep: WorkflowStep = {
      id: `step-${Date.now()}`,
      type: stepType as any,
      name: `${stepType} Step`,
      config: {},
      position: { x: Math.random() * 400, y: Math.random() * 300 }
    }

    const updatedWorkflow = {
      ...selectedWorkflow,
      steps: [...selectedWorkflow.steps, newStep]
    }

    setSelectedWorkflow(updatedWorkflow)
    setWorkflows(workflows.map(wf =>
      wf.id === selectedWorkflow.id ? updatedWorkflow : wf
    ))
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            Workflow Studio
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Design and manage automated workflows
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline">
            <Upload size={16} className="mr-2" />
            Import YAML
          </Button>
          <Button onClick={handleCreateWorkflow}>
            <Plus size={16} className="mr-2" />
            New Workflow
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Workflow List */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Workflows</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {workflows.map((workflow) => (
                <div
                  key={workflow.id}
                  className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                    selectedWorkflow?.id === workflow.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedWorkflow(workflow)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-sm">{workflow.name}</h4>
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${getStatusColor(workflow.status)}`} />
                      <Badge variant="outline" className="text-xs">
                        {workflow.status}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                    {workflow.description}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">
                      {workflow.steps.length} steps
                    </span>
                    <div className="flex space-x-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleWorkflowAction(workflow.id, 'play')
                        }}
                      >
                        <Play size={12} />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleWorkflowAction(workflow.id, 'pause')
                        }}
                      >
                        <Pause size={12} />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleWorkflowAction(workflow.id, 'stop')
                        }}
                      >
                        <Square size={12} />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Workflow Canvas */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>
                  {selectedWorkflow ? selectedWorkflow.name : 'Select a Workflow'}
                </CardTitle>
                {selectedWorkflow && (
                  <div className="flex space-x-2">
                    <Button size="sm" variant="outline">
                      <Save size={16} className="mr-2" />
                      Save
                    </Button>
                    <Button size="sm" variant="outline">
                      <Download size={16} className="mr-2" />
                      Export
                    </Button>
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {selectedWorkflow ? (
                <div className="space-y-4">
                  {/* Step Palette */}
                  <div className="border-b pb-4">
                    <h4 className="text-sm font-medium mb-3">Add Steps</h4>
                    <div className="flex flex-wrap gap-2">
                      {stepTypes.map((stepType) => (
                        <Button
                          key={stepType.type}
                          size="sm"
                          variant="outline"
                          onClick={() => addStepToWorkflow(stepType.type)}
                          className="flex items-center space-x-2"
                        >
                          <stepType.icon size={14} />
                          <span>{stepType.label}</span>
                        </Button>
                      ))}
                    </div>
                  </div>

                  {/* Canvas Area */}
                  <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg h-96 flex items-center justify-center">
                    {selectedWorkflow.steps.length === 0 ? (
                      <div className="text-center text-gray-500">
                        <Code size={48} className="mx-auto mb-4 opacity-50" />
                        <p>Drag steps here to build your workflow</p>
                        <p className="text-sm mt-1">Or click the buttons above to add steps</p>
                      </div>
                    ) : (
                      <div className="relative w-full h-full">
                        {selectedWorkflow.steps.map((step) => (
                          <div
                            key={step.id}
                            className="absolute bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg p-3 shadow-sm"
                            style={{
                              left: step.position.x,
                              top: step.position.y,
                              minWidth: '120px'
                            }}
                          >
                            <div className="flex items-center space-x-2 mb-2">
                              {stepTypes.find(t => t.type === step.type)?.icon &&
                                React.createElement(stepTypes.find(t => t.type === step.type)!.icon, {
                                  size: 16,
                                  className: "text-gray-600"
                                })
                              }
                              <span className="text-sm font-medium">{step.name}</span>
                            </div>
                            <Badge variant="outline" className="text-xs">
                              {step.type}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Step List */}
                  {selectedWorkflow.steps.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium mb-3">Workflow Steps</h4>
                      <div className="space-y-2">
                        {selectedWorkflow.steps.map((step, index) => (
                          <div key={step.id} className="flex items-center space-x-3 p-2 bg-gray-50 dark:bg-gray-800 rounded">
                            <span className="text-sm font-mono text-gray-500 w-6">{index + 1}</span>
                            {stepTypes.find(t => t.type === step.type)?.icon &&
                              React.createElement(stepTypes.find(t => t.type === step.type)!.icon, {
                                size: 16,
                                className: "text-gray-600"
                              })
                            }
                            <span className="text-sm">{step.name}</span>
                            <Badge variant="outline" className="text-xs ml-auto">
                              {step.type}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-96 flex items-center justify-center text-gray-500">
                  <div className="text-center">
                    <Settings size={48} className="mx-auto mb-4 opacity-50" />
                    <p>Select a workflow to start editing</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default WorkflowStudio