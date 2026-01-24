import React, { useState, useCallback } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  addEdge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Play, Save, Upload, Download, Settings, Zap, Database, MessageSquare } from 'lucide-react';
import yaml from 'js-yaml';

const initialNodes = [
  {
    id: '1',
    type: 'input',
    data: { label: 'Start Workflow' },
    position: { x: 250, y: 25 },
  },
];

const initialEdges = [];

const nodeTypes = {
  llm: ({ data }) => (
    <div className="px-4 py-2 shadow-md rounded-md bg-blue-500 text-white">
      <div className="flex items-center gap-2">
        <MessageSquare className="w-4 h-4" />
        <div>
          <div className="text-sm font-bold">{data.label}</div>
          <div className="text-xs">{data.model}</div>
        </div>
      </div>
    </div>
  ),
  tool: ({ data }) => (
    <div className="px-4 py-2 shadow-md rounded-md bg-green-500 text-white">
      <div className="flex items-center gap-2">
        <Settings className="w-4 h-4" />
        <div>
          <div className="text-sm font-bold">{data.label}</div>
          <div className="text-xs">{data.tool}</div>
        </div>
      </div>
    </div>
  ),
  data: ({ data }) => (
    <div className="px-4 py-2 shadow-md rounded-md bg-purple-500 text-white">
      <div className="flex items-center gap-2">
        <Database className="w-4 h-4" />
        <div>
          <div className="text-sm font-bold">{data.label}</div>
          <div className="text-xs">{data.operation}</div>
        </div>
      </div>
    </div>
  ),
};

const WorkflowStudio = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [workflowName, setWorkflowName] = useState('New Workflow');
  const [isRunning, setIsRunning] = useState(false);
  const [executionLog, setExecutionLog] = useState([]);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const addNode = (type) => {
    const newNode = {
      id: `${nodes.length + 1}`,
      type,
      position: {
        x: Math.random() * 500,
        y: Math.random() * 500,
      },
      data: {
        label: `${type.charAt(0).toUpperCase() + type.slice(1)} Node`,
        ...(type === 'llm' && { model: 'gpt-4' }),
        ...(type === 'tool' && { tool: 'web-search' }),
        ...(type === 'data' && { operation: 'query' }),
      },
    };
    setNodes((nds) => nds.concat(newNode));
  };

  const runWorkflow = async () => {
    setIsRunning(true);
    setExecutionLog([]);

    try {
      // Send workflow to backend for execution
      const response = await fetch('/api/workflows/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: workflowName,
          nodes,
          edges,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      const executionId = result.execution_id;

      // Connect to WebSocket for real-time updates
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/workflows/${executionId}`;
      const ws = new WebSocket(wsUrl);

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.logs) {
          const newLogs = data.logs.map(log => ({
            timestamp: new Date(log.timestamp),
            message: log.message
          }));
          setExecutionLog(newLogs);
        }

        if (data.status === 'completed' || data.status === 'failed') {
          setIsRunning(false);
          ws.close();
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setExecutionLog(prev => [...prev, {
          timestamp: new Date(),
          message: 'Connection error occurred'
        }]);
        setIsRunning(false);
      };

      ws.onclose = () => {
        setIsRunning(false);
      };

    } catch (error) {
      console.error('Failed to execute workflow:', error);
      setExecutionLog(prev => [...prev, {
        timestamp: new Date(),
        message: `Failed to execute workflow: ${error.message}`
      }]);
      setIsRunning(false);
    }
  };

  const exportWorkflow = (format = 'yaml') => {
    const workflow = {
      name: workflowName,
      nodes,
      edges,
      created: new Date().toISOString(),
    };

    let dataStr, mimeType, extension;
    if (format === 'yaml') {
      dataStr = yaml.dump(workflow);
      mimeType = 'application/x-yaml';
      extension = 'yaml';
    } else {
      dataStr = JSON.stringify(workflow, null, 2);
      mimeType = 'application/json';
      extension = 'json';
    }

    const dataUri = `data:${mimeType};charset=utf-8,${encodeURIComponent(dataStr)}`;
    const exportFileDefaultName = `${workflowName.replace(/\s+/g, '_')}.${extension}`;

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const importWorkflow = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          let workflow;
          const content = e.target.result;

          if (file.name.endsWith('.yaml') || file.name.endsWith('.yml')) {
            workflow = yaml.load(content);
          } else {
            workflow = JSON.parse(content);
          }

          setWorkflowName(workflow.name);
          setNodes(workflow.nodes || []);
          setEdges(workflow.edges || []);
        } catch (error) {
          alert('Invalid workflow file: ' + error.message);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="workflow-studio h-full">
      {/* Toolbar */}
      <div className="card glass mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <input
              type="text"
              value={workflowName}
              onChange={(e) => setWorkflowName(e.target.value)}
              className="bg-gray-800 border border-gray-600 rounded px-3 py-1 text-white"
            />
            <button
              className="btn-primary flex items-center gap-2"
              onClick={runWorkflow}
              disabled={isRunning}
            >
              <Play className="w-4 h-4" />
              {isRunning ? 'Running...' : 'Run Workflow'}
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              className="btn-secondary flex items-center gap-2"
              onClick={() => exportWorkflow('yaml')}
            >
              <Download className="w-4 h-4" />
              Export YAML
            </button>
            <button
              className="btn-secondary flex items-center gap-2"
              onClick={() => exportWorkflow('json')}
            >
              <Download className="w-4 h-4" />
              Export JSON
            </button>
            <label className="btn-secondary flex items-center gap-2 cursor-pointer">
              <Upload className="w-4 h-4" />
              Import
              <input
                type="file"
                accept=".json,.yaml,.yml"
                onChange={importWorkflow}
                className="hidden"
              />
            </label>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full">
        {/* Node Palette */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">Node Palette</h3>
          <div className="space-y-3">
            <button
              className="w-full btn-secondary flex items-center gap-2 justify-start"
              onClick={() => addNode('llm')}
            >
              <MessageSquare className="w-4 h-4" />
              LLM Node
            </button>
            <button
              className="w-full btn-secondary flex items-center gap-2 justify-start"
              onClick={() => addNode('tool')}
            >
              <Settings className="w-4 h-4" />
              Tool Node
            </button>
            <button
              className="w-full btn-secondary flex items-center gap-2 justify-start"
              onClick={() => addNode('data')}
            >
              <Database className="w-4 h-4" />
              Data Node
            </button>
          </div>

          <h4 className="text-md font-semibold mt-6 mb-3">Templates</h4>
          <div className="space-y-2">
            <button className="w-full btn-secondary text-left text-sm">
              Content Generation
            </button>
            <button className="w-full btn-secondary text-left text-sm">
              Data Analysis
            </button>
            <button className="w-full btn-secondary text-left text-sm">
              Research Pipeline
            </button>
          </div>
        </div>

        {/* Workflow Canvas */}
        <div className="lg:col-span-2 card glass">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
          >
            <Controls />
            <MiniMap />
            <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
          </ReactFlow>
        </div>

        {/* Execution Panel */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">Execution Log</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {executionLog.length === 0 ? (
              <p className="text-gray-400 text-sm">No execution logs yet. Run the workflow to see results.</p>
            ) : (
              executionLog.map((log, index) => (
                <div key={index} className="flex gap-2 text-sm">
                  <span className="text-gray-500">
                    {log.timestamp.toLocaleTimeString()}
                  </span>
                  <span>{log.message}</span>
                </div>
              ))
            )}
          </div>

          {isRunning && (
            <div className="mt-4 p-3 bg-blue-500 bg-opacity-20 rounded">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-blue-500 animate-pulse" />
                <span className="text-sm">Workflow executing...</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WorkflowStudio;