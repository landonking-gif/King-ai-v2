import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Workflow, Play, CheckCircle, AlertCircle } from 'lucide-react';
import { io } from 'socket.io-client';

const TalkToKingAI = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Hello! I\'m King AI. How can I help you today? You can ask me to create workflows, analyze data, or execute tasks.',
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [socket, setSocket] = useState(null);
  const [currentWorkflow, setCurrentWorkflow] = useState(null);
  const [executionStatus, setExecutionStatus] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initialize WebSocket for real-time updates
    const newSocket = io('http://localhost:8100');
    setSocket(newSocket);

    newSocket.on('chat_response', (data) => {
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        type: data.type || 'text',
        workflow: data.workflow
      }]);
      setIsLoading(false);

      if (data.workflow) {
        setCurrentWorkflow(data.workflow);
      }
    });

    newSocket.on('execution_progress', (data) => {
      setExecutionStatus(data);
    });

    return () => newSocket.close();
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date(),
      type: 'text'
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Send to backend
    try {
      const response = await fetch('/api/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userMessage.content })
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      // WebSocket will handle the response
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'system',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        type: 'error'
      }]);
      setIsLoading(false);
    }
  };

  const executeWorkflow = async () => {
    if (!currentWorkflow) return;

    setExecutionStatus({ status: 'running', message: 'Executing workflow...' });

    try {
      // In real app, call workflow execution API
      setTimeout(() => {
        setExecutionStatus({ status: 'completed', message: 'Workflow executed successfully!' });
      }, 3000);
    } catch (error) {
      setExecutionStatus({ status: 'error', message: 'Workflow execution failed.' });
    }
  };

  const detectIntent = (message) => {
    const lower = message.toLowerCase();
    if (lower.includes('create') || lower.includes('build') || lower.includes('workflow')) {
      return 'workflow_creation';
    }
    if (lower.includes('analyze') || lower.includes('data') || lower.includes('report')) {
      return 'analysis';
    }
    if (lower.includes('execute') || lower.includes('run') || lower.includes('start')) {
      return 'execution';
    }
    return 'general';
  };

  return (
    <div className="talk-to-king-ai h-full flex flex-col">
      {/* Header */}
      <div className="card glass mb-4">
        <div className="flex items-center gap-3">
          <Bot className="w-8 h-8 text-blue-500" />
          <div>
            <h3 className="text-lg font-semibold">Talk to King AI</h3>
            <p className="text-sm text-gray-400">Conversational AI assistant for workflow creation and execution</p>
          </div>
        </div>
      </div>

      <div className="flex-1 flex gap-6">
        {/* Chat Interface */}
        <div className="flex-1 card glass flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.role === 'assistant' && (
                  <Bot className="w-6 h-6 text-blue-500 mt-1 flex-shrink-0" />
                )}
                <div
                  className={`max-w-md p-3 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-blue-500 text-white'
                      : message.type === 'error'
                      ? 'bg-red-500 bg-opacity-20 text-red-400'
                      : 'bg-gray-800 text-gray-200'
                  }`}
                >
                  <p className="text-sm">{message.content}</p>
                  {message.workflow && (
                    <div className="mt-2 p-2 bg-gray-700 rounded text-xs">
                      <Workflow className="w-3 h-3 inline mr-1" />
                      Workflow detected: {message.workflow.name}
                    </div>
                  )}
                  <span className="text-xs opacity-70 mt-1 block">
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                </div>
                {message.role === 'user' && (
                  <User className="w-6 h-6 text-gray-400 mt-1 flex-shrink-0" />
                )}
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3 justify-start">
                <Bot className="w-6 h-6 text-blue-500 mt-1" />
                <div className="bg-gray-800 p-3 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="animate-pulse">King AI is thinking...</div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-700 p-4">
            <div className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="Ask me to create a workflow, analyze data, or execute a task..."
                className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={isLoading || !input.trim()}
                className="btn-primary flex items-center gap-2"
              >
                <Send className="w-4 h-4" />
                Send
              </button>
            </div>
          </div>
        </div>

        {/* Workflow Preview & Execution */}
        <div className="w-80 space-y-4">
          {/* Current Workflow */}
          {currentWorkflow && (
            <div className="card glass">
              <h4 className="font-semibold mb-3 flex items-center gap-2">
                <Workflow className="w-4 h-4" />
                Proposed Workflow
              </h4>
              <div className="space-y-2">
                <div className="text-sm">
                  <strong>Name:</strong> {currentWorkflow.name}
                </div>
                <div className="text-sm">
                  <strong>Steps:</strong> {currentWorkflow.steps?.length || 0}
                </div>
                <div className="text-sm">
                  <strong>Estimated Cost:</strong> ${currentWorkflow.estimatedCost || 'N/A'}
                </div>
              </div>
              <button
                onClick={executeWorkflow}
                className="btn-primary w-full mt-3 flex items-center gap-2 justify-center"
              >
                <Play className="w-4 h-4" />
                Execute Workflow
              </button>
            </div>
          )}

          {/* Execution Status */}
          {executionStatus && (
            <div className="card glass">
              <h4 className="font-semibold mb-3">Execution Status</h4>
              <div className={`p-3 rounded flex items-center gap-2 ${
                executionStatus.status === 'running' ? 'bg-blue-500 bg-opacity-20' :
                executionStatus.status === 'completed' ? 'bg-green-500 bg-opacity-20' :
                'bg-red-500 bg-opacity-20'
              }`}>
                {executionStatus.status === 'running' && <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full" />}
                {executionStatus.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-500" />}
                {executionStatus.status === 'error' && <AlertCircle className="w-4 h-4 text-red-500" />}
                <span className="text-sm">{executionStatus.message}</span>
              </div>
            </div>
          )}

          {/* Quick Actions */}
          <div className="card glass">
            <h4 className="font-semibold mb-3">Quick Actions</h4>
            <div className="space-y-2">
              <button className="btn-secondary w-full text-left text-sm">
                Create content workflow
              </button>
              <button className="btn-secondary w-full text-left text-sm">
                Analyze business data
              </button>
              <button className="btn-secondary w-full text-left text-sm">
                Generate reports
              </button>
              <button className="btn-secondary w-full text-left text-sm">
                Optimize processes
              </button>
            </div>
          </div>

          {/* Intent Detection */}
          <div className="card glass">
            <h4 className="font-semibold mb-3">Detected Intent</h4>
            <div className="text-sm text-gray-400">
              {input ? detectIntent(input).replace('_', ' ') : 'Type a message to detect intent...'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TalkToKingAI;