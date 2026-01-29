import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Workflow, Play, CheckCircle, AlertCircle, Users, Activity, Zap, StopCircle, RefreshCw } from 'lucide-react';

const TalkToKingAI = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Hello! I\'m King AI, your autonomous orchestrator. I can delegate tasks to specialized agents, execute workflows, and manage your business empire.\n\nTry commands like:\n• "list agents" - See available agents\n• "use research agent to analyze market trends"\n• "execute content_pipeline workflow"\n• "help" - See all capabilities',
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [systemState, setSystemState] = useState(null);
  const [currentWorkflow, setCurrentWorkflow] = useState(null);
  const [executionStatus, setExecutionStatus] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch initial status
  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/chat/status');
      if (response.ok) {
        const data = await response.json();
        setSystemState(data);
      }
    } catch (error) {
      console.log('Status fetch failed:', error);
    }
  };

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
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: currentInput
        })
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      
      // Update system state if provided
      if (data.system_state) {
        setSystemState({
          autonomous_mode: data.system_state.autonomous_mode,
          agents: data.system_state.active_agents
        });
      }
      
      // Add AI response to messages
      const aiMessage = {
        id: Date.now(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        type: data.type || 'text',
        actions: data.actions_taken || [],
        pendingApprovals: data.pending_approvals || []
      };
      
      setMessages(prev => [...prev, aiMessage]);
      setIsLoading(false);
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

  const handleQuickAction = (action) => {
    setInput(action);
    // Auto-send after a brief delay
    setTimeout(() => {
      const sendBtn = document.querySelector('[data-send-btn]');
      if (sendBtn) sendBtn.click();
    }, 100);
  };

  const renderMessageContent = (message) => {
    const content = message.content || '';
    
    // Simple markdown rendering - split by lines and render
    const lines = content.split('\n');
    return lines.map((line, idx) => {
      // Bold text using regex replace
      let rendered = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      // Bullet points
      if (line.startsWith('•') || line.startsWith('-')) {
        return <div key={idx} className="ml-2" dangerouslySetInnerHTML={{ __html: rendered }} />;
      }
      // Lines with emojis get extra styling
      if (/^[🔄🚀📊🤖💡⚙️📋🛠️🛑✅❌🎯]/.test(line)) {
        return <div key={idx} className="font-semibold mt-2" dangerouslySetInnerHTML={{ __html: rendered }} />;
      }
      if (line.trim() === '') {
        return <div key={idx} className="h-2" />;
      }
      return <div key={idx} dangerouslySetInnerHTML={{ __html: rendered }} />;
    });
  };

  return (
    <div className="talk-to-king-ai h-full flex flex-col">
      {/* Header */}
      <div className="card glass mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Bot className="w-8 h-8 text-blue-500" />
            <div>
              <h3 className="text-lg font-semibold">King AI Orchestrator</h3>
              <p className="text-sm text-gray-400">Full agent delegation and workflow execution</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {systemState?.autonomous_mode && (
              <span className="px-2 py-1 bg-green-500 bg-opacity-20 text-green-400 text-xs rounded-full flex items-center gap-1">
                <Activity className="w-3 h-3" /> Autonomous Mode
              </span>
            )}
            <button 
              onClick={fetchStatus}
              className="p-2 hover:bg-gray-700 rounded-lg transition"
              title="Refresh Status"
            >
              <RefreshCw className="w-4 h-4 text-gray-400" />
            </button>
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
                <div className={`max-w-lg ${message.role === 'user' ? '' : 'w-full'}`}>
                  <div
                    className={`p-3 rounded-lg ${
                      message.role === 'user'
                        ? 'bg-blue-500 text-white'
                        : message.type === 'error'
                        ? 'bg-red-500 bg-opacity-20 text-red-400'
                        : message.type === 'action'
                        ? 'bg-purple-500 bg-opacity-20 text-purple-200'
                        : 'bg-gray-800 text-gray-200'
                    }`}
                  >
                    <div className="text-sm whitespace-pre-wrap">
                      {renderMessageContent(message)}
                    </div>
                    
                    {/* Show actions taken */}
                    {message.actions && message.actions.length > 0 && (
                      <div className="mt-3 p-2 bg-gray-700 rounded border-l-2 border-green-500">
                        <div className="text-xs text-green-400 font-semibold mb-1">
                          <Zap className="w-3 h-3 inline mr-1" />
                          Actions Executed:
                        </div>
                        {message.actions.map((action, idx) => (
                          <div key={idx} className="text-xs text-gray-300 flex items-center gap-2">
                            {action.success ? (
                              <CheckCircle className="w-3 h-3 text-green-500" />
                            ) : (
                              <AlertCircle className="w-3 h-3 text-red-500" />
                            )}
                            <span>{action.step_name} ({action.agent})</span>
                            {action.duration_ms && (
                              <span className="text-gray-500">{Math.round(action.duration_ms)}ms</span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {/* Show pending approvals */}
                    {message.pendingApprovals && message.pendingApprovals.length > 0 && (
                      <div className="mt-3 p-2 bg-yellow-500 bg-opacity-20 rounded border-l-2 border-yellow-500">
                        <div className="text-xs text-yellow-400 font-semibold mb-1">
                          ⏳ Pending Approvals:
                        </div>
                        {message.pendingApprovals.map((approval, idx) => (
                          <div key={idx} className="text-xs text-gray-300 mb-2">
                            <div>{approval.task_name || approval.name}</div>
                            <div className="flex gap-2 mt-1">
                              <button className="px-2 py-1 bg-green-600 hover:bg-green-500 rounded text-white text-xs">
                                Approve
                              </button>
                              <button className="px-2 py-1 bg-red-600 hover:bg-red-500 rounded text-white text-xs">
                                Reject
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    <span className="text-xs opacity-50 mt-2 block">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
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
                    <div className="animate-pulse flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      <span className="ml-2 text-gray-400">King AI is processing...</span>
                    </div>
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
                placeholder="Command the orchestrator... (try 'help' or 'list agents')"
                className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                disabled={isLoading}
              />
              <button
                data-send-btn="true"
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

        {/* Right Panel - Orchestrator Status */}
        <div className="w-80 space-y-4">
          {/* Active Agents */}
          <div className="card glass">
            <h4 className="font-semibold mb-3 flex items-center gap-2">
              <Users className="w-4 h-4 text-blue-500" />
              Active Agents
            </h4>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {systemState?.agents?.map((agent, idx) => (
                <div key={idx} className="flex items-center justify-between p-2 bg-gray-800 rounded text-sm">
                  <div>
                    <span className="font-medium">{agent.name}</span>
                    <span className={`ml-2 text-xs ${
                      agent.risk_level === 'low' ? 'text-green-400' :
                      agent.risk_level === 'medium' ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {agent.risk_level}
                    </span>
                  </div>
                  <span className="text-xs text-gray-500">{agent.status || 'idle'}</span>
                </div>
              )) || (
                <div className="text-sm text-gray-500">Say "list agents" to see agents...</div>
              )}
            </div>
            <button 
              onClick={() => handleQuickAction('list agents')}
              className="btn-secondary w-full mt-2 text-xs"
            >
              Refresh Agents
            </button>
          </div>

          {/* Autonomous Mode */}
          <div className="card glass">
            <h4 className="font-semibold mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4 text-purple-500" />
              Autonomous Mode
            </h4>
            <div className={`p-3 rounded flex items-center justify-between ${
              systemState?.autonomous_mode ? 'bg-green-500 bg-opacity-20' : 'bg-gray-800'
            }`}>
              <span className="text-sm">
                {systemState?.autonomous_mode ? '🟢 Active' : '🔴 Inactive'}
              </span>
              <button
                onClick={() => handleQuickAction(
                  systemState?.autonomous_mode ? 'stop ralph loop' : 'run ralph loop'
                )}
                className={`px-3 py-1 rounded text-xs ${
                  systemState?.autonomous_mode 
                    ? 'bg-red-600 hover:bg-red-500' 
                    : 'bg-green-600 hover:bg-green-500'
                }`}
              >
                {systemState?.autonomous_mode ? (
                  <><StopCircle className="w-3 h-3 inline mr-1" /> Stop</>
                ) : (
                  <><Play className="w-3 h-3 inline mr-1" /> Start</>
                )}
              </button>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card glass">
            <h4 className="font-semibold mb-3 flex items-center gap-2">
              <Zap className="w-4 h-4 text-yellow-500" />
              Quick Commands
            </h4>
            <div className="space-y-2">
              <button 
                onClick={() => handleQuickAction('use research agent to analyze current market trends')}
                className="btn-secondary w-full text-left text-sm"
              >
                🔍 Research market trends
              </button>
              <button 
                onClick={() => handleQuickAction('create a new business plan')}
                className="btn-secondary w-full text-left text-sm"
              >
                📝 Create business plan
              </button>
              <button 
                onClick={() => handleQuickAction('list workflows')}
                className="btn-secondary w-full text-left text-sm"
              >
                📋 List workflows
              </button>
              <button 
                onClick={() => handleQuickAction('status')}
                className="btn-secondary w-full text-left text-sm"
              >
                📊 Check system status
              </button>
              <button 
                onClick={() => handleQuickAction('spawn a code agent')}
                className="btn-secondary w-full text-left text-sm"
              >
                🤖 Spawn code agent
              </button>
            </div>
          </div>

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
        </div>
      </div>
    </div>
  );
};

export default TalkToKingAI;
