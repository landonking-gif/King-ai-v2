import React, { useState, useRef, useEffect } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Send, Loader } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    workflow_created?: boolean;
    workflow_id?: string;
    action?: string;
  };
}

export const ConversationalInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      type: 'assistant',
      content: 'Hello! I\'m King AI. I can help you create and manage workflows, analyze data, and execute tasks. What would you like to do today?',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const detectIntentAndMode = (message: string): { intent: string; mode: string } => {
    const lowerMessage = message.toLowerCase();
    
    // Brainstorm mode indicators
    if (lowerMessage.includes('idea') || lowerMessage.includes('think') || lowerMessage.includes('what if') || lowerMessage.includes('brainstorm')) {
      return { intent: 'brainstorm', mode: 'creative' };
    }
    
    // Command mode indicators
    if (lowerMessage.includes('create') || lowerMessage.includes('execute') || lowerMessage.includes('run') || lowerMessage.includes('launch')) {
      return { intent: 'command', mode: 'action' };
    }
    
    // Analysis mode
    if (lowerMessage.includes('analyze') || lowerMessage.includes('report') || lowerMessage.includes('show') || lowerMessage.includes('tell me')) {
      return { intent: 'analysis', mode: 'informational' };
    }
    
    return { intent: 'general', mode: 'conversational' };
  };

  const generateWorkflowFromMessage = async (message: string): Promise<any> => {
    // Simulated workflow generation
    return {
      id: `wf-${Date.now()}`,
      name: message.substring(0, 50),
      description: `Generated from: ${message}`,
      steps: [
        { id: '1', name: 'Initialization', type: 'start', status: 'pending' },
        { id: '2', name: 'Processing', type: 'execute', status: 'pending' },
        { id: '3', name: 'Completion', type: 'end', status: 'pending' },
      ],
    };
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const { intent, mode } = detectIntentAndMode(input);
      
      let assistantResponse = '';
      let metadata: Message['metadata'] = { action: intent };

      // Simulate different responses based on intent
      if (intent === 'command' && input.toLowerCase().includes('workflow')) {
        const workflow = await generateWorkflowFromMessage(input);
        assistantResponse = `I've created a new workflow "${workflow.name}" with 3 steps. Would you like to review or execute it?`;
        metadata = {
          ...metadata,
          workflow_created: true,
          workflow_id: workflow.id,
        };
      } else if (intent === 'brainstorm') {
        assistantResponse = `Great idea! In ${mode} mode. Here are some thoughts on your request: 1) We could approach this from a data perspective, 2) We could focus on automation, 3) We could integrate with external services. Which interests you most?`;
      } else if (intent === 'analysis') {
        assistantResponse = `I'll analyze that for you. Based on current system data, here are the key insights: Performance is optimal, resource utilization is at 65%, and all services are healthy. Would you like more detailed metrics?`;
      } else {
        assistantResponse = `I understand. In ${mode} mode: ${input.substring(0, 30)}... I can help you create workflows, execute tasks, or analyze system performance. What would be most helpful?`;
      }

      const assistantMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: assistantResponse,
        timestamp: new Date(),
        metadata,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError('Failed to process message: ' + String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteWorkflow = (workflowId: string) => {
    const assistantMessage: Message = {
      id: Date.now().toString(),
      type: 'assistant',
      content: `Executing workflow ${workflowId}... The workflow is now running. Step 1 of 3 completed (Initialization). Moving to Step 2...`,
      timestamp: new Date(),
      metadata: { action: 'workflow_execution' },
    };
    setMessages((prev) => [...prev, assistantMessage]);
  };

  return (
    <div className="p-6 bg-white h-screen flex flex-col">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Talk to King AI</h1>
        <p className="text-gray-600 mt-2">Conversational interface for workflow creation and system management</p>
      </div>

      <div className="flex-1 overflow-y-auto mb-6 bg-gray-50 rounded-lg p-4">
        <div className="space-y-4">
          {messages.map((msg, idx) => (
            <div key={msg.id} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              <Card className={`max-w-md p-4 ${msg.type === 'user' ? 'bg-blue-600 text-white' : 'bg-white'}`}>
                <div className="text-sm">{msg.content}</div>
                
                {msg.metadata?.workflow_created && msg.metadata?.workflow_id && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <Button
                      size="sm"
                      onClick={() => handleExecuteWorkflow(msg.metadata!.workflow_id!)}
                      className="w-full bg-green-600 hover:bg-green-700"
                    >
                      Execute Workflow
                    </Button>
                  </div>
                )}
                
                {msg.metadata?.action && (
                  <div className="mt-2 flex justify-end">
                    <Badge variant="secondary" className="text-xs">
                      {msg.metadata.action}
                    </Badge>
                  </div>
                )}
                
                <div className="text-xs opacity-70 mt-2">
                  {msg.timestamp.toLocaleTimeString()}
                </div>
              </Card>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      <form onSubmit={handleSendMessage} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message... (e.g., 'Create a data analysis workflow' or 'What's the system status?')"
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />
        <Button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-700 text-white px-6"
        >
          {loading ? <Loader size={20} className="animate-spin" /> : <Send size={20} />}
        </Button>
      </form>

      <div className="mt-4 p-3 bg-blue-50 rounded-lg text-sm text-gray-700">
        <p className="font-semibold mb-2">💡 Tips:</p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          <li>Use "Create..." to generate workflows</li>
          <li>Use "Analyze..." for system insights</li>
          <li>Use "What if..." for brainstorming ideas</li>
          <li>Use "Show..." to view current state</li>
        </ul>
      </div>
    </div>
  );
};

export default ConversationalInterface;
