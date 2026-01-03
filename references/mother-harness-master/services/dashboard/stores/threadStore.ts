/**
 * Thread Store - Zustand store for chat messages
 * Integrated with real API and WebSocket
 */

import { create } from 'zustand';
import type { ChatMessage, AgentType } from '../types/ui';
import { mockMessages } from '../lib/mockData';
import { ask, type CreateRunResponse } from '../lib/api/client';
import { wsConnection, type StreamContentMessage, type AgentEventMessage } from '../lib/websocket';

interface ThreadState {
    // Messages for current conversation
    messages: ChatMessage[];

    // Active run state
    activeRunId: string | null;
    activeTaskId: string | null;

    // Streaming state
    isStreaming: boolean;
    streamingMessageId: string | null;

    // Input state
    inputValue: string;

    // Error state
    error: string | null;

    // Actions
    setMessages: (messages: ChatMessage[]) => void;
    addMessage: (message: ChatMessage) => void;
    updateMessage: (id: string, updates: Partial<ChatMessage>) => void;
    deleteMessage: (id: string) => void;
    clearMessages: () => void;

    // Streaming actions
    startStreaming: (messageId: string) => void;
    appendToStream: (content: string) => void;
    stopStreaming: () => void;

    // Input actions
    setInputValue: (value: string) => void;

    // Send message action (connects to API)
    sendMessage: (content: string, projectId?: string) => Promise<void>;

    // WebSocket handlers
    setupWebSocketHandlers: () => () => void;
}

export const useThreadStore = create<ThreadState>((set, get) => ({
    // Initial state - start with mock data for now, can be cleared when real data loads
    messages: mockMessages,
    activeRunId: null,
    activeTaskId: null,
    isStreaming: false,
    streamingMessageId: null,
    inputValue: '',
    error: null,

    // Message actions
    setMessages: (messages) => set({ messages }),

    addMessage: (message) => set((state) => ({
        messages: [...state.messages, message],
    })),

    updateMessage: (id, updates) => set((state) => ({
        messages: state.messages.map((m) =>
            m.id === id ? { ...m, ...updates } : m
        ),
    })),

    deleteMessage: (id) => set((state) => ({
        messages: state.messages.filter((m) => m.id !== id),
    })),

    clearMessages: () => set({ messages: [] }),

    // Streaming actions
    startStreaming: (messageId) => set({
        isStreaming: true,
        streamingMessageId: messageId,
    }),

    appendToStream: (content) => set((state) => {
        if (!state.streamingMessageId) return state;
        return {
            messages: state.messages.map((m) =>
                m.id === state.streamingMessageId
                    ? { ...m, content: m.content + content }
                    : m
            ),
        };
    }),

    stopStreaming: () => set((state) => ({
        messages: state.messages.map((m) =>
            m.id === state.streamingMessageId
                ? { ...m, isStreaming: false }
                : m
        ),
        isStreaming: false,
        streamingMessageId: null,
    })),

    // Input actions
    setInputValue: (value) => set({ inputValue: value }),

    // Send message - connects to real API
    sendMessage: async (content, projectId) => {
        const userMessage: ChatMessage = {
            id: `msg-${Date.now()}`,
            role: 'user',
            content,
            timestamp: new Date().toISOString(),
        };

        // Add user message immediately
        set((state) => ({
            messages: [...state.messages, userMessage],
            inputValue: '',
            error: null,
        }));

        try {
            // Call the real API
            const response: CreateRunResponse = await ask({
                query: content,
                ...(projectId && { project_id: projectId }),
            });

            set({
                activeRunId: response.run_id,
                activeTaskId: response.task_id,
            });

            // Create placeholder for assistant response
            const assistantMessage: ChatMessage = {
                id: `msg-${Date.now()}`,
                role: 'assistant',
                content: '',
                timestamp: new Date().toISOString(),
                agentType: 'orchestrator',
                agentName: 'Orchestrator',
                isStreaming: true,
            };

            set((state) => ({
                messages: [...state.messages, assistantMessage],
                isStreaming: true,
                streamingMessageId: assistantMessage.id,
            }));

            // Connect WebSocket for real-time updates
            wsConnection.connect(response.task_id);

        } catch (error) {
            console.error('Failed to send message:', error);

            // Add error message
            const errorMessage: ChatMessage = {
                id: `msg-error-${Date.now()}`,
                role: 'assistant',
                content: `Failed to process request: ${error instanceof Error ? error.message : 'Unknown error'}. Please check your connection and try again.`,
                timestamp: new Date().toISOString(),
                agentType: 'orchestrator',
                agentName: 'System',
            };

            set((state) => ({
                messages: [...state.messages, errorMessage],
                error: error instanceof Error ? error.message : 'Unknown error',
                isStreaming: false,
            }));
        }
    },

    // Setup WebSocket handlers for real-time updates
    setupWebSocketHandlers: () => {
        // Handle stream content
        const unsubStream = wsConnection.on('stream_content', (msg) => {
            const data = (msg as StreamContentMessage).data;

            if (data.done) {
                get().stopStreaming();
            } else if (data.content) {
                get().appendToStream(data.content);
            }
        });

        // Handle agent events
        const unsubAgent = wsConnection.on('agent_event', (msg) => {
            const data = (msg as AgentEventMessage).data;

            if (data.event === 'started') {
                // Create new message for this agent
                const agentMessage: ChatMessage = {
                    id: `msg-${Date.now()}`,
                    role: 'assistant',
                    content: data.message || '',
                    timestamp: new Date().toISOString(),
                    agentType: data.agent as AgentType,
                    agentName: data.agent,
                    isStreaming: true,
                };

                set((state) => ({
                    messages: [...state.messages, agentMessage],
                    streamingMessageId: agentMessage.id,
                }));
            } else if (data.event === 'completed') {
                get().stopStreaming();
            }
        });

        // Handle task completion
        const unsubTask = wsConnection.on('task_update', (msg) => {
            const data = msg.data as { status: string };
            if (data.status === 'completed' || data.status === 'failed') {
                get().stopStreaming();
            }
        });

        // Return cleanup function
        return () => {
            unsubStream();
            unsubAgent();
            unsubTask();
        };
    },
}));
