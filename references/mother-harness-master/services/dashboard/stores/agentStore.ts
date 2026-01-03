/**
 * Agent Store - Zustand store for agent status and approvals
 * Integrated with real API
 */

import { create } from 'zustand';
import type { AgentStatus, TokenUsage, ConnectionStatus, ApprovalDisplay, TaskDisplay, AgentType } from '../types/ui';
import { mockAgents, mockApprovals, mockTasks } from '../lib/mockData';
import { getPendingApprovals, respondToApproval as apiRespondToApproval, type Approval } from '../lib/api/client';
import { wsConnection, type ApprovalNeededMessage } from '../lib/websocket';

interface AgentState {
    // Agent statuses
    agents: AgentStatus[];

    // Approvals
    approvals: ApprovalDisplay[];
    isLoadingApprovals: boolean;

    // Tasks
    tasks: TaskDisplay[];

    // Token usage
    tokenUsage: TokenUsage;

    // Session info
    sessionStart: string;
    sessionDuration: string;

    // Connection status
    connectionStatus: ConnectionStatus;

    // Computed
    getActiveAgentCount: () => number;
    getPendingApprovals: () => ApprovalDisplay[];
    getActiveTasks: () => TaskDisplay[];

    // Actions
    updateAgentStatus: (type: string, status: 'active' | 'idle' | 'error') => void;
    addApproval: (approval: ApprovalDisplay) => void;
    resolveApproval: (id: string, status: 'approved' | 'rejected') => Promise<void>;
    addTask: (task: TaskDisplay) => void;
    updateTaskStatus: (id: string, status: TaskDisplay['status']) => void;
    updateTokenUsage: (local: number, cloud: number) => void;
    updateConnectionStatus: (status: Partial<ConnectionStatus>) => void;
    updateSessionDuration: () => void;

    // API actions
    fetchApprovals: () => Promise<void>;
    setupWebSocketHandlers: () => () => void;
}

// Convert API approval to display format
function mapApprovalToDisplay(approval: Approval): ApprovalDisplay {
    return {
        id: approval.id,
        agentType: approval.agent as AgentType,
        agentName: approval.agent,
        description: approval.description,
        riskLevel: approval.risk_level,
        type: approval.type,
        preview: {
            ...(approval.details.files && { files: approval.details.files }),
            ...(approval.details.commands && { commands: approval.details.commands }),
        },
        createdAt: approval.created_at,
        status: approval.status,
    };
}

export const useAgentStore = create<AgentState>((set, get) => ({
    // Initial state with mock data (will be replaced by API data)
    agents: mockAgents,
    approvals: mockApprovals,
    isLoadingApprovals: false,
    tasks: mockTasks,
    tokenUsage: { local: 45000, cloud: 2000, cost: 1.20 },
    sessionStart: new Date().toISOString(),
    sessionDuration: '0h 0m',
    connectionStatus: {
        redis: 'disconnected',
        ollama: 'disconnected',
        api: 'disconnected',
    },

    // Computed
    getActiveAgentCount: () => {
        return get().agents.filter((a) => a.status === 'active').length;
    },

    getPendingApprovals: () => {
        return get().approvals.filter((a) => a.status === 'pending');
    },

    getActiveTasks: () => {
        return get().tasks.filter((t) => t.status === 'in_progress' || t.status === 'pending');
    },

    // Actions
    updateAgentStatus: (type, status) => set((state) => ({
        agents: state.agents.map((a) =>
            a.type === type ? { ...a, status, lastActive: new Date().toISOString() } : a
        ),
    })),

    addApproval: (approval) => set((state) => ({
        approvals: [approval, ...state.approvals],
    })),

    resolveApproval: async (id, status) => {
        // Optimistic update
        set((state) => ({
            approvals: state.approvals.map((a) =>
                a.id === id ? { ...a, status } : a
            ),
        }));

        try {
            // Call real API
            await apiRespondToApproval(id, status === 'approved');
        } catch (error) {
            console.error('Failed to respond to approval:', error);
            // Revert on error
            set((state) => ({
                approvals: state.approvals.map((a) =>
                    a.id === id ? { ...a, status: 'pending' } : a
                ),
            }));
            throw error;
        }
    },

    addTask: (task) => set((state) => ({
        tasks: [task, ...state.tasks],
    })),

    updateTaskStatus: (id, status) => set((state) => ({
        tasks: state.tasks.map((t) =>
            t.id === id ? { ...t, status } : t
        ),
    })),

    updateTokenUsage: (local, cloud) => set((state) => ({
        tokenUsage: {
            local: state.tokenUsage.local + local,
            cloud: state.tokenUsage.cloud + cloud,
            cost: state.tokenUsage.cost + (cloud * 0.00001), // Simplified cost calc
        },
    })),

    updateConnectionStatus: (status) => set((state) => ({
        connectionStatus: { ...state.connectionStatus, ...status },
    })),

    updateSessionDuration: () => {
        const start = new Date(get().sessionStart);
        const now = new Date();
        const diff = now.getTime() - start.getTime();
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        set({ sessionDuration: `${hours}h ${minutes}m` });
    },

    // Fetch approvals from API
    fetchApprovals: async () => {
        set({ isLoadingApprovals: true });

        try {
            const apiApprovals = await getPendingApprovals();
            const displayApprovals = apiApprovals.map(mapApprovalToDisplay);
            set({ approvals: displayApprovals, isLoadingApprovals: false });
        } catch (error) {
            console.error('Failed to fetch approvals:', error);
            set({ isLoadingApprovals: false });
            // Keep mock data on error
        }
    },

    // Setup WebSocket handlers for real-time approval notifications
    setupWebSocketHandlers: () => {
        const unsubApproval = wsConnection.on('approval_needed', (msg) => {
            const data = (msg as ApprovalNeededMessage).data;

            const newApproval: ApprovalDisplay = {
                id: data.approval_id,
                agentType: data.agent as AgentType,
                agentName: data.agent,
                description: data.description,
                riskLevel: data.risk_level,
                type: 'approval',
                preview: {},
                createdAt: new Date().toISOString(),
                status: 'pending',
            };

            get().addApproval(newApproval);
        });

        return unsubApproval;
    },
}));

// Update session duration every minute
if (typeof window !== 'undefined') {
    setInterval(() => {
        useAgentStore.getState().updateSessionDuration();
    }, 60000);
}
