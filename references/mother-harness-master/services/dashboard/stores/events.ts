/**
 * Event Dispatcher
 * Single entry point for all WebSocket/real-time events
 * Routes events to appropriate Zustand stores
 * 
 * Pattern: WebSocket → dispatchMotherEvent() → stores → UI
 */

import { useThreadStore } from './threadStore';
import { useAgentStore } from './agentStore';
import { useConnectionStore } from './connectionStore';
import type { AgentType, ChatMessage, ApprovalDisplay } from '../types/ui';

// ============================================
// Event Types (matching orchestrator taxonomy)
// ============================================

export type MotherEventType =
    // Run lifecycle
    | 'run_created'
    | 'run_started'
    | 'run_waiting_approval'
    | 'run_completed'
    | 'run_terminated'
    // Step lifecycle  
    | 'step_started'
    | 'step_completed'
    | 'step_failed'
    // Approvals
    | 'approval_requested'
    | 'approval_approved'
    | 'approval_rejected'
    // Streaming
    | 'stream_content'
    | 'stream_done'
    // Agent status
    | 'agent_status'
    // Connection
    | 'connection'
    | 'health';

export interface MotherEvent {
    type: MotherEventType;
    run_id?: string;
    task_id?: string;
    agent?: string;
    timestamp: string;
    data?: Record<string, unknown>;
}

// ============================================
// Event Dispatcher
// ============================================

export function dispatchMotherEvent(event: MotherEvent): void {
    const threadStore = useThreadStore.getState();
    const agentStore = useAgentStore.getState();
    const connectionStore = useConnectionStore.getState();

    console.log('[Event]', event.type, event);

    switch (event.type) {
        // ─────────────────────────────────────────
        // Run Events
        // ─────────────────────────────────────────
        case 'run_created':
        case 'run_started': {
            // Update agent status to active
            if (event.agent) {
                agentStore.updateAgentStatus(event.agent, 'active');
            }
            break;
        }

        case 'run_completed': {
            // Clear streaming, update agent to idle
            threadStore.stopStreaming();
            if (event.agent) {
                agentStore.updateAgentStatus(event.agent, 'idle');
            }
            break;
        }

        case 'run_terminated': {
            // Error state
            threadStore.stopStreaming();
            if (event.agent) {
                agentStore.updateAgentStatus(event.agent, 'error');
            }
            break;
        }

        // ─────────────────────────────────────────
        // Step Events (Agent Activity)
        // ─────────────────────────────────────────
        case 'step_started': {
            const agentName = event.agent || 'orchestrator';
            const details = event.data as { description?: string } | undefined;

            agentStore.updateAgentStatus(agentName, 'active');

            // Create agent message placeholder
            const message: ChatMessage = {
                id: `msg-${Date.now()}`,
                role: 'assistant',
                content: details?.description || '',
                timestamp: event.timestamp,
                agentType: agentName as AgentType,
                agentName: agentName,
                isStreaming: true,
            };

            threadStore.addMessage(message);
            threadStore.startStreaming(message.id);
            break;
        }

        case 'step_completed': {
            const details = event.data as { tokens_used?: number; duration_ms?: number } | undefined;

            threadStore.stopStreaming();

            // Update last message with metadata if we have values
            const messages = threadStore.messages;
            const lastMessage = messages[messages.length - 1];
            if (lastMessage && lastMessage.role === 'assistant') {
                const metadata: { tokensUsed?: number; durationMs?: number } = {};
                if (typeof details?.tokens_used === 'number') {
                    metadata.tokensUsed = details.tokens_used;
                }
                if (typeof details?.duration_ms === 'number') {
                    metadata.durationMs = details.duration_ms;
                }
                threadStore.updateMessage(lastMessage.id, {
                    isStreaming: false,
                    ...(Object.keys(metadata).length > 0 && { metadata }),
                });
            }

            if (event.agent) {
                agentStore.updateAgentStatus(event.agent, 'idle');
            }
            break;
        }

        case 'step_failed': {
            const details = event.data as { error?: string } | undefined;

            threadStore.stopStreaming();

            // Add error message
            const errorMessage: ChatMessage = {
                id: `msg-error-${Date.now()}`,
                role: 'assistant',
                content: `Error: ${details?.error || 'Step failed'}`,
                timestamp: event.timestamp,
                agentType: (event.agent || 'orchestrator') as AgentType,
                agentName: event.agent || 'System',
            };
            threadStore.addMessage(errorMessage);

            if (event.agent) {
                agentStore.updateAgentStatus(event.agent, 'error');
            }
            break;
        }

        // ─────────────────────────────────────────
        // Streaming Content
        // ─────────────────────────────────────────
        case 'stream_content': {
            const details = event.data as { content?: string } | undefined;
            if (details?.content) {
                threadStore.appendToStream(details.content);
            }
            break;
        }

        case 'stream_done': {
            threadStore.stopStreaming();
            break;
        }

        // ─────────────────────────────────────────
        // Approval Events
        // ─────────────────────────────────────────
        case 'approval_requested': {
            const details = event.data as {
                approval_id?: string;
                description?: string;
                risk_level?: 'low' | 'medium' | 'high';
                type?: string;
                files?: string[];
                commands?: string[];
            } | undefined;

            if (details?.approval_id) {
                const approval: ApprovalDisplay = {
                    id: details.approval_id,
                    agentType: (event.agent || 'orchestrator') as AgentType,
                    agentName: event.agent || 'Unknown',
                    description: details.description || 'Approval required',
                    riskLevel: details.risk_level || 'medium',
                    type: details.type || 'action',
                    preview: {
                        ...(details.files && { files: details.files }),
                        ...(details.commands && { commands: details.commands }),
                    },
                    createdAt: event.timestamp,
                    status: 'pending',
                };

                agentStore.addApproval(approval);
            }
            break;
        }

        case 'approval_approved':
        case 'approval_rejected': {
            const details = event.data as { approval_id?: string } | undefined;
            if (details?.approval_id) {
                const status = event.type === 'approval_approved' ? 'approved' : 'rejected';
                // Update locally (API call already made by user action)
                const state = agentStore;
                const existing = state.approvals.find(a => a.id === details.approval_id);
                if (existing && existing.status === 'pending') {
                    // This is a confirmation from server, update status
                    useAgentStore.setState({
                        approvals: state.approvals.map(a =>
                            a.id === details.approval_id ? { ...a, status } : a
                        ),
                    });
                }
            }
            break;
        }

        case 'run_waiting_approval': {
            // Run is paused waiting for approval
            if (event.agent) {
                agentStore.updateAgentStatus(event.agent, 'idle');
            }
            break;
        }

        // ─────────────────────────────────────────
        // Agent Status Updates
        // ─────────────────────────────────────────
        case 'agent_status': {
            const details = event.data as { status?: 'active' | 'idle' | 'error' } | undefined;
            if (event.agent && details?.status) {
                agentStore.updateAgentStatus(event.agent, details.status);
            }
            break;
        }

        // ─────────────────────────────────────────
        // Connection Events
        // ─────────────────────────────────────────
        case 'connection': {
            const details = event.data as { status?: string } | undefined;
            if (details?.status === 'connected') {
                useConnectionStore.setState({ websocket: 'connected' });
            } else {
                useConnectionStore.setState({ websocket: 'disconnected' });
            }
            break;
        }

        case 'health': {
            connectionStore.checkHealth(); // Trigger fresh health check
            break;
        }

        default:
            console.warn('[Event] Unknown event type:', event.type);
    }
}

// ============================================
// WebSocket Integration
// ============================================

import { wsConnection, type WebSocketMessage } from '../lib/websocket';

/**
 * Initialize event stream - call once at app startup
 */
export function initializeEventStream(): () => void {
    // Subscribe to all WebSocket messages
    const unsubscribe = wsConnection.onAll((message: WebSocketMessage) => {
        // Map WebSocket message to MotherEvent format
        const event: MotherEvent = {
            type: message.type as MotherEventType,
            timestamp: message.timestamp || new Date().toISOString(),
        };

        // Only add optional properties if they have values
        if (message.run_id) event.run_id = message.run_id;
        if (message.task_id) event.task_id = message.task_id;
        if (message.data) event.data = message.data as Record<string, unknown>;

        // Extract agent from data if present
        const dataWithAgent = message.data as { agent?: string } | undefined;
        if (dataWithAgent?.agent) {
            event.agent = dataWithAgent.agent;
        }

        dispatchMotherEvent(event);
    });

    // Connect WebSocket
    wsConnection.connect();

    return () => {
        unsubscribe();
        wsConnection.disconnect();
    };
}
