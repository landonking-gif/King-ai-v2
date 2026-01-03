/**
 * Connection Store
 * Manages backend service connection status
 */

import { create } from 'zustand';
import { getHealth, type HealthResponse } from '../lib/api/client';
import { wsConnection } from '../lib/websocket';

interface ConnectionState {
    // Status
    orchestrator: 'connected' | 'disconnected' | 'checking';
    redis: 'connected' | 'disconnected' | 'unknown';
    websocket: 'connected' | 'disconnected' | 'connecting';

    // Health data
    version: string | null;
    lastChecked: string | null;

    // Actions
    checkHealth: () => Promise<void>;
    connectWebSocket: (taskId?: string) => void;
    disconnectWebSocket: () => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
    orchestrator: 'disconnected',
    redis: 'unknown',
    websocket: 'disconnected',
    version: null,
    lastChecked: null,

    checkHealth: async () => {
        set({ orchestrator: 'checking' });

        try {
            const health: HealthResponse = await getHealth();
            set({
                orchestrator: health.status === 'ok' ? 'connected' : 'disconnected',
                redis: health.redis,
                version: health.version,
                lastChecked: new Date().toISOString(),
            });
        } catch (error) {
            console.error('Health check failed:', error);
            set({
                orchestrator: 'disconnected',
                redis: 'unknown',
                lastChecked: new Date().toISOString(),
            });
        }
    },

    connectWebSocket: (taskId?: string) => {
        set({ websocket: 'connecting' });

        wsConnection.on('connection', (msg) => {
            const status = (msg.data as { status: string })?.status;
            set({ websocket: status === 'connected' ? 'connected' : 'disconnected' });
        });

        wsConnection.connect(taskId);
    },

    disconnectWebSocket: () => {
        wsConnection.disconnect();
        set({ websocket: 'disconnected' });
    },
}));
