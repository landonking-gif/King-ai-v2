/**
 * Event Stream Provider
 * Initializes WebSocket connection and event dispatching
 * Wrap this around your app to enable real-time updates
 */

'use client';

import React, { useEffect, useRef } from 'react';
import { initializeEventStream } from '../../stores/events';
import { useConnectionStore } from '../../stores/connectionStore';
import { useAgentStore } from '../../stores/agentStore';

interface EventStreamProviderProps {
    children: React.ReactNode;
}

export function EventStreamProvider({ children }: EventStreamProviderProps) {
    const initialized = useRef(false);
    const { checkHealth } = useConnectionStore();
    const { fetchApprovals } = useAgentStore();

    useEffect(() => {
        // Only initialize once
        if (initialized.current) return;
        initialized.current = true;

        console.log('[EventStream] Initializing...');

        // Initial data fetch
        checkHealth();
        fetchApprovals().catch(console.error);

        // Initialize WebSocket event stream
        const cleanup = initializeEventStream();

        // Periodic health check
        const healthInterval = setInterval(checkHealth, 30000);

        // Periodic approval refresh (fallback for missed WS events)
        const approvalInterval = setInterval(() => {
            fetchApprovals().catch(console.error);
        }, 60000);

        return () => {
            cleanup();
            clearInterval(healthInterval);
            clearInterval(approvalInterval);
        };
    }, [checkHealth, fetchApprovals]);

    return <>{children}</>;
}
