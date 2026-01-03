/**
 * Status Bar Component
 * Bottom bar with connection status, metrics, and shortcuts
 * Polls real health endpoint
 */

'use client';

import React, { useEffect } from 'react';
import { useAgentStore } from '../../stores/agentStore';
import { useConnectionStore } from '../../stores/connectionStore';
import { useShortcutHint } from '../../hooks/useKeyboardShortcuts';
import styles from './StatusBar.module.css';

// Health check interval (30 seconds)
const HEALTH_CHECK_INTERVAL = 30000;

export function StatusBar() {
    const {
        agents,
        tokenUsage,
        sessionDuration,
        getActiveAgentCount,
    } = useAgentStore();

    const {
        orchestrator,
        redis,
        websocket,
        checkHealth,
    } = useConnectionStore();

    const cmdKHint = useShortcutHint('K', true);
    const activeCount = getActiveAgentCount();
    const totalAgents = agents.length;

    // Poll health endpoint
    useEffect(() => {
        // Initial check
        checkHealth();

        // Set up polling
        const interval = setInterval(checkHealth, HEALTH_CHECK_INTERVAL);

        return () => clearInterval(interval);
    }, [checkHealth]);

    const formatNumber = (num: number): string => {
        if (num >= 1000) {
            return `${(num / 1000).toFixed(1)}K`;
        }
        return num.toString();
    };

    const formatCost = (cost: number): string => {
        return `$${cost.toFixed(2)}`;
    };

    const getStatusClass = (status: string): string => {
        if (status === 'connected') return styles.connected || '';
        if (status === 'checking' || status === 'connecting') return styles.checking || '';
        return styles.disconnected || '';
    };

    return (
        <footer className={styles.statusBar}>
            {/* Left - Connection indicators */}
            <div className={styles.section}>
                <div className={styles.connection}>
                    <span className={`${styles.dot} ${getStatusClass(orchestrator)}`} />
                    <span>API</span>
                </div>
                <div className={styles.connection}>
                    <span className={`${styles.dot} ${getStatusClass(redis)}`} />
                    <span>Redis</span>
                </div>
                <div className={styles.connection}>
                    <span className={`${styles.dot} ${getStatusClass(websocket)}`} />
                    <span>WS</span>
                </div>
            </div>

            {/* Center - Metrics */}
            <div className={styles.section}>
                <div className={styles.metric}>
                    <span className={styles.metricLabel}>Active Agents</span>
                    <span className={styles.metricValue}>
                        {activeCount}/{totalAgents}
                    </span>
                </div>
                <div className={styles.divider} />
                <div className={styles.metric}>
                    <span className={styles.metricLabel}>Session</span>
                    <span className={styles.metricValue}>{sessionDuration}</span>
                </div>
                <div className={styles.divider} />
                <div className={styles.metric}>
                    <span className={styles.metricLabel}>Tokens</span>
                    <span className={styles.metricValue}>
                        {formatNumber(tokenUsage.local)} local, {formatNumber(tokenUsage.cloud)} cloud
                    </span>
                </div>
                <div className={styles.divider} />
                <div className={styles.metric}>
                    <span className={styles.metricLabel}>Cost</span>
                    <span className={styles.metricValue}>{formatCost(tokenUsage.cost)}</span>
                </div>
            </div>

            {/* Right - Keyboard hint */}
            <div className={styles.section}>
                <div className={styles.hint}>
                    <kbd className={styles.kbd}>{cmdKHint}</kbd>
                    <span>Command Palette</span>
                </div>
            </div>
        </footer>
    );
}
