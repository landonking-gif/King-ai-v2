/**
 * Agents Panel Component
 * Shows agent statuses and activity
 */

'use client';

import React from 'react';
import { useAgentStore } from '../../stores/agentStore';
import { AgentBadge } from '../messages/AgentBadge';
import styles from './Panel.module.css';

export function AgentsPanel() {
    const { agents, getActiveAgentCount } = useAgentStore();
    const activeCount = getActiveAgentCount();

    return (
        <div className={styles.panel}>
            <div className={styles.header}>
                <h3 className={styles.title}>Active Agents</h3>
                <span className={styles.count}>{activeCount} active</span>
            </div>

            <div className={styles.list}>
                {agents.map((agent) => (
                    <div
                        key={agent.type}
                        className={`${styles.item} ${agent.status === 'active' ? styles.active : ''
                            }`}
                    >
                        <AgentBadge
                            type={agent.type}
                            name={agent.name}
                            status={agent.status}
                            size="md"
                        />
                        <div className={styles.itemContent}>
                            {agent.currentTask && (
                                <p className={styles.itemTask}>{agent.currentTask}</p>
                            )}
                            <div className={styles.itemMeta}>
                                <span>{agent.tokensUsed.toLocaleString()} tokens</span>
                                <span className={`${styles.status} ${styles[agent.status]}`}>
                                    {agent.status}
                                </span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
