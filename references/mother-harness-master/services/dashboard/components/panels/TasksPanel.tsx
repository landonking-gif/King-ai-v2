/**
 * Tasks Panel Component
 * Shows current tasks and progress
 */

'use client';

import React from 'react';
import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react';
import { useAgentStore } from '../../stores/agentStore';
import { AgentBadge } from '../messages/AgentBadge';
import styles from './Panel.module.css';

export function TasksPanel() {
    const { tasks } = useAgentStore();

    const statusIcon = (status: string) => {
        switch (status) {
            case 'completed':
                return <CheckCircle2 size={16} className={styles.iconSuccess} />;
            case 'in_progress':
                return <Loader2 size={16} className={styles.iconActive} />;
            case 'failed':
                return <XCircle size={16} className={styles.iconError} />;
            default:
                return <Circle size={16} className={styles.iconPending} />;
        }
    };

    return (
        <div className={styles.panel}>
            <div className={styles.header}>
                <h3 className={styles.title}>Tasks</h3>
                <span className={styles.count}>{tasks.length} tasks</span>
            </div>

            <div className={styles.list}>
                {tasks.map((task) => (
                    <div key={task.id} className={styles.taskItem}>
                        <div className={styles.taskHeader}>
                            {statusIcon(task.status)}
                            <span className={styles.taskDescription}>
                                {task.description}
                            </span>
                        </div>
                        <div className={styles.taskMeta}>
                            <AgentBadge
                                type={task.agentType}
                                name={task.agentType}
                                size="sm"
                            />
                            {task.progress !== undefined && task.status === 'in_progress' && (
                                <div className={styles.progressBar}>
                                    <div
                                        className={styles.progressFill}
                                        style={{ width: `${task.progress}%` }}
                                    />
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
