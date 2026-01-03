/**
 * Right Panel Component
 * Tabbed panel with 5 sections: Agents | Context | Tasks | Files | Approvals
 */

'use client';

import React, { useState } from 'react';
import {
    Users,
    Brain,
    ListTodo,
    FolderOpen,
    Shield,
} from 'lucide-react';
import { useAgentStore } from '../../stores/agentStore';
import { AgentsPanel } from '../panels/AgentsPanel';
import { ContextPanel } from '../panels/ContextPanel';
import { TasksPanel } from '../panels/TasksPanel';
import { FilesPanel } from '../panels/FilesPanel';
import { ApprovalsPanel } from '../panels/ApprovalsPanel';
import styles from './RightPanel.module.css';

type TabId = 'agents' | 'context' | 'tasks' | 'files' | 'approvals';

interface Tab {
    id: TabId;
    label: string;
    icon: React.ReactNode;
    badge: number | undefined;
}

interface RightPanelProps {
    width?: number;
}

export function RightPanel({ width = 360 }: RightPanelProps) {
    const [activeTab, setActiveTab] = useState<TabId>('agents');
    const { getPendingApprovals, getActiveAgentCount, getActiveTasks } = useAgentStore();

    const pendingApprovals = getPendingApprovals();
    const activeAgents = getActiveAgentCount();
    const activeTasks = getActiveTasks();

    const tabs: Tab[] = [
        {
            id: 'agents',
            label: 'Agents',
            icon: <Users size={16} />,
            badge: activeAgents > 0 ? activeAgents : undefined,
        },
        {
            id: 'context',
            label: 'Context',
            icon: <Brain size={16} />,
            badge: undefined,
        },
        {
            id: 'tasks',
            label: 'Tasks',
            icon: <ListTodo size={16} />,
            badge: activeTasks.length > 0 ? activeTasks.length : undefined,
        },
        {
            id: 'files',
            label: 'Files',
            icon: <FolderOpen size={16} />,
            badge: undefined,
        },
        {
            id: 'approvals',
            label: 'Approvals',
            icon: <Shield size={16} />,
            badge: pendingApprovals.length > 0 ? pendingApprovals.length : undefined,
        },
    ];

    const renderTabContent = () => {
        switch (activeTab) {
            case 'agents':
                return <AgentsPanel />;
            case 'context':
                return <ContextPanel />;
            case 'tasks':
                return <TasksPanel />;
            case 'files':
                return <FilesPanel />;
            case 'approvals':
                return <ApprovalsPanel />;
            default:
                return null;
        }
    };

    return (
        <div className={styles.panel} style={{ width }}>
            {/* Tab bar */}
            <div className={styles.tabBar}>
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        className={`${styles.tab} ${activeTab === tab.id ? styles.active : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                        title={tab.label}
                    >
                        {tab.icon}
                        <span className={styles.tabLabel}>{tab.label}</span>
                        {tab.badge !== undefined && (
                            <span className={`${styles.badge} ${tab.id === 'approvals' ? styles.badgeWarning : ''}`}>
                                {tab.badge}
                            </span>
                        )}
                    </button>
                ))}
            </div>

            {/* Tab content */}
            <div className={styles.content}>
                {renderTabContent()}
            </div>
        </div>
    );
}
