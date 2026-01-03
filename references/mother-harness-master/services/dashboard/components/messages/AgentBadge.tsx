/**
 * Agent Badge Component
 */

'use client';

import React from 'react';
import {
    Brain,
    Code,
    Search,
    BarChart3,
    Shield,
    Palette,
    HelpCircle,
    Database,
    BookOpen,
    Eye,
    RefreshCw,
    Wrench,
} from 'lucide-react';
import { agentColors } from '../../lib/mockData';
import styles from './AgentBadge.module.css';

// Local type to avoid ESM issues with shared package
type AgentType =
    | 'orchestrator' | 'researcher' | 'coder' | 'design'
    | 'analyst' | 'critic' | 'skeptic' | 'rag'
    | 'librarian' | 'vision' | 'update' | 'toolsmith';

interface AgentBadgeProps {
    type: AgentType;
    name: string;
    showName?: boolean;
    status?: 'active' | 'idle' | 'error';
    size?: 'sm' | 'md' | 'lg';
}

const iconMap: Record<AgentType, typeof Brain> = {
    orchestrator: Brain,
    coder: Code,
    researcher: Search,
    analyst: BarChart3,
    critic: Shield,
    design: Palette,
    skeptic: HelpCircle,
    rag: Database,
    librarian: BookOpen,
    vision: Eye,
    update: RefreshCw,
    toolsmith: Wrench,
};

export function AgentBadge({
    type,
    name,
    showName = true,
    status,
    size = 'md',
}: AgentBadgeProps) {
    const Icon = iconMap[type] || Brain;
    const color = agentColors[type] || 'var(--text-secondary)';

    const iconSize = size === 'sm' ? 12 : size === 'lg' ? 18 : 14;

    return (
        <div
            className={`${styles.badge} ${styles[size]}`}
            style={{ '--agent-color': color } as React.CSSProperties}
        >
            <span className={styles.icon}>
                <Icon size={iconSize} />
            </span>
            {showName && <span className={styles.name}>{name}</span>}
            {status === 'active' && <span className={styles.pulse} />}
        </div>
    );
}
