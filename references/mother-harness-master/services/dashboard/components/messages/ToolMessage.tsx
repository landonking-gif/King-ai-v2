/**
 * Tool Message Component
 * Displays tool execution results with collapsible details
 */

'use client';

import React, { useState } from 'react';
import { Terminal, ChevronDown, ChevronRight, CheckCircle, XCircle, Clock } from 'lucide-react';
import type { ChatMessage } from '../../types/ui';
import styles from './Message.module.css';

interface ToolMessageProps {
    message: ChatMessage;
}

export function ToolMessage({ message }: ToolMessageProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    const toolName = message.metadata?.toolName || 'Tool';
    const status = message.metadata?.status || 'completed';
    const duration = message.metadata?.durationMs;
    const args = message.metadata?.args;
    const result = message.metadata?.result;

    const formatTime = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    const getStatusIcon = () => {
        switch (status) {
            case 'completed':
                return <CheckCircle size={14} className={styles.iconSuccess} />;
            case 'error':
                return <XCircle size={14} className={styles.iconError} />;
            case 'running':
                return <Clock size={14} className={styles.iconInfo} />;
            default:
                return <Terminal size={14} />;
        }
    };

    return (
        <div className={`${styles.toolMessage} ${styles[status] || ''}`}>
            <button
                className={styles.toolHeader}
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className={styles.toolInfo}>
                    {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    <Terminal size={14} />
                    <span className={styles.toolName}>{toolName}</span>
                    {getStatusIcon()}
                </div>
                <div className={styles.toolMeta}>
                    {duration && (
                        <span className={styles.toolDuration}>
                            {(duration / 1000).toFixed(2)}s
                        </span>
                    )}
                    <span className={styles.toolTime}>{formatTime(message.timestamp)}</span>
                </div>
            </button>

            {isExpanded && (
                <div className={styles.toolDetails}>
                    {args && (
                        <div className={styles.toolSection}>
                            <span className={styles.toolSectionLabel}>Arguments</span>
                            <pre className={styles.toolCode}>
                                {JSON.stringify(args, null, 2)}
                            </pre>
                        </div>
                    )}
                    {(result || message.content) && (
                        <div className={styles.toolSection}>
                            <span className={styles.toolSectionLabel}>Result</span>
                            <pre className={styles.toolCode}>
                                {typeof result === 'object' && result !== null
                                    ? JSON.stringify(result, null, 2)
                                    : String(result || message.content)}
                            </pre>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
