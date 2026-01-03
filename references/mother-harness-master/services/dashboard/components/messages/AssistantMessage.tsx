/**
 * Assistant Message Component
 */

'use client';

import React from 'react';
import { Copy, RefreshCw, Clock, Zap } from 'lucide-react';
import type { ChatMessage } from '../../types/ui';
import { AgentBadge } from './AgentBadge';
import styles from './Message.module.css';

interface AssistantMessageProps {
    message: ChatMessage;
}

export function AssistantMessage({ message }: AssistantMessageProps) {
    const formatTime = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    const handleCopy = () => {
        navigator.clipboard.writeText(message.content);
    };

    return (
        <div className={`${styles.message} ${styles.assistantMessage}`}>
            <div className={styles.agentAvatar}>
                {message.agentType && (
                    <AgentBadge
                        type={message.agentType}
                        name={message.agentName || message.agentType}
                        showName={false}
                    />
                )}
            </div>
            <div className={styles.content}>
                <div className={styles.header}>
                    {message.agentType && (
                        <AgentBadge
                            type={message.agentType}
                            name={message.agentName || message.agentType}
                            showName={true}
                        />
                    )}
                    <span className={styles.time}>{formatTime(message.timestamp)}</span>
                </div>
                <div className={`${styles.text} ${message.isStreaming ? styles.streaming : ''}`}>
                    {message.content}
                    {message.isStreaming && <span className={styles.cursor}>▌</span>}
                </div>

                {/* Metadata */}
                {message.metadata && !message.isStreaming && (
                    <div className={styles.metadata}>
                        {message.metadata.tokensUsed && (
                            <span className={styles.metaItem}>
                                <Zap size={12} />
                                {message.metadata.tokensUsed} tokens
                            </span>
                        )}
                        {message.metadata.durationMs && (
                            <span className={styles.metaItem}>
                                <Clock size={12} />
                                {(message.metadata.durationMs / 1000).toFixed(1)}s
                            </span>
                        )}
                        {message.metadata.model && (
                            <span className={styles.metaItem}>
                                {message.metadata.model}
                            </span>
                        )}
                    </div>
                )}

                {!message.isStreaming && (
                    <div className={styles.actions}>
                        <button
                            className={styles.actionButton}
                            onClick={handleCopy}
                            title="Copy"
                        >
                            <Copy size={14} />
                        </button>
                        <button className={styles.actionButton} title="Regenerate">
                            <RefreshCw size={14} />
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
