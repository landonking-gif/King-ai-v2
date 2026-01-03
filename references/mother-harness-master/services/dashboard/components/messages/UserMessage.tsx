/**
 * User Message Component
 */

'use client';

import React from 'react';
import { User, Copy, Edit3 } from 'lucide-react';
import type { ChatMessage } from '../../types/ui';
import styles from './Message.module.css';

interface UserMessageProps {
    message: ChatMessage;
}

export function UserMessage({ message }: UserMessageProps) {
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
        <div className={`${styles.message} ${styles.userMessage}`}>
            <div className={styles.avatar}>
                <User size={18} />
            </div>
            <div className={styles.content}>
                <div className={styles.header}>
                    <span className={styles.name}>You</span>
                    <span className={styles.time}>{formatTime(message.timestamp)}</span>
                </div>
                <div className={styles.text}>{message.content}</div>
                <div className={styles.actions}>
                    <button
                        className={styles.actionButton}
                        onClick={handleCopy}
                        title="Copy"
                    >
                        <Copy size={14} />
                    </button>
                    <button className={styles.actionButton} title="Edit">
                        <Edit3 size={14} />
                    </button>
                </div>
            </div>
        </div>
    );
}
