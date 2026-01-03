/**
 * System Message Component
 * For orchestrator status updates, errors, and notifications
 */

'use client';

import React from 'react';
import { Info, AlertCircle, CheckCircle, AlertTriangle } from 'lucide-react';
import type { ChatMessage } from '../../types/ui';
import styles from './Message.module.css';

interface SystemMessageProps {
    message: ChatMessage;
}

type SystemLevel = 'info' | 'success' | 'warning' | 'error';

export function SystemMessage({ message }: SystemMessageProps) {
    const level = (message.metadata?.level as SystemLevel) || 'info';

    const getIcon = () => {
        switch (level) {
            case 'success':
                return <CheckCircle size={16} className={styles.iconSuccess} />;
            case 'warning':
                return <AlertTriangle size={16} className={styles.iconWarning} />;
            case 'error':
                return <AlertCircle size={16} className={styles.iconError} />;
            default:
                return <Info size={16} className={styles.iconInfo} />;
        }
    };

    const formatTime = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    return (
        <div className={`${styles.systemMessage} ${styles[level] || ''}`}>
            <div className={styles.systemIcon}>
                {getIcon()}
            </div>
            <div className={styles.systemContent}>
                <span className={styles.systemText}>{message.content}</span>
                <span className={styles.systemTime}>{formatTime(message.timestamp)}</span>
            </div>
        </div>
    );
}
