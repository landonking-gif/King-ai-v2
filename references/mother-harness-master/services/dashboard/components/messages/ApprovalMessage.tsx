/**
 * Approval Message Component
 * Interactive approval request with accept/reject actions
 */

'use client';

import React, { useState } from 'react';
import { Shield, Check, X, ChevronDown, ChevronRight, FileCode, AlertTriangle } from 'lucide-react';
import type { ChatMessage } from '../../types/ui';
import { useAgentStore } from '../../stores/agentStore';
import styles from './Message.module.css';

interface ApprovalMessageProps {
    message: ChatMessage;
}

export function ApprovalMessage({ message }: ApprovalMessageProps) {
    const [isExpanded, setIsExpanded] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false);
    const { resolveApproval } = useAgentStore();

    const approvalId = message.metadata?.approvalId as string;
    const agentName = message.metadata?.agentName || message.agentName || 'Agent';
    const action = message.metadata?.action || 'action';
    const files = message.metadata?.files as string[] | undefined;
    const risk = message.metadata?.risk as 'low' | 'medium' | 'high' | undefined;
    const status = message.metadata?.status as 'pending' | 'approved' | 'rejected' | undefined;

    const formatTime = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    const handleApprove = async () => {
        if (!approvalId || isProcessing) return;
        setIsProcessing(true);
        try {
            await resolveApproval(approvalId, 'approved');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleReject = async () => {
        if (!approvalId || isProcessing) return;
        setIsProcessing(true);
        try {
            await resolveApproval(approvalId, 'rejected');
        } finally {
            setIsProcessing(false);
        }
    };

    const getRiskBadge = () => {
        if (!risk) return null;
        const riskStyles: Record<string, string> = {
            low: styles.riskLow ?? '',
            medium: styles.riskMedium ?? '',
            high: styles.riskHigh ?? '',
        };
        return (
            <span className={`${styles.riskBadge} ${riskStyles[risk] || ''}`}>
                {risk === 'high' && <AlertTriangle size={12} />}
                {risk} risk
            </span>
        );
    };

    const isPending = status === 'pending' || status === undefined;

    return (
        <div className={`${styles.approvalMessage} ${status ? styles[status] : ''}`}>
            <div className={styles.approvalHeader}>
                <div className={styles.approvalTitle}>
                    <Shield size={18} className={styles.approvalIcon} />
                    <span>Approval Required</span>
                    {getRiskBadge()}
                </div>
                <span className={styles.approvalTime}>{formatTime(message.timestamp)}</span>
            </div>

            <div className={styles.approvalBody}>
                <p className={styles.approvalDescription}>
                    <strong>{agentName}</strong> wants to {action}
                </p>

                {message.content && (
                    <p className={styles.approvalReason}>{message.content}</p>
                )}

                {files && files.length > 0 && (
                    <button
                        className={styles.approvalFiles}
                        onClick={() => setIsExpanded(!isExpanded)}
                    >
                        {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        <FileCode size={14} />
                        <span>{files.length} file{files.length > 1 ? 's' : ''} affected</span>
                    </button>
                )}

                {isExpanded && files && files.length > 0 && (
                    <div className={styles.fileList}>
                        {files.map((file, i) => (
                            <div key={i} className={styles.fileItem}>
                                <FileCode size={12} />
                                <span>{file}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {isPending && (
                <div className={styles.approvalActions}>
                    <button
                        className={`${styles.approvalButton} ${styles.reject}`}
                        onClick={handleReject}
                        disabled={isProcessing}
                    >
                        <X size={16} />
                        Reject
                    </button>
                    <button
                        className={`${styles.approvalButton} ${styles.approve}`}
                        onClick={handleApprove}
                        disabled={isProcessing}
                    >
                        <Check size={16} />
                        Approve
                    </button>
                </div>
            )}

            {status === 'approved' && (
                <div className={`${styles.approvalStatus} ${styles.approved}`}>
                    <Check size={14} />
                    Approved
                </div>
            )}

            {status === 'rejected' && (
                <div className={`${styles.approvalStatus} ${styles.rejected}`}>
                    <X size={14} />
                    Rejected
                </div>
            )}
        </div>
    );
}
