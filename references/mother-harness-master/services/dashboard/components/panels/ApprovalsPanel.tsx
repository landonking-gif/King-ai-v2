/**
 * Approvals Panel Component
 * Shows pending approvals requiring user action
 */

'use client';

import React from 'react';
import { AlertTriangle, Check, X, MessageSquare } from 'lucide-react';
import { useAgentStore } from '../../stores/agentStore';
import { AgentBadge } from '../messages/AgentBadge';
import styles from './Panel.module.css';

export function ApprovalsPanel() {
    const { resolveApproval, getPendingApprovals } = useAgentStore();
    const pendingApprovals = getPendingApprovals();

    const riskColors = {
        low: 'var(--risk-low)',
        medium: 'var(--risk-medium)',
        high: 'var(--risk-high)',
    };

    return (
        <div className={styles.panel}>
            <div className={styles.header}>
                <h3 className={styles.title}>Approvals</h3>
                {pendingApprovals.length > 0 && (
                    <span className={styles.badge}>{pendingApprovals.length}</span>
                )}
            </div>

            {pendingApprovals.length === 0 ? (
                <div className={styles.empty}>
                    <Check size={24} className={styles.emptyIcon} />
                    <p>No pending approvals</p>
                </div>
            ) : (
                <div className={styles.list}>
                    {pendingApprovals.map((approval) => (
                        <div key={approval.id} className={styles.approvalItem}>
                            <div className={styles.approvalHeader}>
                                <AgentBadge
                                    type={approval.agentType}
                                    name={approval.agentName}
                                    size="sm"
                                />
                                <span
                                    className={styles.riskBadge}
                                    style={{ '--risk-color': riskColors[approval.riskLevel] } as React.CSSProperties}
                                >
                                    <AlertTriangle size={12} />
                                    {approval.riskLevel}
                                </span>
                            </div>
                            <p className={styles.approvalDesc}>{approval.description}</p>

                            {approval.preview.files && (
                                <div className={styles.approvalPreview}>
                                    <span className={styles.previewLabel}>Files:</span>
                                    {approval.preview.files.map((file) => (
                                        <code key={file} className={styles.previewFile}>
                                            {file}
                                        </code>
                                    ))}
                                </div>
                            )}

                            {approval.preview.commands && (
                                <div className={styles.approvalPreview}>
                                    <span className={styles.previewLabel}>Commands:</span>
                                    {approval.preview.commands.map((cmd) => (
                                        <code key={cmd} className={styles.previewCommand}>
                                            {cmd}
                                        </code>
                                    ))}
                                </div>
                            )}

                            <div className={styles.approvalActions}>
                                <button
                                    className={styles.approveButton}
                                    onClick={() => resolveApproval(approval.id, 'approved')}
                                >
                                    <Check size={14} />
                                    Approve
                                </button>
                                <button
                                    className={styles.rejectButton}
                                    onClick={() => resolveApproval(approval.id, 'rejected')}
                                >
                                    <X size={14} />
                                    Reject
                                </button>
                                <button className={styles.askButton}>
                                    <MessageSquare size={14} />
                                    Ask
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
