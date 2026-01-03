/**
 * Header Component
 * Top navigation bar with logo, search, notifications, and user controls
 */

'use client';

import React, { useState } from 'react';
import {
    Search,
    Bell,
    Settings,
    Moon,
    Sun,
    ChevronDown,
    Zap,
    X,
    AlertCircle,
    CheckCircle,
    Clock,
} from 'lucide-react';
import { useUIStore } from '../../stores/uiStore';
import { useAgentStore } from '../../stores/agentStore';
import { useIdentityStore } from '../../stores/identityStore';
import { useShortcutHint } from '../../hooks/useKeyboardShortcuts';
import styles from './Header.module.css';

// Mock notifications - will be replaced with real data
const mockNotifications = [
    { id: '1', type: 'approval', message: 'Coder agent needs approval for file changes', time: '2m ago', unread: true },
    { id: '2', type: 'success', message: 'Build completed successfully', time: '5m ago', unread: true },
    { id: '3', type: 'info', message: 'Context loaded from project files', time: '10m ago', unread: false },
];

export function Header() {
    const { theme, setTheme, openCommandPalette } = useUIStore();
    const { getPendingApprovals } = useAgentStore();
    const identity = useIdentityStore((s) => s.identity);
    const cmdKHint = useShortcutHint('K', true);

    const [showNotifications, setShowNotifications] = useState(false);
    const [showProjectMenu, setShowProjectMenu] = useState(false);

    const pendingApprovals = getPendingApprovals();
    const notificationCount = pendingApprovals.length + mockNotifications.filter(n => n.unread).length;

    const getNotificationIcon = (type: string) => {
        switch (type) {
            case 'approval': return <AlertCircle size={14} className={styles.notifIconWarning} />;
            case 'success': return <CheckCircle size={14} className={styles.notifIconSuccess} />;
            default: return <Clock size={14} className={styles.notifIconInfo} />;
        }
    };

    return (
        <header className={styles.header}>
            {/* Left section - Logo and project selector */}
            <div className={styles.left}>
                <div className={styles.logo}>
                    <Zap className={styles.logoIcon} size={24} />
                    <span className={styles.logoText}>Mother-Harness</span>
                </div>

                <div
                    className={styles.projectSelector}
                    onClick={() => setShowProjectMenu(!showProjectMenu)}
                >
                    <span className={styles.projectName}>Current Project</span>
                    <ChevronDown size={16} />

                    {showProjectMenu && (
                        <div className={styles.projectMenu}>
                            <div className={styles.projectMenuItem}>
                                <span>Authentication System</span>
                            </div>
                            <div className={styles.projectMenuItem}>
                                <span>Dashboard UI</span>
                            </div>
                            <div className={styles.projectMenuItem}>
                                <span>API Integration</span>
                            </div>
                            <div className={styles.projectMenuDivider} />
                            <div className={styles.projectMenuItem}>
                                <span>+ New Project</span>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Center section - Command palette trigger */}
            <div className={styles.center}>
                <button
                    className={styles.searchButton}
                    onClick={openCommandPalette}
                    aria-label="Open command palette"
                >
                    <Search size={16} />
                    <span>Search or run commands...</span>
                    <kbd className={styles.kbd}>{cmdKHint}</kbd>
                </button>
            </div>

            {/* Right section - Actions and user */}
            <div className={styles.right}>
                {/* Notifications */}
                <div className={styles.notificationWrapper}>
                    <button
                        className={styles.iconButton}
                        onClick={() => setShowNotifications(!showNotifications)}
                        aria-label="Notifications"
                    >
                        <Bell size={20} />
                        {notificationCount > 0 && (
                            <span className={styles.notificationBadge}>
                                {notificationCount > 9 ? '9+' : notificationCount}
                            </span>
                        )}
                    </button>

                    {showNotifications && (
                        <div className={styles.notificationDropdown}>
                            <div className={styles.notificationHeader}>
                                <span>Notifications</span>
                                <button
                                    className={styles.notificationClose}
                                    onClick={() => setShowNotifications(false)}
                                >
                                    <X size={16} />
                                </button>
                            </div>
                            <div className={styles.notificationList}>
                                {pendingApprovals.map(approval => (
                                    <div key={approval.id} className={`${styles.notificationItem} ${styles.unread}`}>
                                        <AlertCircle size={14} className={styles.notifIconWarning} />
                                        <div className={styles.notificationContent}>
                                            <span className={styles.notificationMessage}>
                                                {approval.agentName} needs approval
                                            </span>
                                            <span className={styles.notificationTime}>Just now</span>
                                        </div>
                                    </div>
                                ))}
                                {mockNotifications.map(notif => (
                                    <div
                                        key={notif.id}
                                        className={`${styles.notificationItem} ${notif.unread ? styles.unread : ''}`}
                                    >
                                        {getNotificationIcon(notif.type)}
                                        <div className={styles.notificationContent}>
                                            <span className={styles.notificationMessage}>{notif.message}</span>
                                            <span className={styles.notificationTime}>{notif.time}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                <button
                    className={styles.iconButton}
                    aria-label="Settings"
                >
                    <Settings size={20} />
                </button>

                <button
                    className={styles.iconButton}
                    onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                    aria-label="Toggle theme"
                >
                    {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
                </button>

                <div className={styles.divider} />

                {/* User identity */}
                <button className={styles.userButton}>
                    <div className={styles.avatar}>
                        {identity?.name?.charAt(0).toUpperCase() || 'U'}
                    </div>
                    <span className={styles.userName}>{identity?.name || 'User'}</span>
                    <ChevronDown size={14} />
                </button>
            </div>
        </header>
    );
}

