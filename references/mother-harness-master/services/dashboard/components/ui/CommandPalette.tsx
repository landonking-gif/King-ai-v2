/**
 * Command Palette Component
 * Global command search and execution
 */

'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
    Search,
    Settings,
    HelpCircle,
    Moon,
    Sun,
    PanelLeft,
    PanelRight,
    Plus,
    FileText,
    Users,
} from 'lucide-react';
import { useUIStore } from '../../stores/uiStore';
import { useWorkspaceStore } from '../../stores/workspaceStore';
import styles from './CommandPalette.module.css';

interface Command {
    id: string;
    label: string;
    icon: React.ReactNode;
    shortcut?: string;
    action: () => void;
    category: 'navigation' | 'actions' | 'settings';
}

export function CommandPalette() {
    const [search, setSearch] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);

    const {
        closeCommandPalette,
        toggleLeftSidebar,
        toggleRightPanel,
        toggleSettings,
        toggleHelp,
        theme,
        setTheme,
        setActiveRightTab,
    } = useUIStore();

    const { createConversation } = useWorkspaceStore();

    const commands: Command[] = [
        {
            id: 'new-conversation',
            label: 'New Conversation',
            icon: <Plus size={16} />,
            shortcut: '⌘N',
            action: () => { createConversation(); closeCommandPalette(); },
            category: 'actions',
        },
        {
            id: 'toggle-sidebar',
            label: 'Toggle Left Sidebar',
            icon: <PanelLeft size={16} />,
            shortcut: '⌘B',
            action: () => { toggleLeftSidebar(); closeCommandPalette(); },
            category: 'navigation',
        },
        {
            id: 'toggle-panel',
            label: 'Toggle Right Panel',
            icon: <PanelRight size={16} />,
            shortcut: '⌘.',
            action: () => { toggleRightPanel(); closeCommandPalette(); },
            category: 'navigation',
        },
        {
            id: 'open-agents',
            label: 'View Agents',
            icon: <Users size={16} />,
            action: () => { setActiveRightTab('agents'); closeCommandPalette(); },
            category: 'navigation',
        },
        {
            id: 'open-approvals',
            label: 'View Approvals',
            icon: <FileText size={16} />,
            shortcut: '⌘⇧A',
            action: () => { setActiveRightTab('approvals'); closeCommandPalette(); },
            category: 'navigation',
        },
        {
            id: 'open-settings',
            label: 'Open Settings',
            icon: <Settings size={16} />,
            shortcut: '⌘,',
            action: () => { toggleSettings(); closeCommandPalette(); },
            category: 'settings',
        },
        {
            id: 'toggle-theme',
            label: theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode',
            icon: theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />,
            action: () => { setTheme(theme === 'dark' ? 'light' : 'dark'); closeCommandPalette(); },
            category: 'settings',
        },
        {
            id: 'open-help',
            label: 'Help & Keyboard Shortcuts',
            icon: <HelpCircle size={16} />,
            shortcut: '⌘/',
            action: () => { toggleHelp(); closeCommandPalette(); },
            category: 'settings',
        },
    ];

    const filteredCommands = commands.filter((cmd) =>
        cmd.label.toLowerCase().includes(search.toLowerCase())
    );

    // Focus input on mount
    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    // Reset selection when search changes
    useEffect(() => {
        setSelectedIndex(0);
    }, [search]);

    // Keyboard navigation
    const handleKeyDown = (e: React.KeyboardEvent) => {
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedIndex((i) => Math.min(i + 1, filteredCommands.length - 1));
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedIndex((i) => Math.max(i - 1, 0));
                break;
            case 'Enter':
                e.preventDefault();
                if (filteredCommands[selectedIndex]) {
                    filteredCommands[selectedIndex].action();
                }
                break;
            case 'Escape':
                e.preventDefault();
                closeCommandPalette();
                break;
        }
    };

    return (
        <div className={styles.overlay} onClick={closeCommandPalette}>
            <div className={styles.palette} onClick={(e) => e.stopPropagation()}>
                {/* Search Input */}
                <div className={styles.searchWrapper}>
                    <Search size={18} className={styles.searchIcon} />
                    <input
                        ref={inputRef}
                        type="text"
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Type a command or search..."
                        className={styles.searchInput}
                    />
                </div>

                {/* Commands List */}
                <div className={styles.commands}>
                    {filteredCommands.length === 0 ? (
                        <div className={styles.empty}>No commands found</div>
                    ) : (
                        filteredCommands.map((cmd, index) => (
                            <button
                                key={cmd.id}
                                className={`${styles.command} ${index === selectedIndex ? styles.selected : ''
                                    }`}
                                onClick={cmd.action}
                                onMouseEnter={() => setSelectedIndex(index)}
                            >
                                <span className={styles.commandIcon}>{cmd.icon}</span>
                                <span className={styles.commandLabel}>{cmd.label}</span>
                                {cmd.shortcut && (
                                    <kbd className={styles.commandShortcut}>{cmd.shortcut}</kbd>
                                )}
                            </button>
                        ))
                    )}
                </div>

                {/* Footer */}
                <div className={styles.footer}>
                    <span>↑↓ to navigate</span>
                    <span>↵ to select</span>
                    <span>esc to close</span>
                </div>
            </div>
        </div>
    );
}
