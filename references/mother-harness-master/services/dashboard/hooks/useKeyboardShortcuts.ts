/**
 * Keyboard Shortcuts Hook
 * Handles global keyboard shortcuts for Mission Control
 */

'use client';

import { useEffect, useCallback } from 'react';
import { useUIStore } from '../stores/uiStore';
import { useWorkspaceStore } from '../stores/workspaceStore';

interface ShortcutConfig {
    key: string;
    ctrl?: boolean;
    meta?: boolean;
    shift?: boolean;
    action: () => void;
    description: string;
}

export function useKeyboardShortcuts() {
    const {
        toggleLeftSidebar,
        toggleRightPanel,
        toggleCommandPalette,
        toggleSettings,
        toggleHelp,
        setActiveRightTab,
        closeAllModals,
    } = useUIStore();

    const { createConversation } = useWorkspaceStore();

    const shortcuts: ShortcutConfig[] = [
        {
            key: 'k',
            meta: true,
            action: toggleCommandPalette,
            description: 'Toggle command palette',
        },
        {
            key: 'b',
            meta: true,
            action: toggleLeftSidebar,
            description: 'Toggle left sidebar',
        },
        {
            key: '.',
            meta: true,
            action: toggleRightPanel,
            description: 'Toggle right panel',
        },
        {
            key: 'a',
            meta: true,
            shift: true,
            action: () => setActiveRightTab('approvals'),
            description: 'Open Approvals tab',
        },
        {
            key: 'n',
            meta: true,
            action: createConversation,
            description: 'New conversation',
        },
        {
            key: ',',
            meta: true,
            action: toggleSettings,
            description: 'Open settings',
        },
        {
            key: '/',
            meta: true,
            action: toggleHelp,
            description: 'Open help',
        },
        {
            key: 'Escape',
            action: () => {
                closeAllModals();
            },
            description: 'Close modals / Cancel streaming',
        },
    ];

    const handleKeyDown = useCallback(
        (event: KeyboardEvent) => {
            // Don't trigger shortcuts when typing in inputs
            const target = event.target as HTMLElement;
            if (
                target.tagName === 'INPUT' ||
                target.tagName === 'TEXTAREA' ||
                target.isContentEditable
            ) {
                // Allow Escape to work even in inputs
                if (event.key !== 'Escape') {
                    return;
                }
            }

            for (const shortcut of shortcuts) {
                const metaMatch = shortcut.meta ? (event.metaKey || event.ctrlKey) : true;
                const ctrlMatch = shortcut.ctrl ? event.ctrlKey : true;
                const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey || shortcut.shift;
                const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();

                if (keyMatch && metaMatch && ctrlMatch && shiftMatch) {
                    event.preventDefault();
                    shortcut.action();
                    return;
                }
            }
        },
        [shortcuts]
    );

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown]);

    return { shortcuts };
}

// Hook for displaying shortcut hints
export function useShortcutHint(key: string, meta?: boolean, shift?: boolean): string {
    const isMac = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform);

    let hint = '';
    if (meta) hint += isMac ? '⌘' : 'Ctrl+';
    if (shift) hint += isMac ? '⇧' : 'Shift+';
    hint += key.toUpperCase();

    return hint;
}
