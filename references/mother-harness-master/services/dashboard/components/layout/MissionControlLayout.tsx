/**
 * Mission Control Layout
 * Main layout component with 7 sections
 */

'use client';

import React from 'react';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import { useUIStore } from '../../stores/uiStore';
import { Header } from './Header';
import { LeftSidebar } from './LeftSidebar';
import { MainChat } from './MainChat';
import { RightPanel } from './RightPanel';
import { StatusBar } from './StatusBar';
import { CommandPalette } from '../ui/CommandPalette';

import styles from './MissionControlLayout.module.css';

interface MissionControlLayoutProps {
    children?: React.ReactNode;
}

export function MissionControlLayout({ children }: MissionControlLayoutProps) {
    useKeyboardShortcuts();

    const {
        leftSidebarOpen,
        leftSidebarWidth,
        rightPanelOpen,
        rightPanelWidth,
        commandPaletteOpen,
    } = useUIStore();

    return (
        <div className={styles.container}>
            {/* Header */}
            <Header />

            {/* Main Content Area */}
            <div className={styles.content}>
                {/* Left Sidebar */}
                {leftSidebarOpen && (
                    <LeftSidebar width={leftSidebarWidth} />
                )}

                {/* Main Chat Area */}
                <MainChat />

                {/* Right Panel */}
                {rightPanelOpen && (
                    <RightPanel width={rightPanelWidth} />
                )}
            </div>

            {/* Status Bar */}
            <StatusBar />

            {/* Command Palette Modal */}
            {commandPaletteOpen && <CommandPalette />}

            {/* Additional children (modals, etc.) */}
            {children}
        </div>
    );
}
