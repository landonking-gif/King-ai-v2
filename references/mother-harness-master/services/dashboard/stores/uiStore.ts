/**
 * UI Store - Zustand store for UI state management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { RightPanelTab } from '../types/ui';

interface UIState {
    // Sidebar state
    leftSidebarOpen: boolean;
    leftSidebarWidth: number;

    // Right panel state
    rightPanelOpen: boolean;
    rightPanelWidth: number;
    activeRightTab: RightPanelTab;

    // Modal states
    commandPaletteOpen: boolean;
    settingsOpen: boolean;
    helpOpen: boolean;

    // Theme
    theme: 'dark' | 'light';

    // Actions
    toggleLeftSidebar: () => void;
    setLeftSidebarWidth: (width: number) => void;
    toggleRightPanel: () => void;
    setRightPanelWidth: (width: number) => void;
    setActiveRightTab: (tab: RightPanelTab) => void;
    toggleCommandPalette: () => void;
    openCommandPalette: () => void;
    closeCommandPalette: () => void;
    toggleSettings: () => void;
    toggleHelp: () => void;
    setTheme: (theme: 'dark' | 'light') => void;
    closeAllModals: () => void;
}

export const useUIStore = create<UIState>()(
    persist(
        (set) => ({
            // Initial state
            leftSidebarOpen: true,
            leftSidebarWidth: 280,
            rightPanelOpen: true,
            rightPanelWidth: 320,
            activeRightTab: 'agents',
            commandPaletteOpen: false,
            settingsOpen: false,
            helpOpen: false,
            theme: 'dark',

            // Actions
            toggleLeftSidebar: () => set((state) => ({ leftSidebarOpen: !state.leftSidebarOpen })),
            setLeftSidebarWidth: (width) => set({ leftSidebarWidth: Math.min(400, Math.max(200, width)) }),
            toggleRightPanel: () => set((state) => ({ rightPanelOpen: !state.rightPanelOpen })),
            setRightPanelWidth: (width) => set({ rightPanelWidth: Math.min(500, Math.max(280, width)) }),
            setActiveRightTab: (tab) => set({ activeRightTab: tab, rightPanelOpen: true }),
            toggleCommandPalette: () => set((state) => ({ commandPaletteOpen: !state.commandPaletteOpen })),
            openCommandPalette: () => set({ commandPaletteOpen: true }),
            closeCommandPalette: () => set({ commandPaletteOpen: false }),
            toggleSettings: () => set((state) => ({ settingsOpen: !state.settingsOpen })),
            toggleHelp: () => set((state) => ({ helpOpen: !state.helpOpen })),
            setTheme: (theme) => set({ theme }),
            closeAllModals: () => set({
                commandPaletteOpen: false,
                settingsOpen: false,
                helpOpen: false,
            }),
        }),
        {
            name: 'mother-harness-ui',
            partialize: (state) => ({
                leftSidebarOpen: state.leftSidebarOpen,
                leftSidebarWidth: state.leftSidebarWidth,
                rightPanelOpen: state.rightPanelOpen,
                rightPanelWidth: state.rightPanelWidth,
                activeRightTab: state.activeRightTab,
                theme: state.theme,
            }),
        }
    )
);
