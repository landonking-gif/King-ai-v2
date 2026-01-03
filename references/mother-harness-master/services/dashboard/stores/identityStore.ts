/**
 * User Identity Store
 * Simple identity management for personalization (no auth)
 * 
 * Stores user identity in localStorage and syncs with Redis
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface UserIdentity {
    id: string;
    name: string;
    createdAt: string;
    preferences?: {
        theme?: 'light' | 'dark' | 'system';
        leftSidebarWidth?: number;
        rightPanelWidth?: number;
    };
}

interface IdentityState {
    identity: UserIdentity | null;
    isLoading: boolean;

    // Actions
    setIdentity: (name: string) => Promise<void>;
    clearIdentity: () => void;
    updatePreferences: (prefs: Partial<UserIdentity['preferences']>) => void;
    syncWithServer: () => Promise<void>;
}

// Generate a simple unique ID
function generateUserId(): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 8);
    return `usr_${timestamp}_${random}`;
}

export const useIdentityStore = create<IdentityState>()(
    persist(
        (set, get) => ({
            identity: null,
            isLoading: false,

            setIdentity: async (name: string) => {
                set({ isLoading: true });

                const identity: UserIdentity = {
                    id: generateUserId(),
                    name: name.trim(),
                    createdAt: new Date().toISOString(),
                    preferences: {
                        theme: 'dark',
                    },
                };

                set({ identity, isLoading: false });

                // Sync to server (fire and forget)
                get().syncWithServer().catch(console.error);
            },

            clearIdentity: () => {
                set({ identity: null });
            },

            updatePreferences: (prefs) => {
                const current = get().identity;
                if (!current) return;

                set({
                    identity: {
                        ...current,
                        preferences: {
                            ...current.preferences,
                            ...prefs,
                        },
                    },
                });

                // Sync to server
                get().syncWithServer().catch(console.error);
            },

            syncWithServer: async () => {
                const identity = get().identity;
                if (!identity) return;

                try {
                    await fetch('/api/identity', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(identity),
                    });
                } catch (error) {
                    // Silent fail - server sync is optional
                    console.warn('[Identity] Failed to sync with server:', error);
                }
            },
        }),
        {
            name: 'mother-harness-identity',
            partialize: (state) => ({ identity: state.identity }),
        }
    )
);

/**
 * Hook to get current user name (for display)
 */
export function useUserName(): string {
    const identity = useIdentityStore((s) => s.identity);
    return identity?.name || 'User';
}

/**
 * Hook to get current user ID (for API calls)
 */
export function useUserId(): string | null {
    const identity = useIdentityStore((s) => s.identity);
    return identity?.id || null;
}

/**
 * Check if user has set their identity
 */
export function useHasIdentity(): boolean {
    const identity = useIdentityStore((s) => s.identity);
    return identity !== null;
}
