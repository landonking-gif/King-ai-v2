/**
 * Workspace Store - Zustand store for workspace and conversation management
 */

import { create } from 'zustand';
import type { Conversation } from '../types/ui';
import { mockConversations } from '../lib/mockData';

interface WorkspaceState {
    // Conversations
    conversations: Conversation[];
    activeConversationId: string | null;

    // Search/filter
    searchQuery: string;

    // Computed getters as actions
    getActiveConversation: () => Conversation | undefined;
    getPinnedConversations: () => Conversation[];
    getTodayConversations: () => Conversation[];
    getYesterdayConversations: () => Conversation[];
    getOlderConversations: () => Conversation[];

    // Actions
    setActiveConversation: (id: string | null) => void;
    createConversation: () => string;
    deleteConversation: (id: string) => void;
    pinConversation: (id: string) => void;
    unpinConversation: (id: string) => void;
    updateConversationTitle: (id: string, title: string) => void;
    setSearchQuery: (query: string) => void;
}

const isToday = (date: Date): boolean => {
    const today = new Date();
    return date.toDateString() === today.toDateString();
};

const isYesterday = (date: Date): boolean => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    return date.toDateString() === yesterday.toDateString();
};

export const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
    // Initial state with mock data
    conversations: mockConversations,
    activeConversationId: mockConversations[0]?.id ?? null,
    searchQuery: '',

    // Getters
    getActiveConversation: () => {
        const state = get();
        return state.conversations.find((c) => c.id === state.activeConversationId);
    },

    getPinnedConversations: () => {
        const state = get();
        return state.conversations
            .filter((c) => c.isPinned)
            .filter((c) => c.title.toLowerCase().includes(state.searchQuery.toLowerCase()));
    },

    getTodayConversations: () => {
        const state = get();
        return state.conversations
            .filter((c) => !c.isPinned && isToday(new Date(c.updatedAt)))
            .filter((c) => c.title.toLowerCase().includes(state.searchQuery.toLowerCase()));
    },

    getYesterdayConversations: () => {
        const state = get();
        return state.conversations
            .filter((c) => !c.isPinned && isYesterday(new Date(c.updatedAt)))
            .filter((c) => c.title.toLowerCase().includes(state.searchQuery.toLowerCase()));
    },

    getOlderConversations: () => {
        const state = get();
        return state.conversations
            .filter((c) => {
                const date = new Date(c.updatedAt);
                return !c.isPinned && !isToday(date) && !isYesterday(date);
            })
            .filter((c) => c.title.toLowerCase().includes(state.searchQuery.toLowerCase()));
    },

    // Actions
    setActiveConversation: (id) => set({ activeConversationId: id }),

    createConversation: () => {
        const id = `conv-${Date.now()}`;
        const now = new Date().toISOString();
        const newConversation: Conversation = {
            id,
            title: 'New Conversation',
            preview: '',
            createdAt: now,
            updatedAt: now,
            isPinned: false,
            messageCount: 0,
            agentsInvolved: [],
        };
        set((state) => ({
            conversations: [newConversation, ...state.conversations],
            activeConversationId: id,
        }));
        return id;
    },

    deleteConversation: (id) => set((state) => ({
        conversations: state.conversations.filter((c) => c.id !== id),
        activeConversationId: state.activeConversationId === id
            ? (state.conversations[0]?.id ?? null)
            : state.activeConversationId,
    })),

    pinConversation: (id) => set((state) => ({
        conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, isPinned: true } : c
        ),
    })),

    unpinConversation: (id) => set((state) => ({
        conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, isPinned: false } : c
        ),
    })),

    updateConversationTitle: (id, title) => set((state) => ({
        conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, title, updatedAt: new Date().toISOString() } : c
        ),
    })),

    setSearchQuery: (query) => set({ searchQuery: query }),
}));
