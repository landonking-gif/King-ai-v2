/**
 * Left Sidebar Component
 * Conversation list with grouping and library section
 */

'use client';

import React from 'react';
import {
    Plus,
    Search,
    Pin,
    MessageSquare,
    MoreHorizontal,
    BookOpen,
    ChevronDown,
    ChevronRight,
    PanelLeftClose,
} from 'lucide-react';
import { useUIStore } from '../../stores/uiStore';
import { useWorkspaceStore } from '../../stores/workspaceStore';
import { useResizable } from '../../hooks/useResizable';
import type { Conversation } from '../../types/ui';
import styles from './LeftSidebar.module.css';

interface LeftSidebarProps {
    width: number;
}

export function LeftSidebar({ width: initialWidth }: LeftSidebarProps) {
    const { setLeftSidebarWidth, toggleLeftSidebar } = useUIStore();
    const {
        activeConversationId,
        searchQuery,
        setActiveConversation,
        createConversation,
        setSearchQuery,
        getPinnedConversations,
        getTodayConversations,
        getYesterdayConversations,
        getOlderConversations,
    } = useWorkspaceStore();

    const [libraryOpen, setLibraryOpen] = React.useState(true);

    const { width, isResizing, handleMouseDown } = useResizable({
        initialWidth,
        minWidth: 200,
        maxWidth: 400,
        direction: 'right',
        onResize: setLeftSidebarWidth,
    });

    const pinnedConversations = getPinnedConversations();
    const todayConversations = getTodayConversations();
    const yesterdayConversations = getYesterdayConversations();
    const olderConversations = getOlderConversations();

    return (
        <aside className={styles.sidebar} style={{ width }}>
            <div className={styles.content}>
                {/* Header */}
                <div className={styles.header}>
                    <button
                        className={styles.newButton}
                        onClick={createConversation}
                    >
                        <Plus size={18} />
                        New Conversation
                    </button>
                    <button
                        className={styles.collapseButton}
                        onClick={toggleLeftSidebar}
                        aria-label="Collapse sidebar"
                    >
                        <PanelLeftClose size={18} />
                    </button>
                </div>

                {/* Search */}
                <div className={styles.search}>
                    <Search size={16} className={styles.searchIcon} />
                    <input
                        type="text"
                        placeholder="Search conversations..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className={styles.searchInput}
                    />
                </div>

                {/* Conversation List */}
                <div className={styles.list}>
                    {/* Pinned */}
                    {pinnedConversations.length > 0 && (
                        <ConversationGroup
                            title="Pinned"
                            icon={<Pin size={14} />}
                            conversations={pinnedConversations}
                            activeId={activeConversationId}
                            onSelect={setActiveConversation}
                        />
                    )}

                    {/* Today */}
                    {todayConversations.length > 0 && (
                        <ConversationGroup
                            title="Today"
                            conversations={todayConversations}
                            activeId={activeConversationId}
                            onSelect={setActiveConversation}
                        />
                    )}

                    {/* Yesterday */}
                    {yesterdayConversations.length > 0 && (
                        <ConversationGroup
                            title="Yesterday"
                            conversations={yesterdayConversations}
                            activeId={activeConversationId}
                            onSelect={setActiveConversation}
                        />
                    )}

                    {/* Last 7 Days */}
                    {olderConversations.length > 0 && (
                        <ConversationGroup
                            title="Last 7 Days"
                            conversations={olderConversations}
                            activeId={activeConversationId}
                            onSelect={setActiveConversation}
                        />
                    )}
                </div>

                {/* Library Section */}
                <div className={styles.library}>
                    <button
                        className={styles.libraryHeader}
                        onClick={() => setLibraryOpen(!libraryOpen)}
                    >
                        <BookOpen size={16} />
                        <span>Library</span>
                        {libraryOpen ? (
                            <ChevronDown size={16} />
                        ) : (
                            <ChevronRight size={16} />
                        )}
                    </button>
                    {libraryOpen && (
                        <div className={styles.libraryContent}>
                            <div className={styles.libraryItem}>
                                📚 Documentation (12 files)
                            </div>
                            <div className={styles.libraryItem}>
                                💻 Project Files (45 files)
                            </div>
                            <div className={styles.libraryItem}>
                                📝 Notes (8 files)
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Resize Handle */}
            <div
                className={`${styles.resizeHandle} ${isResizing ? styles.resizing : ''}`}
                onMouseDown={handleMouseDown}
            />
        </aside>
    );
}

interface ConversationGroupProps {
    title: string;
    icon?: React.ReactNode;
    conversations: Conversation[];
    activeId: string | null;
    onSelect: (id: string) => void;
}

function ConversationGroup({
    title,
    icon,
    conversations,
    activeId,
    onSelect,
}: ConversationGroupProps) {
    return (
        <div className={styles.group}>
            <div className={styles.groupTitle}>
                {icon}
                <span>{title}</span>
            </div>
            {conversations.map((conv) => (
                <button
                    key={conv.id}
                    className={`${styles.conversation} ${conv.id === activeId ? styles.active : ''
                        }`}
                    onClick={() => onSelect(conv.id)}
                >
                    <MessageSquare size={16} className={styles.convIcon} />
                    <div className={styles.convContent}>
                        <div className={styles.convTitle}>{conv.title}</div>
                        <div className={styles.convPreview}>{conv.preview}</div>
                    </div>
                    <button
                        className={styles.convMenu}
                        onClick={(e) => {
                            e.stopPropagation();
                            // TODO: Open context menu
                        }}
                    >
                        <MoreHorizontal size={14} />
                    </button>
                </button>
            ))}
        </div>
    );
}
