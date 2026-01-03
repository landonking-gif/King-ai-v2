/**
 * Message List Component
 * Renders a list of chat messages with appropriate component for each type
 */

'use client';

import React from 'react';
import type { ChatMessage } from '../../types/ui';
import { UserMessage } from './UserMessage';
import { AssistantMessage } from './AssistantMessage';
import { SystemMessage } from './SystemMessage';
import { ToolMessage } from './ToolMessage';
import { ApprovalMessage } from './ApprovalMessage';
import styles from './MessageList.module.css';

interface MessageListProps {
    messages: ChatMessage[];
    className?: string;
}

export function MessageList({ messages, className = '' }: MessageListProps) {
    const renderMessage = (message: ChatMessage) => {
        switch (message.role) {
            case 'user':
                return <UserMessage key={message.id} message={message} />;
            case 'assistant':
                return <AssistantMessage key={message.id} message={message} />;
            case 'system':
                return <SystemMessage key={message.id} message={message} />;
            case 'tool':
                return <ToolMessage key={message.id} message={message} />;
            case 'approval':
                return <ApprovalMessage key={message.id} message={message} />;
            default:
                return null;
        }
    };

    return (
        <div className={`${styles.list} ${className}`}>
            {messages.map(renderMessage)}
        </div>
    );
}

export { UserMessage } from './UserMessage';
export { AssistantMessage } from './AssistantMessage';
export { SystemMessage } from './SystemMessage';
export { ToolMessage } from './ToolMessage';
export { ApprovalMessage } from './ApprovalMessage';
