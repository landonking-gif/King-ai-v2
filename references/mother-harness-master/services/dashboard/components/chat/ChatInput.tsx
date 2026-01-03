/**
 * Chat Input Component
 * Input area with toolbar for formatting, attachments, and submit
 */

'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
    Send,
    Paperclip,
    Image,
    Code,
    AtSign,
    Sparkles,
    ChevronUp,
    X,
} from 'lucide-react';
import { useIdentityStore } from '../../stores/identityStore';
import styles from './ChatInput.module.css';

interface ChatInputProps {
    onSubmit: (message: string, attachments?: File[]) => void;
    isLoading?: boolean;
    placeholder?: string;
}

export function ChatInput({
    onSubmit,
    isLoading = false,
    placeholder = 'Type a message...',
}: ChatInputProps) {
    const [message, setMessage] = useState('');
    const [attachments, setAttachments] = useState<File[]>([]);
    const [showToolbar, setShowToolbar] = useState(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const identity = useIdentityStore((s) => s.identity);

    // Auto-resize textarea
    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
        }
    }, [message]);

    const handleSubmit = useCallback(() => {
        if (message.trim() && !isLoading) {
            onSubmit(message.trim(), attachments.length > 0 ? attachments : undefined);
            setMessage('');
            setAttachments([]);
        }
    }, [message, attachments, isLoading, onSubmit]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        setAttachments((prev) => [...prev, ...files]);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const removeAttachment = (index: number) => {
        setAttachments((prev) => prev.filter((_, i) => i !== index));
    };

    const insertAtCursor = (text: string) => {
        const textarea = textareaRef.current;
        if (textarea) {
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            const newMessage = message.slice(0, start) + text + message.slice(end);
            setMessage(newMessage);
            setTimeout(() => {
                textarea.selectionStart = textarea.selectionEnd = start + text.length;
                textarea.focus();
            }, 0);
        }
    };

    return (
        <div className={styles.container}>
            {/* Attachments preview */}
            {attachments.length > 0 && (
                <div className={styles.attachments}>
                    {attachments.map((file, i) => (
                        <div key={i} className={styles.attachment}>
                            <span className={styles.attachmentName}>{file.name}</span>
                            <button
                                className={styles.removeAttachment}
                                onClick={() => removeAttachment(i)}
                            >
                                <X size={12} />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Toolbar */}
            {showToolbar && (
                <div className={styles.toolbar}>
                    <button
                        className={styles.toolbarButton}
                        onClick={() => insertAtCursor('```\n\n```')}
                        title="Insert code block"
                    >
                        <Code size={16} />
                    </button>
                    <button
                        className={styles.toolbarButton}
                        onClick={() => insertAtCursor('@')}
                        title="Mention agent"
                    >
                        <AtSign size={16} />
                    </button>
                    <button
                        className={styles.toolbarButton}
                        onClick={() => fileInputRef.current?.click()}
                        title="Attach file"
                    >
                        <Paperclip size={16} />
                    </button>
                    <button
                        className={styles.toolbarButton}
                        onClick={() => fileInputRef.current?.click()}
                        title="Add image"
                    >
                        <Image size={16} />
                    </button>
                </div>
            )}

            {/* Main input area */}
            <div className={styles.inputWrapper}>
                <button
                    className={`${styles.toggleToolbar} ${showToolbar ? styles.active : ''}`}
                    onClick={() => setShowToolbar(!showToolbar)}
                    title="Toggle toolbar"
                >
                    <ChevronUp size={16} />
                </button>

                <textarea
                    ref={textareaRef}
                    className={styles.textarea}
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    rows={1}
                    disabled={isLoading}
                />

                <div className={styles.actions}>
                    <button
                        className={styles.enhanceButton}
                        title="Enhance with AI"
                        disabled={!message.trim() || isLoading}
                    >
                        <Sparkles size={16} />
                    </button>
                    <button
                        className={styles.sendButton}
                        onClick={handleSubmit}
                        disabled={!message.trim() || isLoading}
                        title="Send message"
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>

            {/* Hidden file input */}
            <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={handleFileSelect}
                className={styles.hiddenInput}
            />

            {/* Hint */}
            <div className={styles.hint}>
                {identity?.name && (
                    <span className={styles.userName}>Chatting as {identity.name}</span>
                )}
                <span className={styles.shortcut}>
                    <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line
                </span>
            </div>
        </div>
    );
}
