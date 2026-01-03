/**
 * Markdown Renderer Component
 * Renders markdown content with syntax highlighting for code blocks
 */

'use client';

import React, { useMemo } from 'react';
import styles from './MarkdownRenderer.module.css';

interface MarkdownRendererProps {
    content: string;
    className?: string;
}

// Simple token types for syntax highlighting
type TokenType = 'keyword' | 'string' | 'number' | 'comment' | 'function' | 'operator' | 'default';

interface Token {
    type: TokenType;
    value: string;
}

// Language keywords for highlighting
const KEYWORDS: Record<string, Set<string>> = {
    javascript: new Set(['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'async', 'await', 'try', 'catch', 'throw', 'new', 'this', 'true', 'false', 'null', 'undefined']),
    typescript: new Set(['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'async', 'await', 'try', 'catch', 'throw', 'new', 'this', 'true', 'false', 'null', 'undefined', 'interface', 'type', 'enum', 'implements', 'extends']),
    python: new Set(['def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'import', 'from', 'as', 'try', 'except', 'raise', 'with', 'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'lambda', 'async', 'await']),
};

// Simple tokenizer for code highlighting
function tokenize(code: string, language: string): Token[] {
    const keywords = KEYWORDS[language] ?? KEYWORDS['javascript'] ?? new Set<string>();
    const tokens: Token[] = [];
    let current = 0;

    while (current < code.length) {
        const char = code[current];
        if (char === undefined) break;

        // Comments
        if (code.slice(current, current + 2) === '//' || char === '#') {
            let value = '';
            while (current < code.length && code[current] !== '\n') {
                value += code[current] ?? '';
                current++;
            }
            tokens.push({ type: 'comment', value });
            continue;
        }

        // Strings
        if (char === '"' || char === "'" || char === '`') {
            const quote = char;
            let value = quote;
            current++;
            while (current < code.length && code[current] !== quote) {
                if (code[current] === '\\') {
                    value += code[current] ?? '';
                    current++;
                }
                if (current < code.length) {
                    value += code[current] ?? '';
                    current++;
                }
            }
            if (current < code.length) {
                value += code[current] ?? '';
                current++;
            }
            tokens.push({ type: 'string', value });
            continue;
        }

        // Numbers
        if (/\d/.test(char)) {
            let value = '';
            while (current < code.length && /[\d.]/.test(code[current] ?? '')) {
                value += code[current] ?? '';
                current++;
            }
            tokens.push({ type: 'number', value });
            continue;
        }

        // Words (keywords, functions, identifiers)
        if (/[a-zA-Z_]/.test(char)) {
            let value = '';
            while (current < code.length && /[a-zA-Z0-9_]/.test(code[current] ?? '')) {
                value += code[current] ?? '';
                current++;
            }
            if (keywords.has(value)) {
                tokens.push({ type: 'keyword', value });
            } else if (current < code.length && code[current] === '(') {
                tokens.push({ type: 'function', value });
            } else {
                tokens.push({ type: 'default', value });
            }
            continue;
        }

        // Operators
        if (/[+\-*/%=<>!&|^~?:]/.test(char)) {
            tokens.push({ type: 'operator', value: char });
            current++;
            continue;
        }

        // Default
        tokens.push({ type: 'default', value: char });
        current++;
    }

    return tokens;
}

// Render highlighted code
function HighlightedCode({ code, language }: { code: string; language: string }) {
    const tokens = useMemo(() => tokenize(code, language), [code, language]);

    return (
        <code>
            {tokens.map((token, i) => (
                <span key={i} className={styles[token.type] ?? ''}>
                    {token.value}
                </span>
            ))}
        </code>
    );
}

export function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
    const rendered = useMemo(() => {
        const lines = content.split('\n');
        const elements: React.ReactNode[] = [];
        let inCodeBlock = false;
        let codeBlockContent = '';
        let codeBlockLang = '';
        let blockIndex = 0;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line === undefined) continue;

            // Code block start/end
            if (line.startsWith('```')) {
                if (!inCodeBlock) {
                    inCodeBlock = true;
                    codeBlockLang = line.slice(3).trim() || 'text';
                    codeBlockContent = '';
                } else {
                    elements.push(
                        <div key={`code-${blockIndex++}`} className={styles.codeBlock}>
                            <div className={styles.codeHeader}>
                                <span className={styles.codeLang}>{codeBlockLang}</span>
                                <button
                                    className={styles.copyButton}
                                    onClick={() => navigator.clipboard.writeText(codeBlockContent)}
                                >
                                    Copy
                                </button>
                            </div>
                            <pre className={styles.codeContent}>
                                <HighlightedCode code={codeBlockContent} language={codeBlockLang} />
                            </pre>
                        </div>
                    );
                    inCodeBlock = false;
                    codeBlockContent = '';
                }
                continue;
            }

            if (inCodeBlock) {
                codeBlockContent += (codeBlockContent ? '\n' : '') + line;
                continue;
            }

            // Parse inline elements
            let processedLine = line
                // Bold
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                // Italic
                .replace(/\*(.+?)\*/g, '<em>$1</em>')
                // Inline code
                .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
                // Links
                .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

            // Headers
            if (line.startsWith('### ')) {
                elements.push(<h3 key={i} className={styles.h3} dangerouslySetInnerHTML={{ __html: processedLine.slice(4) }} />);
            } else if (line.startsWith('## ')) {
                elements.push(<h2 key={i} className={styles.h2} dangerouslySetInnerHTML={{ __html: processedLine.slice(3) }} />);
            } else if (line.startsWith('# ')) {
                elements.push(<h1 key={i} className={styles.h1} dangerouslySetInnerHTML={{ __html: processedLine.slice(2) }} />);
            }
            // List items
            else if (line.startsWith('- ') || line.startsWith('* ')) {
                elements.push(<li key={i} className={styles.listItem} dangerouslySetInnerHTML={{ __html: processedLine.slice(2) }} />);
            }
            // Numbered list
            else if (/^\d+\.\s/.test(line)) {
                elements.push(<li key={i} className={styles.orderedItem} dangerouslySetInnerHTML={{ __html: line.replace(/^\d+\.\s/, '') }} />);
            }
            // Empty line = paragraph break
            else if (line.trim() === '') {
                elements.push(<br key={i} />);
            }
            // Regular paragraph
            else {
                elements.push(<p key={i} className={styles.paragraph} dangerouslySetInnerHTML={{ __html: processedLine }} />);
            }
        }

        return elements;
    }, [content]);

    return (
        <div className={`${styles.markdown} ${className}`}>
            {rendered}
        </div>
    );
}
