/**
 * Token Counter Example Plugin
 * Demonstrates the plugin SDK capabilities
 */

import type { PluginModule, PluginContext } from '../lib/plugins/types';

interface TokenStats {
    messagesAnalyzed: number;
    totalTokens: number;
    averagePerMessage: number;
}

let context: PluginContext | null = null;
let stats: TokenStats = {
    messagesAnalyzed: 0,
    totalTokens: 0,
    averagePerMessage: 0,
};

function estimateTokens(text: string): number {
    // Simple estimation: ~4 characters per token
    return Math.ceil(text.length / 4);
}

function createPanelContent(): HTMLElement {
    const container = document.createElement('div');
    container.style.cssText = `
        padding: 16px;
        font-family: inherit;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Token Analytics';
    title.style.cssText = `
        margin: 0 0 16px 0;
        font-size: 14px;
        font-weight: 600;
        color: var(--text-primary);
    `;

    const statsContainer = document.createElement('div');
    statsContainer.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 12px;
    `;

    const createStat = (label: string, value: string) => {
        const stat = document.createElement('div');
        stat.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        `;

        const labelSpan = document.createElement('span');
        labelSpan.textContent = label;
        labelSpan.style.cssText = `
            font-size: 13px;
            color: var(--text-secondary);
        `;

        const valueSpan = document.createElement('span');
        valueSpan.textContent = value;
        valueSpan.style.cssText = `
            font-size: 13px;
            font-weight: 500;
            color: var(--text-primary);
            font-family: var(--font-mono);
        `;

        stat.appendChild(labelSpan);
        stat.appendChild(valueSpan);
        return stat;
    };

    statsContainer.appendChild(createStat('Messages Analyzed', stats.messagesAnalyzed.toLocaleString()));
    statsContainer.appendChild(createStat('Total Tokens', stats.totalTokens.toLocaleString()));
    statsContainer.appendChild(createStat('Avg per Message', stats.averagePerMessage.toFixed(1)));

    const refreshButton = document.createElement('button');
    refreshButton.textContent = 'Analyze Messages';
    refreshButton.style.cssText = `
        width: 100%;
        padding: 10px;
        margin-top: 16px;
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: filter 0.15s;
    `;
    refreshButton.onmouseover = () => {
        refreshButton.style.filter = 'brightness(1.1)';
    };
    refreshButton.onmouseout = () => {
        refreshButton.style.filter = 'none';
    };
    refreshButton.onclick = () => {
        if (!context) return;

        const messages = context.data.getMessages();
        stats.messagesAnalyzed = messages.length;
        stats.totalTokens = messages.reduce((sum, m) => sum + estimateTokens(m.content), 0);
        stats.averagePerMessage = stats.messagesAnalyzed > 0
            ? stats.totalTokens / stats.messagesAnalyzed
            : 0;

        // Save stats
        context.storage.set('stats', stats);

        // Show toast
        context.ui.toast.success(`Analyzed ${messages.length} messages`);

        // Update display (re-render)
        statsContainer.innerHTML = '';
        statsContainer.appendChild(createStat('Messages Analyzed', stats.messagesAnalyzed.toLocaleString()));
        statsContainer.appendChild(createStat('Total Tokens', stats.totalTokens.toLocaleString()));
        statsContainer.appendChild(createStat('Avg per Message', stats.averagePerMessage.toFixed(1)));
    };

    container.appendChild(title);
    container.appendChild(statsContainer);
    container.appendChild(refreshButton);

    return container;
}

const plugin: PluginModule = {
    manifest: {
        id: 'token-counter',
        name: 'Token Counter',
        version: '1.0.0',
        description: 'Analyzes and displays token usage statistics for conversations',
        author: 'Mother-Harness Team',
        icon: 'calculator',
        capabilities: ['panel', 'command'],
        permissions: ['read:messages', 'storage'],
        keywords: ['tokens', 'analytics', 'statistics'],
    },

    activate() {
        console.log('Token Counter plugin activated');

        // Load saved stats
        const savedStats = context?.storage.get<TokenStats>('stats');
        if (savedStats) {
            stats = savedStats;
        }

        // Register panel
        context?.ui.registerPanel({
            id: 'token-stats',
            title: 'Token Stats',
            icon: 'calculator',
            render: createPanelContent,
        });

        // Register command
        context?.ui.registerCommand({
            id: 'analyze-tokens',
            title: 'Analyze Token Usage',
            description: 'Count tokens in current conversation',
            icon: 'zap',
            execute: () => {
                const messages = context?.data.getMessages() ?? [];
                const totalTokens = messages.reduce((sum, m) => sum + estimateTokens(m.content), 0);
                context?.ui.toast.info(`Estimated ${totalTokens} tokens in ${messages.length} messages`);
            },
        });

        context?.ui.toast.success('Token Counter activated');
    },

    deactivate() {
        console.log('Token Counter plugin deactivated');
        context?.ui.toast.info('Token Counter deactivated');
    },
};

// Export for direct import
export default plugin;

// Set context when registered
export function setContext(ctx: PluginContext) {
    context = ctx;
}
