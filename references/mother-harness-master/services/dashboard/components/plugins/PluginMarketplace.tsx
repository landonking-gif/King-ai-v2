/**
 * Plugin Marketplace Panel
 * Browse, install, and manage plugins
 */

'use client';

import React, { useState } from 'react';
import {
    Package,
    Download,
    Trash2,
    Power,
    PowerOff,
    ExternalLink,
    Search,
    Filter,
    CheckCircle,
    AlertCircle,
    Loader2,
} from 'lucide-react';
import { usePlugins, usePlugin } from '../../lib/plugins/hooks';
import type { PluginInstance, PluginManifest } from '../../lib/plugins/types';
import styles from './PluginMarketplace.module.css';

// Mock marketplace plugins for demo
const MARKETPLACE_PLUGINS: PluginManifest[] = [
    {
        id: 'token-counter',
        name: 'Token Counter',
        version: '1.0.0',
        description: 'Analyzes and displays token usage statistics',
        author: 'Mother-Harness Team',
        icon: 'calculator',
        capabilities: ['panel', 'command'],
        permissions: ['read:messages', 'storage'],
        keywords: ['tokens', 'analytics'],
    },
    {
        id: 'code-highlighter',
        name: 'Code Highlighter Plus',
        version: '2.1.0',
        description: 'Enhanced syntax highlighting for 50+ languages',
        author: 'DevTools Inc',
        icon: 'code',
        capabilities: ['message-action'],
        permissions: ['read:messages'],
        keywords: ['code', 'syntax', 'highlighting'],
    },
    {
        id: 'agent-metrics',
        name: 'Agent Metrics Dashboard',
        version: '1.2.0',
        description: 'Real-time performance metrics and charts for agents',
        author: 'Analytics Co',
        icon: 'bar-chart-2',
        capabilities: ['panel'],
        permissions: ['read:agents'],
        keywords: ['metrics', 'analytics', 'charts'],
    },
    {
        id: 'quick-commands',
        name: 'Quick Commands',
        version: '1.0.0',
        description: 'Add custom slash commands for common tasks',
        author: 'Productivity Labs',
        icon: 'terminal',
        capabilities: ['command'],
        permissions: ['write:messages'],
        keywords: ['commands', 'productivity'],
    },
];

interface PluginCardProps {
    manifest: PluginManifest;
    installed?: PluginInstance;
}

function PluginCard({ manifest, installed }: PluginCardProps) {
    const { activate, deactivate, uninstall, loading } = usePlugin(manifest.id);
    const isActive = installed?.status === 'active';
    const hasError = installed?.status === 'error';

    const handleToggle = async () => {
        if (isActive) {
            await deactivate();
        } else {
            await activate();
        }
    };

    const handleInstall = async () => {
        // In a real implementation, this would fetch and register the plugin
        console.log('Would install:', manifest.id);
    };

    return (
        <div className={`${styles.card} ${hasError ? styles.error : ''}`}>
            <div className={styles.cardHeader}>
                <div className={styles.iconWrapper}>
                    <Package size={20} />
                </div>
                <div className={styles.cardInfo}>
                    <h3 className={styles.cardTitle}>{manifest.name}</h3>
                    <span className={styles.cardVersion}>v{manifest.version}</span>
                </div>
                {installed && (
                    <div className={styles.statusBadge}>
                        {isActive ? (
                            <>
                                <CheckCircle size={12} />
                                Active
                            </>
                        ) : hasError ? (
                            <>
                                <AlertCircle size={12} />
                                Error
                            </>
                        ) : (
                            'Installed'
                        )}
                    </div>
                )}
            </div>

            <p className={styles.cardDescription}>{manifest.description}</p>

            <div className={styles.cardMeta}>
                <span className={styles.author}>by {manifest.author}</span>
                <div className={styles.capabilities}>
                    {manifest.capabilities.slice(0, 2).map((cap) => (
                        <span key={cap} className={styles.capability}>
                            {cap}
                        </span>
                    ))}
                </div>
            </div>

            {hasError && installed?.error && (
                <div className={styles.errorMessage}>
                    <AlertCircle size={14} />
                    {installed.error}
                </div>
            )}

            <div className={styles.cardActions}>
                {installed ? (
                    <>
                        <button
                            className={`${styles.actionButton} ${isActive ? styles.deactivate : styles.activate}`}
                            onClick={handleToggle}
                            disabled={loading}
                        >
                            {loading ? (
                                <Loader2 size={14} className={styles.spinner} />
                            ) : isActive ? (
                                <PowerOff size={14} />
                            ) : (
                                <Power size={14} />
                            )}
                            {isActive ? 'Deactivate' : 'Activate'}
                        </button>
                        <button
                            className={`${styles.actionButton} ${styles.uninstall}`}
                            onClick={uninstall}
                            disabled={loading}
                        >
                            <Trash2 size={14} />
                        </button>
                    </>
                ) : (
                    <button
                        className={`${styles.actionButton} ${styles.install}`}
                        onClick={handleInstall}
                    >
                        <Download size={14} />
                        Install
                    </button>
                )}
                {manifest.homepage && (
                    <a
                        href={manifest.homepage}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={styles.externalLink}
                    >
                        <ExternalLink size={14} />
                    </a>
                )}
            </div>
        </div>
    );
}

export function PluginMarketplace() {
    const installedPlugins = usePlugins();
    const [searchQuery, setSearchQuery] = useState('');
    const [filter, setFilter] = useState<'all' | 'installed'>('all');

    const installedIds = new Set(installedPlugins.map((p) => p.manifest.id));

    const filteredPlugins = MARKETPLACE_PLUGINS.filter((plugin) => {
        // Filter by installed status
        if (filter === 'installed' && !installedIds.has(plugin.id)) {
            return false;
        }

        // Filter by search query
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            return (
                plugin.name.toLowerCase().includes(query) ||
                plugin.description.toLowerCase().includes(query) ||
                plugin.keywords?.some((k) => k.toLowerCase().includes(query))
            );
        }

        return true;
    });

    return (
        <div className={styles.marketplace}>
            <div className={styles.header}>
                <h2 className={styles.title}>
                    <Package size={20} />
                    Plugins
                </h2>
                <span className={styles.count}>
                    {installedPlugins.length} installed
                </span>
            </div>

            <div className={styles.controls}>
                <div className={styles.searchWrapper}>
                    <Search size={16} className={styles.searchIcon} />
                    <input
                        type="text"
                        placeholder="Search plugins..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className={styles.searchInput}
                    />
                </div>
                <div className={styles.filterWrapper}>
                    <Filter size={14} />
                    <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value as 'all' | 'installed')}
                        className={styles.filterSelect}
                    >
                        <option value="all">All Plugins</option>
                        <option value="installed">Installed</option>
                    </select>
                </div>
            </div>

            <div className={styles.grid}>
                {filteredPlugins.map((manifest) => (
                    <PluginCard
                        key={manifest.id}
                        manifest={manifest}
                        installed={installedPlugins.find((p) => p.manifest.id === manifest.id)}
                    />
                ))}
            </div>

            {filteredPlugins.length === 0 && (
                <div className={styles.empty}>
                    <Package size={32} className={styles.emptyIcon} />
                    <p>No plugins found</p>
                </div>
            )}
        </div>
    );
}
