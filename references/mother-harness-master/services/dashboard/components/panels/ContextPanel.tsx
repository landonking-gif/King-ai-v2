/**
 * Context Panel Component
 * Shows current context files and documents
 */

'use client';

import React from 'react';
import { FileText, Code, Image, Database, X } from 'lucide-react';
import styles from './Panel.module.css';

const mockContextFiles = [
    { id: '1', name: 'README.md', type: 'document', size: 2400 },
    { id: '2', name: 'auth.ts', type: 'code', size: 4500 },
    { id: '3', name: 'schema.prisma', type: 'code', size: 1800 },
    { id: '4', name: 'architecture.png', type: 'image', size: 45000 },
];

const iconMap = {
    document: FileText,
    code: Code,
    image: Image,
    data: Database,
};

export function ContextPanel() {
    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        return `${(bytes / 1024).toFixed(1)} KB`;
    };

    return (
        <div className={styles.panel}>
            <div className={styles.header}>
                <h3 className={styles.title}>Context Files</h3>
                <span className={styles.count}>{mockContextFiles.length} files</span>
            </div>

            <div className={styles.list}>
                {mockContextFiles.map((file) => {
                    const Icon = iconMap[file.type as keyof typeof iconMap] || FileText;
                    return (
                        <div key={file.id} className={styles.item}>
                            <Icon size={16} className={styles.itemIcon} />
                            <div className={styles.itemContent}>
                                <span className={styles.itemName}>{file.name}</span>
                                <span className={styles.itemSize}>{formatSize(file.size)}</span>
                            </div>
                            <button className={styles.removeButton}>
                                <X size={14} />
                            </button>
                        </div>
                    );
                })}
            </div>

            <button className={styles.addButton}>
                + Add Context
            </button>
        </div>
    );
}
