/**
 * Files Panel Component
 * Shows project files
 */

'use client';

import React from 'react';
import { Folder, FileCode, FileText, ChevronRight } from 'lucide-react';
import styles from './Panel.module.css';

const mockFiles = [
    { id: '1', name: 'src', type: 'folder', children: 12 },
    { id: '2', name: 'lib', type: 'folder', children: 5 },
    { id: '3', name: 'package.json', type: 'file' },
    { id: '4', name: 'tsconfig.json', type: 'file' },
    { id: '5', name: 'README.md', type: 'file' },
];

export function FilesPanel() {
    return (
        <div className={styles.panel}>
            <div className={styles.header}>
                <h3 className={styles.title}>Project Files</h3>
            </div>

            <div className={styles.list}>
                {mockFiles.map((file) => (
                    <button key={file.id} className={styles.fileItem}>
                        {file.type === 'folder' ? (
                            <>
                                <ChevronRight size={14} className={styles.chevron} />
                                <Folder size={16} className={styles.folderIcon} />
                            </>
                        ) : (
                            <>
                                <span className={styles.spacer} />
                                {file.name.endsWith('.json') || file.name.endsWith('.ts') ? (
                                    <FileCode size={16} className={styles.fileIcon} />
                                ) : (
                                    <FileText size={16} className={styles.fileIcon} />
                                )}
                            </>
                        )}
                        <span className={styles.fileName}>{file.name}</span>
                        {file.type === 'folder' && (
                            <span className={styles.fileCount}>{file.children}</span>
                        )}
                    </button>
                ))}
            </div>
        </div>
    );
}
