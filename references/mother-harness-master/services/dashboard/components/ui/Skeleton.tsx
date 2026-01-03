/**
 * Skeleton Loading Components
 * Animated placeholder components for loading states
 */

'use client';

import React from 'react';
import styles from './Skeleton.module.css';

interface SkeletonProps {
    className?: string;
}

// Base skeleton with shimmer animation
export function Skeleton({ className = '' }: SkeletonProps) {
    return <div className={`${styles.skeleton} ${className}`} />;
}

// Text line skeleton
interface SkeletonTextProps extends SkeletonProps {
    lines?: number;
    width?: string;
}

export function SkeletonText({ lines = 1, width = '100%', className = '' }: SkeletonTextProps) {
    return (
        <div className={`${styles.textContainer} ${className}`}>
            {Array.from({ length: lines }).map((_, i) => (
                <div
                    key={i}
                    className={styles.textLine}
                    style={{
                        width: i === lines - 1 && lines > 1 ? '70%' : width,
                    }}
                />
            ))}
        </div>
    );
}

// Avatar skeleton
interface SkeletonAvatarProps extends SkeletonProps {
    size?: number;
}

export function SkeletonAvatar({ size = 40, className = '' }: SkeletonAvatarProps) {
    return (
        <div
            className={`${styles.avatar} ${className}`}
            style={{ width: size, height: size }}
        />
    );
}

// Message skeleton (full chat message placeholder)
export function SkeletonMessage({ className = '' }: SkeletonProps) {
    return (
        <div className={`${styles.message} ${className}`}>
            <SkeletonAvatar size={36} />
            <div className={styles.messageContent}>
                <div className={styles.messageHeader}>
                    <div className={styles.messageName} />
                    <div className={styles.messageTime} />
                </div>
                <SkeletonText lines={3} />
            </div>
        </div>
    );
}

// Card skeleton
export function SkeletonCard({ className = '' }: SkeletonProps) {
    return (
        <div className={`${styles.card} ${className}`}>
            <div className={styles.cardHeader}>
                <SkeletonAvatar size={32} />
                <SkeletonText lines={1} width="60%" />
            </div>
            <SkeletonText lines={2} />
        </div>
    );
}

// Panel skeleton (for right panel tabs)
export function SkeletonPanel({ className = '' }: SkeletonProps) {
    return (
        <div className={`${styles.panel} ${className}`}>
            {Array.from({ length: 3 }).map((_, i) => (
                <SkeletonCard key={i} />
            ))}
        </div>
    );
}

// Button skeleton
interface SkeletonButtonProps extends SkeletonProps {
    width?: string;
}

export function SkeletonButton({ width = '100px', className = '' }: SkeletonButtonProps) {
    return <div className={`${styles.button} ${className}`} style={{ width }} />;
}
