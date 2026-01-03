/**
 * Welcome Prompt Component
 * Shows on first visit to collect user's name/nickname
 */

'use client';

import React, { useState } from 'react';
import { useIdentityStore } from '../../stores/identityStore';
import styles from './WelcomePrompt.module.css';

export function WelcomePrompt() {
    const [name, setName] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const setIdentity = useIdentityStore((s) => s.setIdentity);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!name.trim()) return;

        setIsSubmitting(true);
        await setIdentity(name);
        setIsSubmitting(false);
    };

    return (
        <div className={styles.overlay}>
            <div className={styles.card}>
                <div className={styles.logo}>
                    <span className={styles.logoIcon}>⚡</span>
                </div>

                <h1 className={styles.title}>Welcome to Mother-Harness</h1>
                <p className={styles.subtitle}>
                    Your multi-agent AI assistant. What should I call you?
                </p>

                <form onSubmit={handleSubmit} className={styles.form}>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Enter your name or nickname"
                        className={styles.input}
                        autoFocus
                        maxLength={50}
                    />

                    <button
                        type="submit"
                        className={styles.button}
                        disabled={!name.trim() || isSubmitting}
                    >
                        {isSubmitting ? 'Getting started...' : "Let's go"}
                    </button>
                </form>

                <p className={styles.note}>
                    This helps personalize your experience. You can change it anytime.
                </p>
            </div>
        </div>
    );
}
