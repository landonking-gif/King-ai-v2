/**
 * Identity Provider
 * Shows welcome prompt if user hasn't set identity
 * Wraps app with identity context
 */

'use client';

import React, { useState, useEffect } from 'react';
import { useIdentityStore } from '../../stores/identityStore';
import { WelcomePrompt } from './WelcomePrompt';

interface IdentityProviderProps {
    children: React.ReactNode;
}

export function IdentityProvider({ children }: IdentityProviderProps) {
    // Wait for hydration to complete (localStorage only available on client)
    const [isHydrated, setIsHydrated] = useState(false);
    const identity = useIdentityStore((s) => s.identity);

    useEffect(() => {
        setIsHydrated(true);
    }, []);

    // Show nothing while hydrating (prevents flash)
    if (!isHydrated) {
        return null;
    }

    // Show welcome prompt if no identity
    if (!identity) {
        return <WelcomePrompt />;
    }

    // Identity exists, render app
    return <>{children}</>;
}
