'use client';

import './globals.css';
import React from 'react';
import { IdentityProvider } from '../components/identity/IdentityProvider';
import { EventStreamProvider } from '../components/providers/EventStreamProvider';
import { ErrorBoundary } from '../components/providers/ErrorBoundary';
import { ToastProvider } from '../components/ui/Toast';
import { useUIStore } from '../stores/uiStore';

function ThemeWrapper({ children }: { children: React.ReactNode }) {
    const theme = useUIStore((s) => s.theme);

    React.useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    return <>{children}</>;
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" data-theme="dark">
            <head>
                <title>Mother-Harness | Mission Control</title>
                <meta name="description" content="Multi-agent orchestration dashboard" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <link
                    href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
                    rel="stylesheet"
                />
            </head>
            <body>
                <ErrorBoundary>
                    <ToastProvider>
                        <IdentityProvider>
                            <EventStreamProvider>
                                <ThemeWrapper>
                                    {children}
                                </ThemeWrapper>
                            </EventStreamProvider>
                        </IdentityProvider>
                    </ToastProvider>
                </ErrorBoundary>
            </body>
        </html>
    );
}


