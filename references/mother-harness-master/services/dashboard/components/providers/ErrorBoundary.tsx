/**
 * Error Boundary Component
 * Catches React rendering errors and displays a fallback UI
 * Reports errors to telemetry
 */

'use client';

import React, { Component, type ReactNode, type ErrorInfo } from 'react';

// Simple inline error logging (telemetry module will be loaded dynamically to avoid build issues)
function logErrorToConsole(error: Error, context: Record<string, unknown>): void {
    console.error('[ErrorBoundary] Error:', {
        name: error.name,
        message: error.message,
        stack: error.stack,
        ...context,
        timestamp: new Date().toISOString(),
    });
}

interface ErrorBoundaryProps {
    children: ReactNode;
    fallback?: ReactNode;
    onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
    hasError: boolean;
    error: Error | null;
    errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = {
            hasError: false,
            error: null,
            errorInfo: null,
        };
    }

    static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
        this.setState({ errorInfo });

        // Log to console
        logErrorToConsole(error, {
            componentStack: errorInfo.componentStack,
            source: 'ErrorBoundary',
        });

        // Call custom error handler if provided
        this.props.onError?.(error, errorInfo);
    }

    handleRetry = (): void => {
        this.setState({
            hasError: false,
            error: null,
            errorInfo: null,
        });
    };

    render(): ReactNode {
        if (this.state.hasError) {
            // Use custom fallback if provided
            if (this.props.fallback) {
                return this.props.fallback;
            }

            // Default fallback UI
            return (
                <div style={{
                    padding: '2rem',
                    textAlign: 'center',
                    color: 'var(--text-secondary)',
                }}>
                    <h2 style={{
                        color: 'var(--error)',
                        marginBottom: '1rem',
                    }}>
                        Something went wrong
                    </h2>
                    <p style={{ marginBottom: '1rem' }}>
                        {this.state.error?.message || 'An unexpected error occurred'}
                    </p>
                    <button
                        onClick={this.handleRetry}
                        style={{
                            padding: '0.5rem 1rem',
                            background: 'var(--accent-primary)',
                            color: 'white',
                            border: 'none',
                            borderRadius: 'var(--radius-md)',
                            cursor: 'pointer',
                        }}
                    >
                        Try Again
                    </button>
                    {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                        <details style={{
                            marginTop: '1rem',
                            textAlign: 'left',
                            fontSize: '0.75rem',
                            color: 'var(--text-tertiary)',
                        }}>
                            <summary>Error Details</summary>
                            <pre style={{
                                overflow: 'auto',
                                background: 'var(--bg-tertiary)',
                                padding: '1rem',
                                borderRadius: 'var(--radius-md)',
                            }}>
                                {this.state.error?.stack}
                                {'\n\nComponent Stack:'}
                                {this.state.errorInfo.componentStack}
                            </pre>
                        </details>
                    )}
                </div>
            );
        }

        return this.props.children;
    }
}

/**
 * Hook-friendly error boundary wrapper
 */
export function withErrorBoundary<P extends object>(
    WrappedComponent: React.ComponentType<P>,
    fallback?: ReactNode
): React.FC<P> {
    return function WithErrorBoundary(props: P) {
        return (
            <ErrorBoundary fallback={fallback}>
                <WrappedComponent {...props} />
            </ErrorBoundary>
        );
    };
}
