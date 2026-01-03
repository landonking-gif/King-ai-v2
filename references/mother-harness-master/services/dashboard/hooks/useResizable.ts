/**
 * Resizable Panel Hook
 * Handles panel resizing with drag behavior
 */

'use client';

import { useState, useCallback, useEffect, useRef } from 'react';

interface UseResizableOptions {
    initialWidth: number;
    minWidth: number;
    maxWidth: number;
    direction: 'left' | 'right';
    onResize?: (width: number) => void;
}

export function useResizable({
    initialWidth,
    minWidth,
    maxWidth,
    direction,
    onResize,
}: UseResizableOptions) {
    const [width, setWidth] = useState(initialWidth);
    const [isResizing, setIsResizing] = useState(false);
    const startX = useRef(0);
    const startWidth = useRef(0);

    const handleMouseDown = useCallback(
        (e: React.MouseEvent) => {
            e.preventDefault();
            setIsResizing(true);
            startX.current = e.clientX;
            startWidth.current = width;
        },
        [width]
    );

    const handleMouseMove = useCallback(
        (e: MouseEvent) => {
            if (!isResizing) return;

            const diff = e.clientX - startX.current;
            const newWidth = direction === 'right'
                ? startWidth.current + diff
                : startWidth.current - diff;

            const clampedWidth = Math.min(maxWidth, Math.max(minWidth, newWidth));
            setWidth(clampedWidth);
            onResize?.(clampedWidth);
        },
        [isResizing, direction, minWidth, maxWidth, onResize]
    );

    const handleMouseUp = useCallback(() => {
        setIsResizing(false);
    }, []);

    useEffect(() => {
        if (isResizing) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
        }

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };
    }, [isResizing, handleMouseMove, handleMouseUp]);

    return {
        width,
        isResizing,
        handleMouseDown,
    };
}
