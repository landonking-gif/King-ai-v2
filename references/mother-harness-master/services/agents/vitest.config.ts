import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

export default defineConfig({
    test: {
        globals: true,
        environment: 'node',
    },
    resolve: {
        alias: {
            // Allow workspace tests to run without building @mother-harness/shared first.
            '@mother-harness/shared': fileURLToPath(new URL('../shared/src/index.ts', import.meta.url)),
        },
    },
});
