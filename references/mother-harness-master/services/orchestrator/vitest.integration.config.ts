import { defineConfig } from 'vitest/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
    resolve: {
        alias: {
            '@mother-harness/shared': path.resolve(__dirname, '../shared/src/index.ts'),
            '@mother-harness/agents': path.resolve(__dirname, '../agents/src/index.ts'),
        },
    },
    test: {
        globals: true,
        environment: 'node',
        include: ['src/**/*.integration.test.ts'],
        testTimeout: 30000,
        hookTimeout: 30000,
    },
});
