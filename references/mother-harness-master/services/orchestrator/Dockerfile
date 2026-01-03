FROM node:20-alpine AS base

# Install pnpm
RUN corepack enable && corepack prepare pnpm@8.15.0 --activate

WORKDIR /app

# Copy workspace config
COPY pnpm-workspace.yaml ./
COPY package.json ./
COPY pnpm-lock.yaml* ./

# Copy shared library
COPY services/shared/package.json ./services/shared/
COPY services/shared/tsconfig.json ./services/shared/

# Copy agents library
COPY services/agents/package.json ./services/agents/
COPY services/agents/tsconfig.json ./services/agents/

# Copy orchestrator
COPY services/orchestrator/package.json ./services/orchestrator/
COPY services/orchestrator/tsconfig.json ./services/orchestrator/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Build stage
FROM base AS builder

COPY services/shared/src ./services/shared/src
COPY services/agents/src ./services/agents/src
COPY services/orchestrator/src ./services/orchestrator/src
COPY services/orchestrator/openapi.yaml ./services/orchestrator/
COPY tsconfig.json ./

RUN pnpm --filter @mother-harness/shared build
RUN pnpm --filter @mother-harness/agents build
RUN pnpm --filter @mother-harness/orchestrator build

# Production stage
FROM node:20-alpine AS production

RUN corepack enable && corepack prepare pnpm@8.15.0 --activate

WORKDIR /app

# Copy built artifacts
COPY --from=builder /app/package.json ./
COPY --from=builder /app/pnpm-workspace.yaml ./
COPY --from=builder /app/pnpm-lock.yaml ./
COPY --from=builder /app/services/shared/package.json ./services/shared/
COPY --from=builder /app/services/shared/dist ./services/shared/dist
COPY --from=builder /app/services/agents/package.json ./services/agents/
COPY --from=builder /app/services/agents/dist ./services/agents/dist
COPY --from=builder /app/services/orchestrator/package.json ./services/orchestrator/
COPY --from=builder /app/services/orchestrator/dist ./services/orchestrator/dist
COPY --from=builder /app/services/orchestrator/openapi.yaml ./services/orchestrator/

# Install production dependencies only
RUN pnpm install --prod --frozen-lockfile

EXPOSE 8000

WORKDIR /app/services/orchestrator

CMD ["node", "dist/server.js"]
