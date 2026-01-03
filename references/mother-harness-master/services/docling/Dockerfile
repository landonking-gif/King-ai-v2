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

# Copy docling
COPY services/docling/package.json ./services/docling/
COPY services/docling/tsconfig.json ./services/docling/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Build stage
FROM base AS builder

COPY services/shared/src ./services/shared/src
COPY services/docling/src ./services/docling/src
COPY tsconfig.json ./

RUN pnpm --filter @mother-harness/shared build
RUN pnpm --filter @mother-harness/docling build

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
COPY --from=builder /app/services/docling/package.json ./services/docling/
COPY --from=builder /app/services/docling/dist ./services/docling/dist

# Install production dependencies only
RUN pnpm install --prod --frozen-lockfile

WORKDIR /app/services/docling

CMD ["node", "dist/index.js"]

