# syntax=docker/dockerfile:1.4

# ============================================
# Dashboard Dockerfile
# Multi-stage build for Next.js application
# ============================================

FROM node:20-alpine AS base
RUN corepack enable pnpm

# ============================================
# Builder Stage
# ============================================
FROM base AS builder
WORKDIR /app

# Copy workspace config
COPY pnpm-workspace.yaml ./
COPY package.json ./
COPY pnpm-lock.yaml ./

# Copy shared library
COPY services/shared/package.json ./services/shared/
COPY services/shared/tsconfig.json ./services/shared/

# Copy dashboard
COPY services/dashboard/package.json ./services/dashboard/
COPY services/dashboard/tsconfig.json ./services/dashboard/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy source files
COPY tsconfig.json ./
COPY services/shared/src ./services/shared/src
COPY services/dashboard/app ./services/dashboard/app
COPY services/dashboard/components ./services/dashboard/components
COPY services/dashboard/lib ./services/dashboard/lib
COPY services/dashboard/stores ./services/dashboard/stores
COPY services/dashboard/types ./services/dashboard/types
COPY services/dashboard/hooks ./services/dashboard/hooks
COPY services/dashboard/plugins ./services/dashboard/plugins
COPY services/dashboard/next.config.js ./services/dashboard/
COPY services/dashboard/next-env.d.ts ./services/dashboard/

# Build shared library first
RUN pnpm --filter @mother-harness/shared build

# Build dashboard
RUN pnpm --filter @mother-harness/dashboard build

# ============================================
# Production Stage
# ============================================
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy standalone build
COPY --from=builder /app/services/dashboard/.next/standalone ./
COPY --from=builder /app/services/dashboard/.next/static ./services/dashboard/.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "services/dashboard/server.js"]
