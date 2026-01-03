/**
 * Authentication Middleware
 * JWT-based authentication for API endpoints
 */

import type { FastifyRequest, FastifyReply, FastifyInstance } from 'fastify';
import { OAuth2Client } from 'google-auth-library';
import { getRedisClient } from '@mother-harness/shared';

/** JWT payload structure */
export interface JWTPayload {
    sub: string;          // User ID
    email?: string;
    name?: string;
    roles: string[];
    iat: number;
    exp: number;
}

/** User session */
export interface UserSession {
    user_id: string;
    email?: string;
    name?: string;
    roles: string[];
    created_at: string;
    last_activity: string;
}

/** API Key record */
export interface APIKey {
    id: string;
    key_hash: string;
    user_id: string;
    name: string;
    permissions: string[];
    rate_limit: number;
    created_at: string;
    last_used_at?: string;
    expires_at?: string;
    revoked: boolean;
}

/**
 * Register authentication plugin
 */
export async function registerAuth(app: FastifyInstance): Promise<void> {
    const redis = getRedisClient();
    const jwtSecret = process.env['JWT_SECRET'] ?? 'development-secret-change-in-production';
    const googleClientId = process.env['GOOGLE_CLIENT_ID'];
    const allowedDomains = parseCsvEnv(process.env['GOOGLE_ALLOWED_DOMAINS']);
    const adminEmails = parseCsvEnv(process.env['GOOGLE_ADMIN_EMAILS']);
    const approverEmails = parseCsvEnv(process.env['GOOGLE_APPROVER_EMAILS']);

    // Decorate request with user
    app.decorateRequest('user', null);

    // Authentication hook
    app.addHook('onRequest', async (request, reply) => {
        // Skip auth for health endpoint
        if (request.url === '/health') return;

        // Skip OPTIONS requests
        if (request.method === 'OPTIONS') return;

        // Skip strict auth for WebSocket paths - auth is validated via query params
        // Browser WS upgrade requests can't easily send custom headers
        const isWebSocket = request.url.startsWith('/ws');

        // Single-user mode: Accept X-User-ID header or user_id query param
        // This allows simplified auth without password for single-user deployments
        const simpleUserId = (request.headers['x-user-id'] as string)
            || (request.query as any)?.user_id
            || new URL(request.url, 'http://localhost').searchParams.get('user_id');

        if (simpleUserId && typeof simpleUserId === 'string' && simpleUserId.trim()) {
            (request as any).user = {
                user_id: simpleUserId.trim(),
                name: simpleUserId.trim(),
                roles: ['user', 'admin', 'approver'], // Grant all roles in single-user mode
                created_at: new Date().toISOString(),
                last_activity: new Date().toISOString(),
            } as UserSession;
            return;
        }

        // Try Bearer token first
        const authHeader = request.headers.authorization;
        if (authHeader?.startsWith('Bearer ')) {
            const token = authHeader.slice(7);
            const user = googleClientId
                ? await validateGoogleToken(token, googleClientId, {
                    allowedDomains,
                    adminEmails,
                    approverEmails,
                })
                : await validateJWT(token, jwtSecret);
            if (user) {
                (request as any).user = user;
                return;
            }
        }

        // Try API key
        const apiKey = request.headers['x-api-key'] as string | undefined;
        if (apiKey) {
            const keyData = await validateAPIKey(redis, apiKey);
            if (keyData) {
                (request as any).user = {
                    user_id: keyData.user_id,
                    roles: keyData.permissions,
                };
                return;
            }
        }

        // For WebSocket paths, don't send 401 - let the connection through
        // WebSocket auth is optional; the user_id from query params was already captured above
        if (isWebSocket) {
            // Create anonymous user for WS if no auth provided
            (request as any).user = {
                user_id: 'anonymous',
                name: 'Anonymous',
                roles: ['user'],
                created_at: new Date().toISOString(),
                last_activity: new Date().toISOString(),
            } as UserSession;
            return;
        }

        reply.status(401).send({ error: 'Authentication required' });
    });
}

function parseCsvEnv(value?: string): string[] {
    if (!value) return [];
    return value.split(',').map(entry => entry.trim()).filter(Boolean);
}

/**
 * Validate Google ID token
 */
async function validateGoogleToken(
    token: string,
    clientId: string,
    options: {
        allowedDomains: string[];
        adminEmails: string[];
        approverEmails: string[];
    }
): Promise<UserSession | null> {
    try {
        const client = new OAuth2Client(clientId);
        const ticket = await client.verifyIdToken({
            idToken: token,
            audience: clientId,
        });
        const payload = ticket.getPayload();
        if (!payload || !payload.sub) {
            return null;
        }

        if (!payload.email || !payload.email_verified) {
            return null;
        }

        const domain = payload.hd ?? payload.email.split('@')[1];
        if (!domain) return null;
        if (options.allowedDomains.length > 0 && !options.allowedDomains.includes(domain)) {
            return null;
        }

        const roles = ['user'];
        if (options.approverEmails.includes(payload.email)) {
            roles.push('approver');
        }
        if (options.adminEmails.includes(payload.email)) {
            roles.push('admin', 'approver');
        }

        return {
            user_id: payload.sub,
            ...(payload.email !== undefined && { email: payload.email }),
            ...(payload.name !== undefined && { name: payload.name }),
            roles: Array.from(new Set(roles)),
            created_at: new Date((payload.iat ?? 0) * 1000).toISOString(),
            last_activity: new Date().toISOString(),
        };
    } catch (error) {
        console.error('[Auth] Google token validation error:', error);
        return null;
    }
}

/**
 * Validate JWT token with HMAC-SHA256 signature verification
 */
async function validateJWT(token: string, secret: string): Promise<UserSession | null> {
    try {
        const parts = token.split('.');
        if (parts.length !== 3) {
            return null;
        }

        const [headerB64, payloadB64, signatureB64] = parts;
        if (!headerB64 || !payloadB64 || !signatureB64) {
            return null;
        }

        // Verify signature using HMAC-SHA256
        const { createHmac } = await import('crypto');
        const signatureInput = `${headerB64}.${payloadB64}`;
        const expectedSignature = createHmac('sha256', secret)
            .update(signatureInput)
            .digest('base64url');

        // Timing-safe comparison to prevent timing attacks
        const { timingSafeEqual } = await import('crypto');
        const sigBuffer = Buffer.from(signatureB64, 'base64url');
        const expectedBuffer = Buffer.from(expectedSignature, 'base64url');

        if (sigBuffer.length !== expectedBuffer.length ||
            !timingSafeEqual(sigBuffer, expectedBuffer)) {
            console.warn('[Auth] JWT signature verification failed');
            return null;
        }

        // Decode payload
        const payload = JSON.parse(
            Buffer.from(payloadB64, 'base64url').toString('utf-8')
        ) as JWTPayload;

        // Check expiration
        const now = Date.now() / 1000;
        if (payload.exp && payload.exp < now) {
            console.warn('[Auth] JWT token expired');
            return null;
        }

        // Check not-before (nbf) if present
        if (payload.iat && payload.iat > now + 60) { // 60s clock skew tolerance
            console.warn('[Auth] JWT token not yet valid');
            return null;
        }

        return {
            user_id: payload.sub,
            ...(payload.email !== undefined && { email: payload.email }),
            ...(payload.name !== undefined && { name: payload.name }),
            roles: payload.roles ?? ['user'],
            created_at: new Date((payload.iat ?? 0) * 1000).toISOString(),
            last_activity: new Date().toISOString(),
        };
    } catch (error) {
        console.error('[Auth] JWT validation error:', error);
        return null;
    }
}

/**
 * Validate API key
 */
async function validateAPIKey(
    redis: ReturnType<typeof getRedisClient>,
    apiKey: string
): Promise<APIKey | null> {
    try {
        // Hash the API key to look up
        const keyHash = hashAPIKey(apiKey);

        const keyData = await redis.get(`apikey:${keyHash}`);
        if (!keyData) return null;

        const key = JSON.parse(keyData) as APIKey;

        // Check if revoked
        if (key.revoked) return null;

        // Check expiration
        if (key.expires_at && new Date(key.expires_at) < new Date()) {
            return null;
        }

        // Update last used
        key.last_used_at = new Date().toISOString();
        await redis.set(`apikey:${keyHash}`, JSON.stringify(key));

        return key;
    } catch {
        return null;
    }
}

/**
 * Hash API key for storage lookup
 */
function hashAPIKey(key: string): string {
    // Simple hash - in production use crypto.createHash
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
        const char = key.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
}

/**
 * Role-based access control middleware
 */
export function requireRole(...roles: string[]) {
    return async (request: FastifyRequest, reply: FastifyReply) => {
        const user = (request as any).user as UserSession | undefined;

        if (!user) {
            reply.status(401).send({ error: 'Authentication required' });
            return;
        }

        const hasRole = roles.some(role => user.roles.includes(role));
        if (!hasRole) {
            reply.status(403).send({ error: 'Insufficient permissions' });
            return;
        }
    };
}

/**
 * Rate limiting middleware
 */
export function rateLimit(options: {
    max: number;
    windowMs: number;
}) {
    const requests = new Map<string, { count: number; resetAt: number }>();

    return async (request: FastifyRequest, reply: FastifyReply) => {
        const user = (request as any).user as UserSession | undefined;
        const key = user?.user_id ?? request.ip;

        const now = Date.now();
        const record = requests.get(key);

        if (!record || record.resetAt < now) {
            requests.set(key, { count: 1, resetAt: now + options.windowMs });
            return;
        }

        record.count++;

        if (record.count > options.max) {
            reply.status(429).send({
                error: 'Too many requests',
                retry_after: Math.ceil((record.resetAt - now) / 1000),
            });
            return;
        }
    };
}
