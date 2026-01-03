/**
 * JWT Token Generation
 * Utility for generating signed JWT tokens
 */

import { createHmac, randomBytes } from 'crypto';

/** JWT Payload */
interface JWTPayload {
    sub: string;          // User ID
    email?: string;
    name?: string;
    roles: string[];
    iat: number;
    exp: number;
}

/** Token options */
interface TokenOptions {
    expiresIn?: number;   // Seconds until expiration (default: 24 hours)
    secret?: string;      // JWT secret (default: from env)
}

/**
 * Generate a signed JWT token
 */
export function generateToken(
    userId: string,
    claims: { email?: string; name?: string; roles?: string[] } = {},
    options: TokenOptions = {}
): string {
    const secret = options.secret ?? process.env['JWT_SECRET'] ?? 'development-secret-change-in-production';
    const expiresIn = options.expiresIn ?? 24 * 60 * 60; // 24 hours default

    const now = Math.floor(Date.now() / 1000);

    const header = {
        alg: 'HS256',
        typ: 'JWT',
    };

    const payload: JWTPayload = {
        sub: userId,
        ...(claims.email !== undefined && { email: claims.email }),
        ...(claims.name !== undefined && { name: claims.name }),
        roles: claims.roles ?? ['user'],
        iat: now,
        exp: now + expiresIn,
    };

    // Encode header and payload
    const headerB64 = Buffer.from(JSON.stringify(header)).toString('base64url');
    const payloadB64 = Buffer.from(JSON.stringify(payload)).toString('base64url');

    // Sign with HMAC-SHA256
    const signatureInput = `${headerB64}.${payloadB64}`;
    const signature = createHmac('sha256', secret)
        .update(signatureInput)
        .digest('base64url');

    return `${headerB64}.${payloadB64}.${signature}`;
}

/**
 * Generate a refresh token
 */
export function generateRefreshToken(): string {
    return randomBytes(32).toString('hex');
}

/**
 * User credentials (would come from database in production)
 */
const DEMO_USERS: Record<string, { password: string; name: string; roles: string[] }> = {
    'admin@mother-harness.local': {
        password: 'admin123',  // In production, use bcrypt hashes
        name: 'Admin User',
        roles: ['admin', 'user'],
    },
    'user@mother-harness.local': {
        password: 'user123',
        name: 'Demo User',
        roles: ['user'],
    },
};

/**
 * Validate credentials and return token
 */
export function authenticateUser(
    email: string,
    password: string
): { token: string; refreshToken: string } | null {
    const user = DEMO_USERS[email.toLowerCase()];

    if (!user || user.password !== password) {
        return null;
    }

    const userId = `user-${email.toLowerCase().split('@')[0]}`;

    return {
        token: generateToken(
            userId,
            {
                ...(email !== undefined && { email }),
                ...(user.name !== undefined && { name: user.name }),
                ...(user.roles !== undefined && { roles: user.roles }),
            }
        ),
        refreshToken: generateRefreshToken(),
    };
}

/**
 * Create login route handler
 */
export function createLoginHandler() {
    return async (request: { body: { email?: string; password?: string } }) => {
        const { email, password } = request.body;

        if (!email || !password) {
            return { error: 'Email and password required', statusCode: 400 };
        }

        const result = authenticateUser(email, password);

        if (!result) {
            return { error: 'Invalid credentials', statusCode: 401 };
        }

        return {
            token: result.token,
            refreshToken: result.refreshToken,
            expiresIn: 24 * 60 * 60,
        };
    };
}

/**
 * Create logout route handler
 */
export function createLogoutHandler() {
    return async () => {
        // In production, invalidate refresh token in Redis
        return { success: true };
    };
}
