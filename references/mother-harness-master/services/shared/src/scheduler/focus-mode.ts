/**
 * Focus Mode
 * Context-aware agent behavior modification
 */

import { getRedisJSON } from '../redis/index.js';
import { nanoid } from 'nanoid';

/** Focus mode definition */
export interface FocusMode {
    id: string;
    name: string;
    description: string;

    // Agent modifications
    agent_configs: Array<{
        agent: string;
        enabled: boolean;
        priority_boost?: number;      // Adjust priority
        model_override?: string;      // Use specific model
        temperature_override?: number;
        system_prompt_prefix?: string;
        system_prompt_suffix?: string;
    }>;

    // Library/resource focus
    preferred_libraries?: string[];
    excluded_libraries?: string[];

    // Behavior
    response_style?: 'concise' | 'detailed' | 'technical' | 'simple';
    approval_threshold?: number;    // Require approval above this score

    // Access
    owner_id: string;
    shared: boolean;

    // Metadata
    created_at: string;
    updated_at: string;
}

/** Active session with focus mode */
export interface FocusSession {
    id: string;
    user_id: string;
    focus_mode_id: string;
    started_at: string;
    ends_at?: string;
    active: boolean;
}

export class FocusModeManager {
    private redis = getRedisJSON();
    private readonly modePrefix = 'focus_mode:';
    private readonly sessionPrefix = 'focus_session:';

    /**
     * Create a focus mode
     */
    async createFocusMode(
        ownerId: string,
        data: Omit<FocusMode, 'id' | 'owner_id' | 'created_at' | 'updated_at'>
    ): Promise<FocusMode> {
        const now = new Date().toISOString();

        const mode: FocusMode = {
            ...data,
            id: `focus-${nanoid()}`,
            owner_id: ownerId,
            created_at: now,
            updated_at: now,
        };

        await this.redis.set(`${this.modePrefix}${mode.id}`, '$', mode);

        return mode;
    }

    /**
     * Get a focus mode
     */
    async getFocusMode(modeId: string): Promise<FocusMode | null> {
        return await this.redis.get<FocusMode>(`${this.modePrefix}${modeId}`);
    }

    /**
     * Start a focus session
     */
    async startSession(
        userId: string,
        modeId: string,
        duration_hours?: number
    ): Promise<FocusSession> {
        // End any existing sessions
        await this.endActiveSession(userId);

        const now = new Date();
        const endsAt = duration_hours
            ? new Date(now.getTime() + duration_hours * 60 * 60 * 1000).toISOString()
            : undefined;

        const session: FocusSession = {
            id: `session-${nanoid()}`,
            user_id: userId,
            focus_mode_id: modeId,
            started_at: now.toISOString(),
            ...(endsAt !== undefined && { ends_at: endsAt }),
            active: true,
        };

        await this.redis.set(`${this.sessionPrefix}${session.id}`, '$', session);
        await this.redis.set(`${this.sessionPrefix}active:${userId}`, '$', session);

        return session;
    }

    /**
     * Get active focus session for user
     */
    async getActiveSession(userId: string): Promise<FocusSession | null> {
        const session = await this.redis.get<FocusSession>(`${this.sessionPrefix}active:${userId}`);

        if (!session || !session.active) return null;

        // Check if expired
        if (session.ends_at && new Date(session.ends_at) < new Date()) {
            await this.endActiveSession(userId);
            return null;
        }

        return session;
    }

    /**
     * End active session
     */
    async endActiveSession(userId: string): Promise<void> {
        const session = await this.getActiveSession(userId);
        if (!session) return;

        session.active = false;
        await this.redis.set(`${this.sessionPrefix}${session.id}`, '$', session);
        await this.redis.del(`${this.sessionPrefix}active:${userId}`);
    }

    /**
     * Get agent configuration for active focus mode
     */
    async getAgentConfig(
        userId: string,
        agent: string
    ): Promise<FocusMode['agent_configs'][number] | null> {
        const session = await this.getActiveSession(userId);
        if (!session) return null;

        const mode = await this.getFocusMode(session.focus_mode_id);
        if (!mode) return null;

        return mode.agent_configs.find(c => c.agent === agent) ?? null;
    }

    /**
     * Get available focus modes for user
     */
    async getAvailableModes(userId: string): Promise<FocusMode[]> {
        const keys = await this.redis.keys(`${this.modePrefix}*`);
        const modes: FocusMode[] = [];

        for (const key of keys) {
            const mode = await this.redis.get<FocusMode>(key);
            if (mode && (mode.owner_id === userId || mode.shared)) {
                modes.push(mode);
            }
        }

        return modes;
    }

    /**
     * Default focus modes
     */
    static readonly DEFAULT_MODES: Array<Omit<FocusMode, 'id' | 'owner_id' | 'created_at' | 'updated_at'>> = [
        {
            name: 'Deep Research',
            description: 'Thorough research with extensive sources',
            agent_configs: [
                { agent: 'researcher', enabled: true, priority_boost: 2, temperature_override: 0.3 },
                { agent: 'critic', enabled: true, priority_boost: 1 },
                { agent: 'skeptic', enabled: true, priority_boost: 1 },
            ],
            response_style: 'detailed',
            shared: true,
        },
        {
            name: 'Quick Coding',
            description: 'Fast code generation with minimal approval',
            agent_configs: [
                { agent: 'coder', enabled: true, priority_boost: 3, temperature_override: 0.5 },
                { agent: 'critic', enabled: true },
                { agent: 'researcher', enabled: false },
            ],
            response_style: 'concise',
            approval_threshold: 0.9,
            shared: true,
        },
        {
            name: 'Architecture Design',
            description: 'System design with documentation',
            agent_configs: [
                { agent: 'design', enabled: true, priority_boost: 3 },
                { agent: 'analyst', enabled: true, priority_boost: 2 },
                { agent: 'skeptic', enabled: true, priority_boost: 2 },
                { agent: 'coder', enabled: false },
            ],
            response_style: 'detailed',
            shared: true,
        },
    ];
}

// Singleton
let managerInstance: FocusModeManager | null = null;

export function getFocusModeManager(): FocusModeManager {
    if (!managerInstance) {
        managerInstance = new FocusModeManager();
    }
    return managerInstance;
}
