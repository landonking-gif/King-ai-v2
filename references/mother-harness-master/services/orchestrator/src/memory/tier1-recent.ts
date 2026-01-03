/**
 * Tier 1 Memory - Recent Context
 * Manages the last 10 messages per project (verbatim storage)
 */

import type { Message, Project } from '@mother-harness/shared';
import { getRedisJSON } from '@mother-harness/shared';
import { nanoid } from 'nanoid';

/** Maximum messages to keep in Tier 1 */
const MAX_RECENT_MESSAGES = 10;

export class Tier1Memory {
    private redis = getRedisJSON();

    /**
     * Add a user message to project context
     */
    async addUserMessage(projectId: string, content: string): Promise<Message> {
        const message: Message = {
            id: `msg-${nanoid()}`,
            role: 'user',
            content,
            timestamp: new Date().toISOString(),
        };

        await this.addMessage(projectId, message);
        return message;
    }

    /**
     * Add an assistant message to project context
     */
    async addAssistantMessage(
        projectId: string,
        content: string,
        agentInvoked?: Message['agent_invoked']
    ): Promise<Message> {
        const message: Message = {
            id: `msg-${nanoid()}`,
            role: 'assistant',
            content,
            ...(agentInvoked && { agent_invoked: agentInvoked }),
            timestamp: new Date().toISOString(),
        };

        await this.addMessage(projectId, message);
        return message;
    }

    /**
     * Get recent messages for a project
     */
    async getRecentMessages(projectId: string): Promise<Message[]> {
        const project = await this.redis.get<Project>(`project:${projectId}`);
        return project?.recent_messages ?? [];
    }

    /**
     * Get formatted context string for agent prompts
     */
    async getContextString(projectId: string): Promise<string> {
        const messages = await this.getRecentMessages(projectId);

        if (messages.length === 0) {
            return 'No previous context available.';
        }

        return messages
            .map(m => `[${m.role.toUpperCase()}${m.agent_invoked ? ` via ${m.agent_invoked}` : ''}]: ${m.content}`)
            .join('\n\n');
    }

    /**
     * Add a message and maintain max limit
     */
    private async addMessage(projectId: string, message: Message): Promise<void> {
        // Get current messages
        const project = await this.redis.get<Project>(`project:${projectId}`);
        if (!project) {
            throw new Error(`Project not found: ${projectId}`);
        }

        // Add message and trim to max
        const messages = [...project.recent_messages, message];
        const trimmed = messages.slice(-MAX_RECENT_MESSAGES);

        // Update in Redis
        await this.redis.set(`project:${projectId}`, '$.recent_messages', trimmed);
        await this.redis.set(`project:${projectId}`, '$.last_activity', new Date().toISOString());
    }

    /**
     * Clear all messages for a project (used when starting fresh)
     */
    async clearMessages(projectId: string): Promise<void> {
        await this.redis.set(`project:${projectId}`, '$.recent_messages', []);
    }

    /**
     * Get message count
     */
    async getMessageCount(projectId: string): Promise<number> {
        const messages = await this.getRecentMessages(projectId);
        return messages.length;
    }
}
