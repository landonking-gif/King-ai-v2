/**
 * Project Types and Schemas
 * Project management with 3-tier memory system
 */

import type { AgentType } from './agent.js';

/** Project type classification */
export type ProjectType = 'research' | 'code' | 'design' | 'mixed';

/** Project lifecycle status */
export type ProjectStatus = 'active' | 'completed' | 'archived';

/** Message role in conversation */
export type MessageRole = 'user' | 'assistant' | 'system';

/** Individual message in project conversation (Tier 1 memory) */
export interface Message {
    id: string;
    role: MessageRole;
    content: string;
    agent_invoked?: AgentType;     // Which agent responded
    timestamp: string;
}

/** Session summary after task completion (Tier 2 memory) */
export interface SessionSummary {
    session_id: string;
    started: string;
    ended: string;
    summary: string;               // Natural language summary
    key_decisions: string[];       // Important decisions made
    artifacts: string[];           // Artifact IDs produced
    agents_used: AgentType[];      // Agents that participated
    task_ids: string[];            // Tasks completed in this session
}

/** Long-term memory entry (Tier 3 memory) */
export interface LongTermMemory {
    chunk_id: string;
    project_id: string;
    content: string;
    embedding: number[];           // 768-dim vector
    metadata: {
        type: 'decision' | 'finding' | 'code' | 'artifact';
        session_id: string;
        timestamp: string;
        tags: string[];
    };
}

/** Main Project interface - stored in Redis as project:{id} */
export interface Project {
    id: string;
    name: string;
    type: ProjectType;
    status: ProjectStatus;

    // Thread references
    threads: string[];             // Array of task_ids

    // Tier 1: Recent Context (verbatim, last 10 messages)
    recent_messages: Message[];

    // Tier 2: Session Summaries
    session_summaries: SessionSummary[];

    // Metadata
    created_at: string;
    updated_at: string;
    last_activity: string;
}

/** Non-project chat thread */
export interface ChatThread {
    id: string;
    title: string;
    messages: Message[];
    summary?: string;              // Generated when closed
    created_at: string;
    updated_at: string;
}

/** Create a new project with defaults */
export function createProject(
    id: string,
    name: string,
    type: ProjectType = 'mixed'
): Project {
    const now = new Date().toISOString();
    return {
        id,
        name,
        type,
        status: 'active',
        threads: [],
        recent_messages: [],
        session_summaries: [],
        created_at: now,
        updated_at: now,
        last_activity: now,
    };
}

/** Add a message to project's recent messages (maintains max 10) */
export function addMessage(project: Project, message: Message): Project {
    const recent = [...project.recent_messages, message];
    return {
        ...project,
        recent_messages: recent.slice(-10), // Keep last 10
        updated_at: new Date().toISOString(),
        last_activity: new Date().toISOString(),
    };
}
