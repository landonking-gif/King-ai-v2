/**
 * Agent Factory
 * Creates agent instances based on type
 */

import type { AgentType } from '@mother-harness/shared';
import { BaseAgent } from './base-agent.js';
import { ResearcherAgent } from './researcher.js';
import { CoderAgent } from './coder.js';
import { CriticAgent } from './critic.js';
import { DesignerAgent } from './designer.js';
import { AnalystAgent } from './analyst.js';
import { RAGAgent } from './rag.js';
import { SkepticAgent } from './skeptic.js';
import { VisionAgent } from './vision.js';
import { LibrarianAgent } from './librarian.js';
import { UpdateAgent } from './update.js';
import { ToolsmithAgent } from './toolsmith.js';

export interface AgentFactory {
    create(type: AgentType): BaseAgent;
    getAvailableTypes(): AgentType[];
}

/** Agent type to class mapping */
const AGENT_CLASSES: Record<AgentType, new () => BaseAgent> = {
    orchestrator: ResearcherAgent, // Orchestrator uses researcher as fallback
    researcher: ResearcherAgent,
    coder: CoderAgent,
    design: DesignerAgent,
    analyst: AnalystAgent,
    critic: CriticAgent,
    skeptic: SkepticAgent,
    rag: RAGAgent,
    librarian: LibrarianAgent,
    vision: VisionAgent,
    update: UpdateAgent,
    toolsmith: ToolsmithAgent,
};

/** Agent instance cache */
const agentCache = new Map<AgentType, BaseAgent>();

/**
 * Create an agent instance
 */
export function createAgent(type: AgentType): BaseAgent {
    // Check cache first
    if (agentCache.has(type)) {
        return agentCache.get(type)!;
    }

    const AgentClass = AGENT_CLASSES[type];
    if (!AgentClass) {
        throw new Error(`Unknown agent type: ${type}`);
    }

    const agent = new AgentClass();
    agentCache.set(type, agent);

    return agent;
}

/**
 * Get all available agent types
 */
export function getAvailableAgentTypes(): AgentType[] {
    return Object.keys(AGENT_CLASSES) as AgentType[];
}

/**
 * Clear agent cache (useful for testing)
 */
export function clearAgentCache(): void {
    agentCache.clear();
}
