/**
 * Agent Contract Types
 * Defines action allowlists and artifact requirements per agent
 */

import type { AgentType } from './agent.js';

/** Contract describing what an agent is allowed to do */
export interface AgentContract {
    agent: AgentType;
    action_allowlist: string[];
    default_action: string;
    required_artifacts: string[];
}

/** Default contracts aligned to the agent roster */
export const DEFAULT_AGENT_CONTRACTS: Record<AgentType, AgentContract> = {
    orchestrator: {
        agent: 'orchestrator',
        action_allowlist: ['model_selection', 'rag_retrieval'],
        default_action: 'model_selection',
        required_artifacts: [],
    },
    researcher: {
        agent: 'researcher',
        action_allowlist: ['web_search', 'rag_retrieval'],
        default_action: 'rag_retrieval',
        required_artifacts: [],
    },
    coder: {
        agent: 'coder',
        action_allowlist: ['code_generation', 'file_read', 'file_write', 'git_operations', 'code_execution'],
        default_action: 'code_generation',
        required_artifacts: [],
    },
    design: {
        agent: 'design',
        action_allowlist: ['diagram_generation', 'rag_retrieval'],
        default_action: 'diagram_generation',
        required_artifacts: [],
    },
    analyst: {
        agent: 'analyst',
        action_allowlist: ['database_read', 'code_execution', 'rag_retrieval'],
        default_action: 'rag_retrieval',
        required_artifacts: [],
    },
    critic: {
        agent: 'critic',
        action_allowlist: ['rag_retrieval'],
        default_action: 'rag_retrieval',
        required_artifacts: [],
    },
    skeptic: {
        agent: 'skeptic',
        action_allowlist: ['rag_retrieval'],
        default_action: 'rag_retrieval',
        required_artifacts: [],
    },
    rag: {
        agent: 'rag',
        action_allowlist: ['rag_retrieval', 'embedding_generation'],
        default_action: 'rag_retrieval',
        required_artifacts: [],
    },
    librarian: {
        agent: 'librarian',
        action_allowlist: ['file_read', 'embedding_generation', 'database_write'],
        default_action: 'file_read',
        required_artifacts: [],
    },
    vision: {
        agent: 'vision',
        action_allowlist: ['image_analysis', 'file_read'],
        default_action: 'image_analysis',
        required_artifacts: [],
    },
    update: {
        agent: 'update',
        action_allowlist: ['web_search', 'rag_retrieval', 'api_calls'],
        default_action: 'rag_retrieval',
        required_artifacts: [],
    },
    toolsmith: {
        agent: 'toolsmith',
        action_allowlist: ['code_generation', 'tool_creation'],
        default_action: 'tool_creation',
        required_artifacts: [],
    },
};
