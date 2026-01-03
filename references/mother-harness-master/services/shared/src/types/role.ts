/**
 * Role Definitions
 * Complete agent role definitions with capabilities and constraints
 */

import type { AgentType } from './agent.js';
import type { RiskLevel, ApprovalType } from './task.js';

/** Capability that an agent may have */
export type AgentCapability =
    | 'web_search'           // Can search the web
    | 'code_generation'      // Can generate code
    | 'code_execution'       // Can execute code (sandboxed)
    | 'file_read'            // Can read files
    | 'file_write'           // Can write files
    | 'git_operations'       // Can perform git operations
    | 'database_read'        // Can read from database
    | 'database_write'       // Can write to database
    | 'api_calls'            // Can make external API calls
    | 'rag_retrieval'        // Can query document stores
    | 'embedding_generation' // Can generate embeddings
    | 'image_analysis'       // Can analyze images
    | 'diagram_generation'   // Can generate diagrams
    | 'workflow_creation'    // Can create n8n workflows
    | 'tool_creation'        // Can create deterministic tools
    | 'model_selection';     // Can select models for other agents

/** Action that requires approval */
export interface ApprovalRequirement {
    action: string;
    approval_type: ApprovalType;
    risk_level: RiskLevel;
    description: string;
}

/** Complete role definition */
export interface RoleDefinition {
    type: AgentType;
    name: string;
    description: string;

    // Capabilities
    capabilities: AgentCapability[];

    // Constraints
    max_tokens_per_call: number;
    max_retries: number;
    timeout_ms: number;

    // Actions requiring approval
    approval_requirements: ApprovalRequirement[];

    // Phase exit criteria
    required_outputs: string[];
    optional_outputs: string[];

    // Model preferences
    preferred_local_model: string;
    preferred_cloud_model: string;
    requires_vision: boolean;
    requires_tool_calling: boolean;

    // Metadata
    version: string;
    enabled: boolean;
}

/** Default role definitions for all agents */
export const DEFAULT_ROLES: Record<AgentType, RoleDefinition> = {
    orchestrator: {
        type: 'orchestrator',
        name: 'Mother',
        description: 'Plans tasks, routes to specialists, synthesizes results',
        capabilities: ['model_selection', 'rag_retrieval'],
        max_tokens_per_call: 8000,
        max_retries: 3,
        timeout_ms: 60000,
        approval_requirements: [],
        required_outputs: ['execution_plan'],
        optional_outputs: ['context_summary'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'gpt-oss:120b-cloud',
        requires_vision: false,
        requires_tool_calling: true,
        version: '1.0.0',
        enabled: true,
    },

    researcher: {
        type: 'researcher',
        name: 'Researcher',
        description: 'Conducts web research, synthesizes findings with citations',
        capabilities: ['rag_retrieval'],
        max_tokens_per_call: 8000,
        max_retries: 3,
        timeout_ms: 120000,
        approval_requirements: [],
        required_outputs: ['research_summary', 'citations'],
        optional_outputs: ['key_findings', 'recommendations'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'qwen3-next:80b-cloud',
        requires_vision: false,
        requires_tool_calling: false,
        version: '1.0.0',
        enabled: true,
    },

    coder: {
        type: 'coder',
        name: 'Coder',
        description: 'Generates code, creates patches, runs tests',
        capabilities: ['code_generation', 'file_read', 'file_write', 'git_operations', 'code_execution'],
        max_tokens_per_call: 16000,
        max_retries: 2,
        timeout_ms: 300000,
        approval_requirements: [
            {
                action: 'file_write',
                approval_type: 'file_write',
                risk_level: 'medium',
                description: 'Writing files to the filesystem',
            },
            {
                action: 'git_push',
                approval_type: 'git_push',
                risk_level: 'high',
                description: 'Pushing changes to remote repository',
            },
            {
                action: 'code_execution',
                approval_type: 'code_execution',
                risk_level: 'high',
                description: 'Executing generated code',
            },
        ],
        required_outputs: ['code_changes', 'explanation'],
        optional_outputs: ['tests', 'documentation'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'devstral-2:123b-cloud',
        requires_vision: false,
        requires_tool_calling: true,
        version: '1.0.0',
        enabled: true,
    },

    design: {
        type: 'design',
        name: 'Designer',
        description: 'Creates architecture diagrams, UI/UX designs, system designs',
        capabilities: ['diagram_generation', 'rag_retrieval'],
        max_tokens_per_call: 8000,
        max_retries: 3,
        timeout_ms: 120000,
        approval_requirements: [],
        required_outputs: ['design_document'],
        optional_outputs: ['diagrams', 'alternatives'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'gemini-3-flash-preview-cloud',
        requires_vision: true,
        requires_tool_calling: false,
        version: '1.0.0',
        enabled: true,
    },

    analyst: {
        type: 'analyst',
        name: 'Analyst',
        description: 'Analyzes data, generates reports and visualizations',
        capabilities: ['rag_retrieval'],
        max_tokens_per_call: 8000,
        max_retries: 3,
        timeout_ms: 180000,
        approval_requirements: [
            {
                action: 'code_execution',
                approval_type: 'code_execution',
                risk_level: 'medium',
                description: 'Executing analysis scripts',
            },
        ],
        required_outputs: ['analysis_report'],
        optional_outputs: ['visualizations', 'data_exports'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'qwen3-coder:30b-cloud',
        requires_vision: false,
        requires_tool_calling: false,
        version: '1.0.0',
        enabled: true,
    },

    critic: {
        type: 'critic',
        name: 'Critic',
        description: 'Reviews work, validates quality, checks security',
        capabilities: ['rag_retrieval'],
        max_tokens_per_call: 8000,
        max_retries: 2,
        timeout_ms: 120000,
        approval_requirements: [],
        required_outputs: ['review_report', 'issues_found'],
        optional_outputs: ['suggestions', 'security_notes'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'deepseek-v3.1:671b-cloud',
        requires_vision: false,
        requires_tool_calling: false,
        version: '1.0.0',
        enabled: true,
    },

    skeptic: {
        type: 'skeptic',
        name: 'Skeptic',
        description: 'Challenges assumptions, proposes alternatives, devil\'s advocate',
        capabilities: ['rag_retrieval'],
        max_tokens_per_call: 8000,
        max_retries: 2,
        timeout_ms: 60000,
        approval_requirements: [],
        required_outputs: ['challenges', 'alternatives'],
        optional_outputs: ['risk_assessment'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'deepseek-v3.2-cloud',
        requires_vision: false,
        requires_tool_calling: false,
        version: '1.0.0',
        enabled: true,
    },

    rag: {
        type: 'rag',
        name: 'RAG Agent',
        description: 'Retrieves and synthesizes information from document libraries',
        capabilities: ['rag_retrieval', 'embedding_generation'],
        max_tokens_per_call: 8000,
        max_retries: 3,
        timeout_ms: 30000,
        approval_requirements: [],
        required_outputs: ['answer', 'sources'],
        optional_outputs: ['confidence_score', 'related_chunks'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'gpt-oss:20b', // Local only
        requires_vision: false,
        requires_tool_calling: false,
        version: '1.0.0',
        enabled: true,
    },

    librarian: {
        type: 'librarian',
        name: 'Librarian',
        description: 'Ingests documents, creates embeddings, manages libraries',
        capabilities: ['file_read', 'embedding_generation', 'database_write'],
        max_tokens_per_call: 4000,
        max_retries: 3,
        timeout_ms: 600000, // 10 minutes for large documents
        approval_requirements: [],
        required_outputs: ['ingestion_report'],
        optional_outputs: ['chunk_count', 'embedding_stats'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'gpt-oss:20b', // Local only
        requires_vision: false,
        requires_tool_calling: false,
        version: '1.0.0',
        enabled: true,
    },

    vision: {
        type: 'vision',
        name: 'Vision',
        description: 'Analyzes images, extracts text via OCR, parses diagrams',
        capabilities: ['image_analysis', 'file_read'],
        max_tokens_per_call: 8000,
        max_retries: 2,
        timeout_ms: 120000,
        approval_requirements: [],
        required_outputs: ['analysis_result'],
        optional_outputs: ['extracted_text', 'diagram_elements'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'gemini-3-flash-preview-cloud',
        requires_vision: true,
        requires_tool_calling: false,
        version: '1.0.0',
        enabled: true,
    },

    update: {
        type: 'update',
        name: 'Update Agent',
        description: 'Tracks software inventory, recommends updates, assesses impact',
        capabilities: ['rag_retrieval'],
        max_tokens_per_call: 8000,
        max_retries: 3,
        timeout_ms: 180000,
        approval_requirements: [],
        required_outputs: ['update_recommendations'],
        optional_outputs: ['breaking_changes', 'migration_steps'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'gpt-oss:120b-cloud',
        requires_vision: false,
        requires_tool_calling: false,
        version: '1.0.0',
        enabled: true,
    },

    toolsmith: {
        type: 'toolsmith',
        name: 'Toolsmith',
        description: 'Creates deterministic tool wrappers for common operations',
        capabilities: ['code_generation', 'tool_creation'],
        max_tokens_per_call: 8000,
        max_retries: 2,
        timeout_ms: 120000,
        approval_requirements: [
            {
                action: 'tool_creation',
                approval_type: 'workflow_creation',
                risk_level: 'medium',
                description: 'Creating new tool wrapper',
            },
        ],
        required_outputs: ['tool_definition', 'tool_code'],
        optional_outputs: ['tests', 'documentation'],
        preferred_local_model: 'gpt-oss:20b',
        preferred_cloud_model: 'gpt-oss:20b', // Local only
        requires_vision: false,
        requires_tool_calling: true,
        version: '1.0.0',
        enabled: true,
    },
};
