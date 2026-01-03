/**
 * Agent Types and Schemas
 * Defines the specialist agents available in Mother-Harness
 */

/** Available agent types in the system */
export type AgentType =
    | 'orchestrator'  // Mother - planning, routing, synthesis
    | 'researcher'    // Web research, documentation
    | 'coder'         // Code generation, git ops
    | 'design'        // Architecture, UI/UX design
    | 'analyst'       // Data analysis, visualization
    | 'critic'        // Verification, security review
    | 'skeptic'       // Business validation, devil's advocate
    | 'rag'           // Document retrieval, synthesis
    | 'librarian'     // Document ingestion, chunking, embedding
    | 'vision'        // Multimodal analysis, OCR, diagrams
    | 'update'        // Software inventory, upgrade intelligence
    | 'toolsmith';    // Deterministic tool wrapper creation

/** Model assignment for an agent */
export interface ModelAssignment {
    local: string;      // e.g., 'gpt-oss:20b'
    cloud: string;      // e.g., 'devstral-2:123b-cloud'
}

/** Agent roster with default model assignments */
export const DEFAULT_AGENT_MODELS: Record<AgentType, ModelAssignment> = {
    orchestrator: { local: 'gpt-oss:20b', cloud: 'gpt-oss:120b-cloud' },
    researcher: { local: 'gpt-oss:20b', cloud: 'qwen3-next:80b-cloud' },
    coder: { local: 'gpt-oss:20b', cloud: 'devstral-2:123b-cloud' },
    design: { local: 'gpt-oss:20b', cloud: 'gemini-3-flash-preview-cloud' },
    analyst: { local: 'gpt-oss:20b', cloud: 'qwen3-coder:30b-cloud' },
    critic: { local: 'gpt-oss:20b', cloud: 'deepseek-v3.1:671b-cloud' },
    skeptic: { local: 'gpt-oss:20b', cloud: 'deepseek-v3.2-cloud' },
    rag: { local: 'gpt-oss:20b', cloud: 'gpt-oss:20b' }, // Local only
    librarian: { local: 'gpt-oss:20b', cloud: 'gpt-oss:20b' },
    vision: { local: 'gpt-oss:20b', cloud: 'gemini-3-flash-preview-cloud' },
    update: { local: 'gpt-oss:20b', cloud: 'gpt-oss:120b-cloud' },
    toolsmith: { local: 'gpt-oss:20b', cloud: 'gpt-oss:20b' },
};

/** Response from an agent execution */
export interface AgentResponse {
    agent: AgentType;
    model_used: string;
    result: unknown;
    explanation?: string;
    sources?: string[];
    artifacts?: string[];
    tokens_used: number;
    duration_ms: number;
    timestamp: string;
}

/** Agent execution request */
export interface AgentRequest {
    task_id: string;
    step_id: string;
    agent: AgentType;
    model?: string; // Override default model
    inputs: string;
    context?: Record<string, unknown>;
}
