/**
 * Zod Validation Schemas
 * Runtime validation for all core types
 */

import { z } from 'zod';

// ============ Agent Schemas ============

export const AgentTypeSchema = z.enum([
    'orchestrator',
    'researcher',
    'coder',
    'design',
    'analyst',
    'critic',
    'skeptic',
    'rag',
    'librarian',
    'vision',
    'update',
    'toolsmith',
]);

export const AgentRequestSchema = z.object({
    task_id: z.string(),
    step_id: z.string(),
    agent: AgentTypeSchema,
    model: z.string().optional(),
    inputs: z.string(),
    context: z.record(z.unknown()).optional(),
});

export const AgentResponseSchema = z.object({
    agent: AgentTypeSchema,
    model_used: z.string(),
    result: z.unknown(),
    explanation: z.string().optional(),
    sources: z.array(z.string()).optional(),
    artifacts: z.array(z.string()).optional(),
    tokens_used: z.number(),
    duration_ms: z.number(),
    timestamp: z.string().datetime(),
});

// ============ Task Schemas ============

export const TaskTypeSchema = z.enum(['research', 'code', 'design', 'analysis', 'mixed']);

export const TaskStatusSchema = z.enum([
    'planning',
    'executing',
    'approval_needed',
    'completed',
    'failed',
]);

export const RiskLevelSchema = z.enum(['low', 'medium', 'high']);

export const ApprovalTypeSchema = z.enum([
    'file_write',
    'code_execution',
    'git_push',
    'workflow_creation',
    'api_call',
]);

export const TodoItemSchema = z.object({
    id: z.string(),
    description: z.string(),
    agent: AgentTypeSchema,
    status: z.enum(['pending', 'in_progress', 'completed', 'failed']),
    depends_on: z.array(z.string()),
    require_approval: z.boolean().optional(),
    approval_type: ApprovalTypeSchema.optional(),
    risk: RiskLevelSchema.optional(),
    result: z.unknown().optional(),
    error: z.string().optional(),
    started_at: z.string().datetime().optional(),
    completed_at: z.string().datetime().optional(),
});

export const ExecutionPlanSchema = z.object({
    steps: z.array(TodoItemSchema),
    estimated_duration: z.string().optional(),
    created_at: z.string().datetime(),
    updated_at: z.string().datetime(),
});

export const ArtifactSchema = z.object({
    id: z.string(),
    task_id: z.string(),
    type: z.enum(['report', 'code', 'diagram', 'data', 'other']),
    name: z.string(),
    content: z.string(),
    format: z.enum(['markdown', 'json', 'typescript', 'mermaid', 'text']),
    created_at: z.string().datetime(),
});

export const TaskResultSchema = z.object({
    summary: z.string(),
    findings: z.array(z.string()).optional(),
    recommendations: z.array(z.string()).optional(),
    artifacts: z.array(ArtifactSchema),
    agents_used: z.array(AgentTypeSchema),
    total_tokens: z.number(),
    total_duration_ms: z.number(),
});

export const TaskSchema = z.object({
    id: z.string(),
    project_id: z.string(),
    user_id: z.string(),
    type: TaskTypeSchema,
    query: z.string(),
    status: TaskStatusSchema,
    todo_list: z.array(TodoItemSchema),
    execution_plan: ExecutionPlanSchema,
    current_step: z.string().optional(),
    steps_completed: z.array(z.string()),
    result: TaskResultSchema.optional(),
    artifacts: z.array(ArtifactSchema),
    created_at: z.string().datetime(),
    updated_at: z.string().datetime(),
    completed_at: z.string().datetime().optional(),
});

// ============ Project Schemas ============

export const ProjectTypeSchema = z.enum(['research', 'code', 'design', 'mixed']);
export const ProjectStatusSchema = z.enum(['active', 'completed', 'archived']);
export const MessageRoleSchema = z.enum(['user', 'assistant', 'system']);

export const MessageSchema = z.object({
    id: z.string(),
    role: MessageRoleSchema,
    content: z.string(),
    agent_invoked: AgentTypeSchema.optional(),
    timestamp: z.string().datetime(),
});

export const SessionSummarySchema = z.object({
    session_id: z.string(),
    started: z.string().datetime(),
    ended: z.string().datetime(),
    summary: z.string(),
    key_decisions: z.array(z.string()),
    artifacts: z.array(z.string()),
    agents_used: z.array(AgentTypeSchema),
    task_ids: z.array(z.string()),
});

export const ProjectSchema = z.object({
    id: z.string(),
    name: z.string(),
    type: ProjectTypeSchema,
    status: ProjectStatusSchema,
    threads: z.array(z.string()),
    recent_messages: z.array(MessageSchema),
    session_summaries: z.array(SessionSummarySchema),
    created_at: z.string().datetime(),
    updated_at: z.string().datetime(),
    last_activity: z.string().datetime(),
});

// ============ Approval Schemas ============

export const ApprovalStatusSchema = z.enum(['pending', 'approved', 'rejected']);

export const ApprovalPreviewSchema = z.object({
    files: z.array(z.string()).optional(),
    commands: z.array(z.string()).optional(),
    workflow: z.record(z.unknown()).optional(),
    api_calls: z.array(z.object({
        method: z.string(),
        url: z.string(),
        description: z.string(),
    })).optional(),
});

export const ApprovalSchema = z.object({
    id: z.string(),
    run_id: z.string(),
    task_id: z.string(),
    project_id: z.string(),
    step_id: z.string(),
    user_id: z.string(),
    type: ApprovalTypeSchema,
    description: z.string(),
    risk_level: RiskLevelSchema,
    preview: ApprovalPreviewSchema,
    status: ApprovalStatusSchema,
    response_notes: z.string().optional(),
    created_at: z.string().datetime(),
    responded_at: z.string().datetime().optional(),
    expires_at: z.string().datetime().optional(),
});

// ============ Document Schemas ============

export const ChunkTypeSchema = z.enum(['text', 'table', 'figure', 'code']);
export const ScanStatusSchema = z.enum(['idle', 'scanning', 'processing']);

export const ImageReferenceSchema = z.object({
    id: z.string(),
    file_path: z.string(),
    caption: z.string().optional(),
    page_number: z.number().optional(),
});

export const DocumentChunkSchema = z.object({
    id: z.string(),
    library: z.string(),
    document_id: z.string(),
    document_name: z.string(),
    file_path: z.string(),
    content: z.string(),
    embedding: z.array(z.number()),
    images: z.array(ImageReferenceSchema),
    tables: z.array(z.object({
        id: z.string(),
        content: z.string(),
        caption: z.string().optional(),
        page_number: z.number().optional(),
    })),
    page_number: z.number().optional(),
    section_title: z.string().optional(),
    hierarchy: z.array(z.string()),
    chunk_type: ChunkTypeSchema,
    chunk_index: z.number(),
    total_chunks: z.number(),
    content_hash: z.string(),
    indexed_at: z.string().datetime(),
    source_modified_at: z.string().datetime(),
    searchable: z.boolean(),
});

export const LibrarySchema = z.object({
    id: z.string(),
    name: z.string(),
    folder_path: z.string(),
    description: z.string().optional(),
    document_count: z.number(),
    chunk_count: z.number(),
    total_size_bytes: z.number(),
    last_scanned: z.string().datetime(),
    scan_status: ScanStatusSchema,
    processed_count: z.number().optional(),
    total_files: z.number().optional(),
    auto_scan: z.boolean(),
    scan_schedule: z.string().optional(),
    created_at: z.string().datetime(),
    updated_at: z.string().datetime(),
});

// ============ Result Schemas ============

export const TerminationReasonSchema = z.enum([
    'completed',
    'user_cancelled',
    'approval_rejected',
    'approval_timeout',
    'budget_exhausted',
    'rate_limited',
    'timeout',
    'max_retries',
    'agent_error',
    'validation_error',
    'dependency_failed',
    'conflict_unresolved',
    'circuit_breaker',
]);

export const TerminationRecordSchema = z.object({
    run_id: z.string(),
    task_id: z.string(),
    project_id: z.string(),
    user_id: z.string(),
    reason: TerminationReasonSchema,
    details: z.string(),
    last_step_id: z.string().optional(),
    last_agent: AgentTypeSchema.optional(),
    error_stack: z.string().optional(),
    total_steps_planned: z.number(),
    steps_completed: z.number(),
    total_tokens: z.number(),
    total_duration_ms: z.number(),
    started_at: z.string().datetime(),
    terminated_at: z.string().datetime(),
});

// ============ Run Schemas ============

export const RunStatusSchema = z.enum([
    'created',
    'executing',
    'waiting_approval',
    'terminated',
]);

export const RunSchema = z.object({
    id: z.string(),
    task_id: z.string(),
    project_id: z.string(),
    user_id: z.string(),
    status: RunStatusSchema,
    created_at: z.string().datetime(),
    updated_at: z.string().datetime(),
    started_at: z.string().datetime().optional(),
    terminated_at: z.string().datetime().optional(),
    termination_reason: TerminationReasonSchema.optional(),
    termination_details: z.string().optional(),
    total_tokens: z.number().optional(),
    total_duration_ms: z.number().optional(),
});

export const ResultEnvelopeSchema = z.object({
    success: z.boolean(),
    data: z.unknown().optional(),
    error: z.object({
        code: z.string(),
        message: z.string(),
        details: z.unknown().optional(),
    }).optional(),
    task_id: z.string(),
    run_id: z.string(),
    termination: z.object({
        reason: TerminationReasonSchema,
        details: z.string(),
    }),
    metrics: z.object({
        total_tokens: z.number(),
        total_duration_ms: z.number(),
        agents_used: z.array(AgentTypeSchema),
        steps_completed: z.number(),
    }),
    context_summary: z.object({
        original_tokens: z.number(),
        summarized_tokens: z.number(),
        summarization_method: z.enum(['extractive', 'llm', 'none']),
    }).optional(),
    timestamp: z.string().datetime(),
});

// ============ Resource Budget Schemas ============

export const BudgetScopeSchema = z.enum(['run', 'user', 'global']);

export const ResourceTypeSchema = z.enum([
    'tokens',
    'api_calls',
    'tool_executions',
    'agent_invocations',
    'embeddings',
    'storage_bytes',
    'cost',
]);

export const ResourceBudgetCountersSchema = z.object({
    tokens: z.number(),
    api_calls: z.number(),
    tool_executions: z.number(),
    agent_invocations: z.number(),
    embeddings: z.number(),
    storage_bytes: z.number(),
    cost: z.number(),
});

export const ResourceBudgetSchema = z.object({
    scope: BudgetScopeSchema,
    scope_id: z.string(),
    limits: ResourceBudgetCountersSchema,
    used: ResourceBudgetCountersSchema,
    warnings_sent: z.array(ResourceTypeSchema),
    created_at: z.string().datetime(),
    updated_at: z.string().datetime(),
});

export const ResourceUsageReportItemSchema = z.object({
    used: z.number(),
    limit: z.number(),
    percent: z.number(),
});

export const ResourceUsageReportSchema = z.record(ResourceTypeSchema, ResourceUsageReportItemSchema);
