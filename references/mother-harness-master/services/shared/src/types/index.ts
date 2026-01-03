/**
 * Shared Types Index
 * Re-exports all types from the shared library
 */

// Agent types
export type { AgentType, ModelAssignment, AgentResponse, AgentRequest } from './agent.js';
export { DEFAULT_AGENT_MODELS } from './agent.js';

// Role types
export type {
    AgentCapability,
    ApprovalRequirement,
    RoleDefinition,
} from './role.js';
export { DEFAULT_ROLES } from './role.js';

// Contract types
export type { AgentContract } from './contract.js';
export { DEFAULT_AGENT_CONTRACTS } from './contract.js';

// Registry types
export type { RoleRegistryEntry } from './registry.js';


// Task types
export type {
    TaskType,
    TaskStatus,
    TodoItem,
    ApprovalType,
    RiskLevel,
    ExecutionPlan,
    Artifact,
    TaskResult,
    Task,
} from './task.js';
export { createTask } from './task.js';

// Project types
export type {
    ProjectType,
    ProjectStatus,
    MessageRole,
    Message,
    SessionSummary,
    LongTermMemory,
    Project,
    ChatThread,
} from './project.js';
export { createProject, addMessage } from './project.js';

// Approval types
export type { ApprovalStatus, Approval, ApprovalPreview, ApiCallPreview } from './approval.js';
export { createApproval, isApprovalPending } from './approval.js';

// Document types
export type {
    ChunkType,
    ScanStatus,
    ImageReference,
    TableReference,
    DocumentChunk,
    Library,
    DoclingJob,
} from './document.js';
export { createLibrary } from './document.js';

// Retrieval report types
export type { RetrievalChunkSummary, RetrievalReport } from './retrieval.js';

// Result types
export type {
    TerminationReason,
    TerminationRecord,
    ConflictType,
    ResolutionStrategy,
    ConflictResolution,
    ResultEnvelope,
} from './result.js';
export { createSuccessResult, createErrorResult } from './result.js';

// Run types
export type { RunStatus, Run } from './run.js';

// Resource budget types
export type {
    BudgetScope,
    ResourceType,
    ResourceBudgetCounters,
    ResourceBudget,
    ResourceUsageReportItem,
    ResourceUsageReport,
} from './resource-budget.js';
