/**
 * Agents Service Index
 * Exports all agent implementations
 */

// Base
export { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';

// Agents
export { ResearcherAgent } from './researcher.js';
export { CoderAgent, type CodeChange } from './coder.js';
export { CriticAgent, type ReviewIssue, type IssueSeverity } from './critic.js';
export { DesignerAgent, type DiagramType } from './designer.js';
export { AnalystAgent, type AnalysisResult } from './analyst.js';
export { RAGAgent, type RAGResult } from './rag.js';
export { SkepticAgent, type Challenge, type Alternative } from './skeptic.js';
export { VisionAgent, type VisualElement } from './vision.js';
export { LibrarianAgent, type IngestionReport } from './librarian.js';
export { UpdateAgent, type SoftwareItem, type UpdateRecommendation } from './update.js';
export { ToolsmithAgent, type ToolDefinition } from './toolsmith.js';

// Agent factory
export { createAgent, type AgentFactory } from './factory.js';
