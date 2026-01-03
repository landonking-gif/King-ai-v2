/**
 * Task Planner
 * Generates execution plans from user queries
 */

import type { TodoItem, ExecutionPlan, AgentType } from '@mother-harness/shared';

/** Context for planning */
export interface PlanningContext {
    project_id: string;
    user_id: string;
    recent_context?: string;
}

/**
 * Keywords that suggest specific agent types.
 * Note: skeptic = critique/stress-test; critic = QA/review gate.
 */
const AGENT_KEYWORDS: Record<AgentType, string[]> = {
    orchestrator: [],
    researcher: ['research', 'find', 'search', 'look up', 'investigate', 'explore', 'learn about'],
    coder: ['code', 'implement', 'build', 'create', 'develop', 'program', 'write code', 'fix bug'],
    design: ['design', 'architecture', 'diagram', 'ui', 'ux', 'layout', 'wireframe', 'mockup'],
    analyst: ['analyze', 'data', 'statistics', 'metrics', 'report', 'visualize', 'chart', 'graph'],
    critic: ['review', 'verify', 'check', 'validate', 'test', 'security', 'quality'],
    skeptic: ['challenge', 'question', 'alternative', 'risk', 'what if', 'devil\'s advocate'],
    rag: ['document', 'retrieve', 'find in', 'search documents'],
    librarian: ['ingest', 'index', 'store', 'library'],
    vision: ['image', 'screenshot', 'visual', 'ocr', 'diagram'],
    update: ['update', 'upgrade', 'version', 'dependency'],
    toolsmith: ['tool', 'wrapper', 'integration'],
};

export class TaskPlanner {
    /**
     * Create an execution plan from a query
     */
    async createPlan(query: string, _context: PlanningContext): Promise<ExecutionPlan> {
        // Detect which agents are needed based on query
        const agentsNeeded = this.detectAgents(query);

        // Generate steps based on detected agents
        const steps = this.generateSteps(query, agentsNeeded);

        const now = new Date().toISOString();

        return {
            steps,
            estimated_duration: this.estimateDuration(steps),
            created_at: now,
            updated_at: now,
        };
    }

    /**
     * Detect which agents are needed based on query keywords
     */
    private detectAgents(query: string): AgentType[] {
        const lowerQuery = query.toLowerCase();
        const detected: Set<AgentType> = new Set();

        for (const [agent, keywords] of Object.entries(AGENT_KEYWORDS)) {
            for (const keyword of keywords) {
                if (lowerQuery.includes(keyword)) {
                    detected.add(agent as AgentType);
                    break;
                }
            }
        }

        // Default to researcher if nothing detected
        if (detected.size === 0) {
            detected.add('researcher');
        }

        // Always add critic for review
        detected.add('critic');

        return Array.from(detected);
    }

    /**
     * Generate execution steps based on agents needed
     */
    private generateSteps(query: string, agents: AgentType[]): TodoItem[] {
        const steps: TodoItem[] = [];
        let stepNum = 1;

        // Research step if researcher detected
        if (agents.includes('researcher')) {
            steps.push({
                id: `step-${stepNum++}`,
                description: `Research: ${this.extractTopic(query)}`,
                agent: 'researcher',
                status: 'pending',
                depends_on: [],
            });
        }

        // Design step if design detected
        if (agents.includes('design')) {
            steps.push({
                id: `step-${stepNum++}`,
                description: `Design architecture for: ${this.extractTopic(query)}`,
                agent: 'design',
                status: 'pending',
                depends_on: steps.length > 0 ? [steps[steps.length - 1]!.id] : [],
            });
        }

        // Coder step if coder detected
        if (agents.includes('coder')) {
            steps.push({
                id: `step-${stepNum++}`,
                description: `Implement: ${this.extractTopic(query)}`,
                agent: 'coder',
                status: 'pending',
                depends_on: steps.length > 0 ? [steps[steps.length - 1]!.id] : [],
                require_approval: true,
                approval_type: 'code_execution',
                risk: 'medium',
            });
        }

        // Analysis step if analyst detected
        if (agents.includes('analyst')) {
            steps.push({
                id: `step-${stepNum++}`,
                description: `Analyze: ${this.extractTopic(query)}`,
                agent: 'analyst',
                status: 'pending',
                depends_on: steps.length > 0 ? [steps[steps.length - 1]!.id] : [],
            });
        }

        // Skeptic challenge if skeptic detected
        if (agents.includes('skeptic')) {
            steps.push({
                id: `step-${stepNum++}`,
                description: `Challenge assumptions for: ${this.extractTopic(query)}`,
                agent: 'skeptic',
                status: 'pending',
                depends_on: steps.length > 0 ? [steps[steps.length - 1]!.id] : [],
            });
        }

        // Always end with critic review
        if (agents.includes('critic')) {
            steps.push({
                id: `step-${stepNum++}`,
                description: 'Review and validate findings',
                agent: 'critic',
                status: 'pending',
                depends_on: steps.length > 0 ? [steps[steps.length - 1]!.id] : [],
            });
        }

        return steps;
    }

    /**
     * Extract main topic from query
     */
    private extractTopic(query: string): string {
        // Simple extraction - just trim and limit length
        const topic = query.trim();
        if (topic.length > 50) {
            return topic.substring(0, 47) + '...';
        }
        return topic;
    }

    /**
     * Estimate duration based on steps
     */
    private estimateDuration(steps: TodoItem[]): string {
        const minutes = steps.length * 5; // ~5 min per step estimate
        if (minutes < 60) {
            return `${minutes} minutes`;
        }
        const hours = Math.ceil(minutes / 60);
        return `${hours} hour${hours > 1 ? 's' : ''}`;
    }
}
