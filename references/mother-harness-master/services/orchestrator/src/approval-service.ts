/**
 * Approval Service
 * Enhanced approval gating with risk assessment and auto-approval rules
 */

import type {
    TodoItem,
    Task,
    Approval,
    ApprovalType,
    RiskLevel,
    ApprovalPreview,
    AgentType
} from '@mother-harness/shared';
import { nanoid } from 'nanoid';

/** Auto-approval policy configuration */
export interface AutoApprovalPolicy {
    enabled: boolean;
    max_risk_level: RiskLevel;
    allowed_types: ApprovalType[];
    require_security_scan?: boolean;
}

/** Risk assessment result */
export interface RiskAssessment {
    level: RiskLevel;
    factors: string[];
    requires_manual_approval: boolean;
    auto_approvable: boolean;
}

/** Risky patterns that should trigger approval */
const RISKY_PATTERNS = {
    file_operations: [
        /\.env/i,
        /\.git/,
        /\.ssh/,
        /password/i,
        /secret/i,
        /token/i,
        /credentials/i,
        /private.*key/i,
    ],
    commands: [
        /rm\s+-rf/,
        /sudo/,
        /chmod\s+777/,
        /curl.*\|.*sh/,
        /wget.*\|.*sh/,
        /docker\s+run/,
        /eval\s*\(/,
        /exec\s*\(/,
    ],
    api_endpoints: [
        /delete/i,
        /destroy/i,
        /drop/i,
        /truncate/i,
        /payment/i,
        /billing/i,
    ],
    code_patterns: [
        /process\.exit/,
        /child_process/,
        /fs\.unlink/,
        /fs\.rmdir/,
        /require\s*\(\s*['"]child_process['"]\s*\)/,
    ],
};

/** Agent types that require approval for certain operations */
const HIGH_RISK_AGENTS: Set<AgentType> = new Set(['coder', 'toolsmith']);

export class ApprovalService {
    private defaultPolicy: AutoApprovalPolicy = {
        enabled: true,
        max_risk_level: 'low',
        allowed_types: [],
        require_security_scan: false,
    };

    /**
     * Assess risk level for a step
     */
    assessRisk(step: TodoItem, task: Task, stepResult?: unknown): RiskAssessment {
        const factors: string[] = [];
        let riskScore = 0;

        // Check agent type
        if (HIGH_RISK_AGENTS.has(step.agent)) {
            factors.push(`High-risk agent: ${step.agent}`);
            riskScore += 20;
        }

        // Check approval type if specified
        if (step.approval_type) {
            const typeRisk = this.getApprovalTypeRisk(step.approval_type);
            factors.push(`Operation type: ${step.approval_type}`);
            riskScore += typeRisk;
        }

        // Check step description for risky keywords
        const descriptionRisk = this.assessDescriptionRisk(step.description);
        if (descriptionRisk.score > 0) {
            factors.push(...descriptionRisk.factors);
            riskScore += descriptionRisk.score;
        }

        // Check result content if available
        if (stepResult) {
            const resultRisk = this.assessResultRisk(stepResult);
            if (resultRisk.score > 0) {
                factors.push(...resultRisk.factors);
                riskScore += resultRisk.score;
            }
        }

        // Check project sensitivity
        if (task.project_id.includes('prod') || task.project_id.includes('production')) {
            factors.push('Production environment detected');
            riskScore += 30;
        }

        // Determine risk level
        let level: RiskLevel;
        if (riskScore >= 50) {
            level = 'high';
        } else if (riskScore >= 20) {
            level = 'medium';
        } else {
            level = 'low';
        }

        const requires_manual_approval = riskScore >= 20 || factors.length > 2;
        const auto_approvable = riskScore < 20 && this.defaultPolicy.enabled;

        return {
            level,
            factors,
            requires_manual_approval,
            auto_approvable,
        };
    }

    /**
     * Get base risk score for approval type
     */
    private getApprovalTypeRisk(type: ApprovalType): number {
        const riskMap: Record<ApprovalType, number> = {
            file_write: 15,
            code_execution: 25,
            git_push: 20,
            workflow_creation: 30,
            api_call: 10,
        };
        return riskMap[type] ?? 10;
    }

    /**
     * Assess risk from step description
     */
    private assessDescriptionRisk(description: string): { score: number; factors: string[] } {
        const factors: string[] = [];
        let score = 0;

        const lower = description.toLowerCase();

        // Check for file operations
        for (const pattern of RISKY_PATTERNS.file_operations) {
            if (pattern.test(description)) {
                factors.push(`Risky file pattern: ${pattern.source}`);
                score += 15;
            }
        }

        // Check for command patterns
        for (const pattern of RISKY_PATTERNS.commands) {
            if (pattern.test(description)) {
                factors.push(`Risky command: ${pattern.source}`);
                score += 20;
            }
        }

        // Check for API endpoint patterns
        for (const pattern of RISKY_PATTERNS.api_endpoints) {
            if (pattern.test(description)) {
                factors.push(`Risky API operation: ${pattern.source}`);
                score += 15;
            }
        }

        // Check for destructive keywords
        if (lower.includes('delete') || lower.includes('remove') || lower.includes('drop')) {
            factors.push('Destructive operation detected');
            score += 10;
        }

        // Check for external access
        if (lower.includes('external') || lower.includes('public') || lower.includes('internet')) {
            factors.push('External access detected');
            score += 5;
        }

        return { score, factors };
    }

    /**
     * Assess risk from step result
     */
    private assessResultRisk(result: unknown): { score: number; factors: string[] } {
        const factors: string[] = [];
        let score = 0;

        if (typeof result === 'object' && result !== null) {
            const resultStr = JSON.stringify(result);

            // Check for code patterns
            for (const pattern of RISKY_PATTERNS.code_patterns) {
                if (pattern.test(resultStr)) {
                    factors.push(`Risky code pattern in result: ${pattern.source}`);
                    score += 15;
                }
            }

            // Check for file modifications
            if ('files' in result && Array.isArray((result as { files: unknown }).files)) {
                const files = (result as { files: string[] }).files;
                if (files.length > 10) {
                    factors.push(`Large number of files affected: ${files.length}`);
                    score += 10;
                }
            }

            // Check for command execution
            if ('commands' in result && Array.isArray((result as { commands: unknown }).commands)) {
                factors.push('Command execution in result');
                score += 10;
            }
        }

        return { score, factors };
    }

    /**
     * Generate detailed preview for approval request
     */
    generatePreview(_step: TodoItem, stepResult?: unknown): ApprovalPreview {
        const preview: ApprovalPreview = {};

        if (!stepResult || typeof stepResult !== 'object') {
            return preview;
        }

        const result = stepResult as Record<string, unknown>;

        // Extract files
        if ('files' in result && Array.isArray(result.files)) {
            preview.files = result.files as string[];
        } else if ('outputs' in result) {
            const outputs = result.outputs as Record<string, unknown>;
            if (outputs && typeof outputs === 'object' && 'files' in outputs) {
                preview.files = outputs.files as string[];
            }
        }

        // Extract commands
        if ('commands' in result && Array.isArray(result.commands)) {
            preview.commands = result.commands as string[];
        } else if ('outputs' in result) {
            const outputs = result.outputs as Record<string, unknown>;
            if (outputs && typeof outputs === 'object' && 'commands' in outputs) {
                preview.commands = outputs.commands as string[];
            }
        }

        // Extract API calls
        if ('api_calls' in result && Array.isArray(result.api_calls)) {
            preview.api_calls = result.api_calls as any[];
        }

        // Extract workflow config
        if ('workflow' in result && typeof result.workflow === 'object') {
            preview.workflow = result.workflow as Record<string, unknown>;
        }

        return preview;
    }

    /**
     * Determine if step should require approval
     */
    shouldRequireApproval(
        step: TodoItem,
        task: Task,
        stepResult?: unknown,
        policy: AutoApprovalPolicy = this.defaultPolicy
    ): { required: boolean; reason?: string; assessment: RiskAssessment } {
        // If step explicitly requires approval, honor it
        if (step.require_approval === true) {
            const assessment = this.assessRisk(step, task, stepResult);
            return {
                required: true,
                reason: 'Explicitly marked for approval',
                assessment,
            };
        }

        // Assess risk
        const assessment = this.assessRisk(step, task, stepResult);

        // High risk always requires approval
        if (assessment.level === 'high') {
            return {
                required: true,
                reason: `High risk operation: ${assessment.factors.join(', ')}`,
                assessment,
            };
        }

        // Check if auto-approval policy applies
        if (policy.enabled && assessment.auto_approvable) {
            if (assessment.level === 'low' ||
                (assessment.level === 'medium' && policy.max_risk_level === 'medium')) {
                return {
                    required: false,
                    reason: 'Auto-approved by policy',
                    assessment,
                };
            }
        }

        // Medium risk with manual approval factors
        if (assessment.requires_manual_approval) {
            return {
                required: true,
                reason: `Manual approval required: ${assessment.factors.join(', ')}`,
                assessment,
            };
        }

        return {
            required: false,
            assessment,
        };
    }

    /**
     * Create approval request with enhanced risk assessment
     */
    createApprovalRequest(
        step: TodoItem,
        task: Task,
        runId: string,
        stepResult?: unknown
    ): Approval {
        const assessment = this.assessRisk(step, task, stepResult);
        const preview = this.generatePreview(step, stepResult);

        const approval: Approval = {
            id: `approval-${nanoid()}`,
            run_id: runId,
            task_id: task.id,
            project_id: task.project_id,
            step_id: step.id,
            user_id: task.user_id,
            type: step.approval_type ?? this.inferApprovalType(step),
            description: this.enhanceDescription(step.description, assessment),
            risk_level: assessment.level,
            preview,
            status: 'pending',
            created_at: new Date().toISOString(),
            expires_at: this.calculateExpiration(assessment.level),
        };

        return approval;
    }

    /**
     * Infer approval type from step context
     */
    private inferApprovalType(step: TodoItem): ApprovalType {
        const desc = step.description.toLowerCase();

        if (desc.includes('file') || desc.includes('write')) return 'file_write';
        if (desc.includes('code') || desc.includes('execute')) return 'code_execution';
        if (desc.includes('git') || desc.includes('push')) return 'git_push';
        if (desc.includes('workflow')) return 'workflow_creation';
        if (desc.includes('api') || desc.includes('call')) return 'api_call';

        return 'code_execution'; // default
    }

    /**
     * Enhance description with risk factors
     */
    private enhanceDescription(description: string, assessment: RiskAssessment): string {
        if (assessment.factors.length === 0) {
            return description;
        }

        return `${description}\n\nRisk factors:\n${assessment.factors.map(f => `- ${f}`).join('\n')}`;
    }

    /**
     * Calculate expiration time based on risk level
     */
    private calculateExpiration(riskLevel: RiskLevel): string {
        const now = new Date();
        let hoursToExpire: number;

        switch (riskLevel) {
            case 'high':
                hoursToExpire = 24; // 1 day for high risk
                break;
            case 'medium':
                hoursToExpire = 8; // 8 hours for medium risk
                break;
            case 'low':
                hoursToExpire = 4; // 4 hours for low risk
                break;
        }

        now.setHours(now.getHours() + hoursToExpire);
        return now.toISOString();
    }
}

/** Singleton instance */
let approvalService: ApprovalService | null = null;

export function getApprovalService(): ApprovalService {
    if (!approvalService) {
        approvalService = new ApprovalService();
    }
    return approvalService;
}
