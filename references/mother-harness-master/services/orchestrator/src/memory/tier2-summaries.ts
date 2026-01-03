/**
 * Tier 2 Memory - Session Summaries
 * LLM-generated summaries stored after task completion
 */

import type { SessionSummary, Project } from '@mother-harness/shared';
import { getRedisJSON, getLLMClient } from '@mother-harness/shared';
import { nanoid } from 'nanoid';

/** Maximum session summaries per project */
const MAX_SESSION_SUMMARIES = 20;

const SUMMARY_PROMPT = `Create a concise summary of this task session:

Goal: {goal}
Agents Used: {agents}
Results: {results}

Return a JSON object:
{
  "summary": "One sentence summary of what was accomplished",
  "key_findings": ["Key finding 1", "Key finding 2", "Key finding 3"],
  "decisions_made": ["Any decisions made during this session"],
  "follow_up_items": ["Items that need follow-up"]
}

Be concise - each finding should be one sentence.`;

export class Tier2Memory {
    private redis = getRedisJSON();
    private llm = getLLMClient();

    /**
     * Create a session summary after task completion using LLM
     */
    async createSummary(
        projectId: string,
        taskId: string,
        details: {
            goal: string;
            outcome: string;
            agents_invoked: { agent: string; models: string[]; tokens: number; duration: number }[];
        }
    ): Promise<SessionSummary> {
        // Generate LLM summary if we have enough content
        let llmSummary: {
            summary: string;
            key_findings: string[];
            decisions_made?: string[];
            follow_up_items?: string[];
        } | null = null;

        if (details.agents_invoked.length > 0) {
            const prompt = SUMMARY_PROMPT
                .replace('{goal}', details.goal)
                .replace('{agents}', details.agents_invoked.map(a => a.agent).join(', '))
                .replace('{results}', details.goal); // Fallback since we removed raw_results

            const result = await this.llm.json<{
                summary: string;
                key_findings: string[];
                decisions_made?: string[];
                follow_up_items?: string[];
            }>(prompt, {
                system: 'You are a session summarizer. Create concise, informative summaries.',
                temperature: 0.3,
                max_tokens: 512,
            });

            if (result.data) {
                llmSummary = result.data;
            }
        }

        const summary: SessionSummary = {
            session_id: `session-${nanoid()}`,
            started: new Date().toISOString(), // Approximate if not provided
            ended: new Date().toISOString(),
            task_ids: [taskId],
            summary: llmSummary?.summary ?? `${details.goal} - ${details.outcome}`,
            key_decisions: llmSummary?.key_findings ?? [details.outcome],
            agents_used: details.agents_invoked.map(a => a.agent as any),
            artifacts: [],
        };

        // Get current project
        const project = await this.redis.get(`project:${projectId}`) as Project | null;
        if (!project) {
            throw new Error(`Project not found: ${projectId}`);
        }

        // Add summary and trim to max
        const summaries = [...project.session_summaries, summary];
        const trimmed = summaries.slice(-MAX_SESSION_SUMMARIES);

        await this.redis.set(`project:${projectId}`, '$.session_summaries', trimmed);

        return summary;
    }

    /**
     * Get session summaries for a project
     */
    async getSummaries(projectId: string): Promise<SessionSummary[]> {
        const project = await this.redis.get(`project:${projectId}`) as Project | null;
        return project?.session_summaries ?? [];
    }

    /**
     * Get formatted context from session summaries
     */
    async getContextString(projectId: string, limit: number = 5): Promise<string> {
        const summaries = await this.getSummaries(projectId);
        const recent = summaries.slice(-limit);

        if (recent.length === 0) {
            return 'No previous session summaries.';
        }

        return recent
            .map((s, i) => {
                const agents = s.agents_used.join(', ');
                const findings = s.key_decisions.slice(0, 2).join('; ');
                return `Session ${i + 1} (${new Date(s.ended).toLocaleDateString()}):\n  Summary: ${s.summary}\n  Agents: ${agents}\n  Key Findings: ${findings}`;
            })
            .join('\n\n');
    }

    /**
     * Search summaries by keyword
     */
    async searchSummaries(projectId: string, keyword: string): Promise<SessionSummary[]> {
        const summaries = await this.getSummaries(projectId);
        const lowerKeyword = keyword.toLowerCase();

        return summaries.filter(s =>
            s.summary.toLowerCase().includes(lowerKeyword) ||
            s.key_decisions.some(f => f.toLowerCase().includes(lowerKeyword))
        );
    }

    /**
     * Get summary by task ID
     */
    async getSummaryByTask(projectId: string, taskId: string): Promise<SessionSummary | undefined> {
        const summaries = await this.getSummaries(projectId);
        return summaries.find(s => s.task_ids.includes(taskId));
    }

    /**
     * Generate a consolidated summary across multiple sessions
     */
    async generateProjectSummary(projectId: string): Promise<string> {
        const summaries = await this.getSummaries(projectId);

        if (summaries.length === 0) {
            return 'No sessions recorded for this project.';
        }

        // Compile all findings
        const allFindings = summaries.flatMap(s => s.key_decisions);
        const allAgents = [...new Set(summaries.flatMap(s => s.agents_used))];

        const prompt = `Create a brief project summary:

Sessions: ${summaries.length}
Agents Used: ${allAgents.join(', ')}
All Findings:
${allFindings.slice(0, 15).map((f, i) => `${i + 1}. ${f}`).join('\n')}

Write a 2-3 sentence summary of what has been accomplished in this project.`;

        const result = await this.llm.complete(prompt, {
            system: 'You are a project summarizer. Be concise.',
            temperature: 0.3,
            max_tokens: 256,
        });

        return result.finish_reason === 'error'
            ? `Project has ${summaries.length} sessions using ${allAgents.join(', ')}.`
            : result.content;
    }
}
