/**
 * Researcher Agent
 * Conducts web research, synthesizes findings with citations
 */

import type { AgentType } from '@mother-harness/shared';
import { getLLMClient } from '@mother-harness/shared';
import { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';

/** Citation structure */
interface Citation {
    title: string;
    url: string;
    snippet: string;
    relevance?: number;
}

/** Research result */
interface ResearchResult {
    summary: string;
    citations: Citation[];
    findings: string[];
    recommendations: string[];
    tokens: number;
}

const RESEARCHER_SYSTEM_PROMPT = `You are an expert research analyst. Your role is to:
1. Thoroughly analyze research queries
2. Synthesize information from multiple perspectives
3. Provide well-structured findings with evidence
4. Make actionable recommendations based on findings
5. Always cite sources when making claims

Be thorough but concise. Focus on facts and evidence.`;

const RESEARCH_PROMPT = `Research Query: {query}

Context from previous conversation:
{context}

RAG Context (if available):
{rag_context}

Based on your knowledge and the context provided, conduct research on this topic.

Return a JSON object with:
{
  "summary": "A comprehensive summary of findings (2-3 paragraphs)",
  "key_findings": [
    "Finding 1 with supporting evidence",
    "Finding 2 with supporting evidence",
    "Finding 3 with supporting evidence"
  ],
  "recommendations": [
    "Actionable recommendation 1",
    "Actionable recommendation 2"
  ],
  "sources": [
    {
      "title": "Source title",
      "url": "https://example.com (use plausible URLs based on topic)",
      "snippet": "Relevant quote or information from this source"
    }
  ],
  "confidence": "high" | "medium" | "low",
  "limitations": "Any limitations or caveats about this research"
}`;

export class ResearcherAgent extends BaseAgent {
    readonly agentType: AgentType = 'researcher';
    private llm = getLLMClient();

    protected async run(inputs: string, context: AgentContext): Promise<AgentResult> {
        const startTime = Date.now();

        // Conduct research using LLM
        const researchResult = await this.conductResearch(inputs, context);

        return {
            success: researchResult.findings.length > 0,
            outputs: {
                research_summary: researchResult.summary,
                citations: researchResult.citations,
                key_findings: researchResult.findings,
                recommendations: researchResult.recommendations,
            },
            explanation: `Researched: "${inputs}" - Found ${researchResult.findings.length} key findings`,
            sources: researchResult.citations.map(c => c.url),
            tokens_used: researchResult.tokens,
            duration_ms: Date.now() - startTime,
        };
    }

    private async conductResearch(
        query: string,
        context: AgentContext
    ): Promise<ResearchResult> {
        const prompt = RESEARCH_PROMPT
            .replace('{query}', query)
            .replace('{context}', context.recent_context ?? 'No previous context')
            .replace('{rag_context}', context.rag_context ?? 'No RAG context available');

        const result = await this.llm.json<{
            summary: string;
            key_findings: string[];
            recommendations: string[];
            sources: Citation[];
            confidence?: string;
            limitations?: string;
        }>(prompt, {
            system: RESEARCHER_SYSTEM_PROMPT,
            temperature: 0.4, // Moderate temperature for balanced creativity/accuracy
            max_tokens: 4096,
        });

        if (result.data) {
            return {
                summary: result.data.summary + (result.data.limitations
                    ? `\n\n**Limitations**: ${result.data.limitations}`
                    : ''),
                citations: result.data.sources || [],
                findings: result.data.key_findings || [],
                recommendations: result.data.recommendations || [],
                tokens: result.raw.tokens_used.total,
            };
        }

        // Fallback if LLM fails
        console.warn('[ResearcherAgent] Research failed, returning error');
        return {
            summary: `Unable to complete research for: "${query}". The LLM request failed.`,
            citations: [],
            findings: ['Research could not be completed due to a processing error'],
            recommendations: ['Retry the research request', 'Check LLM service availability'],
            tokens: result.raw.tokens_used.total,
        };
    }
}
