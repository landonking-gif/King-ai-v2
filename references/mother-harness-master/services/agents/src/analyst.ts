/**
 * Analyst Agent
 * Analyzes data, generates reports and visualizations
 */

import type { AgentType } from '@mother-harness/shared';
import { getLLMClient } from '@mother-harness/shared';
import { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';

/** Data analysis result */
export interface AnalysisResult {
    summary: string;
    insights: string[];
    data_points: Array<{ label: string; value: number | string; trend?: string }>;
    trends?: string[];
    recommendations?: string[];
    confidence: 'high' | 'medium' | 'low';
}

/** Chart configuration */
interface ChartConfig {
    type: 'bar' | 'line' | 'pie' | 'scatter' | 'table';
    title: string;
    data: Array<{ label: string; value: number | string }>;
    config: Record<string, unknown>;
}

const ANALYST_SYSTEM_PROMPT = `You are an expert data analyst. Your role is to:
1. Analyze data and extract meaningful insights
2. Identify trends, patterns, and anomalies
3. Provide data-driven recommendations
4. Present findings in clear, actionable formats
5. Be honest about data limitations and confidence levels

Focus on insights that drive decision-making. Be quantitative when possible.`;

const ANALYSIS_PROMPT = `Analyze this data/request:

Query: {query}
Context: {context}
Available Data: {data_context}

Return a JSON object:
{
  "summary": "Brief executive summary of findings",
  "insights": [
    "Key insight 1 with supporting data",
    "Key insight 2 with supporting data",
    "Key insight 3 with supporting data"
  ],
  "data_points": [
    { "label": "Metric name", "value": 123, "trend": "up/down/stable" }
  ],
  "trends": ["Observed trend 1", "Observed trend 2"],
  "recommendations": [
    "Data-driven recommendation 1",
    "Data-driven recommendation 2"
  ],
  "confidence": "high" | "medium" | "low",
  "limitations": "Any caveats or limitations"
}`;

const VISUALIZATION_PROMPT = `Based on this analysis, suggest appropriate visualizations:

Data Points: {data_points}
Insights: {insights}

Return a JSON array of chart configurations:
[
  {
    "type": "bar" | "line" | "pie" | "scatter" | "table",
    "title": "Chart title",
    "description": "What this visualization shows",
    "x_axis": "label field",
    "y_axis": "value field",
    "recommended_for": "which insight this supports"
  }
]`;

export class AnalystAgent extends BaseAgent {
    readonly agentType: AgentType = 'analyst';
    private llm = getLLMClient();

    protected async run(inputs: string, context: AgentContext): Promise<AgentResult> {
        const startTime = Date.now();
        let totalTokens = 0;

        // Parse and enhance request
        const request = this.parseRequest(inputs);

        // Conduct analysis
        const analysis = await this.conductAnalysis(request, context);
        totalTokens += analysis.tokens;

        // Generate visualizations if needed
        // Generate visualizations if needed
        const visualizations = request.needs_viz
            ? await this.generateVisualizations(analysis.result)
            : { charts: [], tokens: 0 };
        totalTokens += visualizations.tokens;

        return {
            success: true,
            outputs: {
                analysis_report: this.formatReport(analysis.result),
                raw_analysis: analysis.result,
                visualizations: visualizations.charts,
                data_exports: request.export_data ? analysis.result.data_points : undefined,
            },
            explanation: `Analysis complete: ${analysis.result.insights.length} insights, confidence: ${analysis.result.confidence}`,
            tokens_used: totalTokens,
            duration_ms: Date.now() - startTime,
        };
    }

    private parseRequest(inputs: string): {
        query: string;
        data_source?: string;
        needs_viz: boolean;
        export_data: boolean;
    } {
        const lower = inputs.toLowerCase();
        return {
            query: inputs,
            needs_viz: lower.includes('chart') || lower.includes('graph') || lower.includes('visualiz'),
            export_data: lower.includes('export') || lower.includes('download'),
        };
    }

    private async conductAnalysis(
        request: ReturnType<typeof this.parseRequest>,
        context: AgentContext
    ): Promise<{ result: AnalysisResult; tokens: number }> {
        const prompt = ANALYSIS_PROMPT
            .replace('{query}', request.query)
            .replace('{context}', context.recent_context ?? 'No previous context')
            .replace('{data_context}', context.rag_context ?? 'No specific data provided');

        const result = await this.llm.json<AnalysisResult & { limitations?: string }>(prompt, {
            system: ANALYST_SYSTEM_PROMPT,
            temperature: 0.3,
            max_tokens: 4096,
        });

        if (result.data) {
            return {
                result: {
                    summary: result.data.summary + (result.data.limitations
                        ? `\n\n_Note: ${result.data.limitations}_`
                        : ''),
                    insights: result.data.insights || [],
                    data_points: result.data.data_points || [],
                    ...(result.data.trends !== undefined && { trends: result.data.trends }),
                    ...(result.data.recommendations !== undefined && { recommendations: result.data.recommendations }),
                    confidence: result.data.confidence || 'medium',
                },
                tokens: result.raw.tokens_used.total,
            };
        }

        return {
            result: {
                summary: 'Analysis could not be completed.',
                insights: ['Unable to analyze the provided data/request'],
                data_points: [],
                confidence: 'low',
            },
            tokens: result.raw?.tokens_used?.total ?? 0,
        };
    }

    private async generateVisualizations(
        analysis: AnalysisResult
    ): Promise<{ charts: ChartConfig[]; tokens: number }> {
        if (analysis.data_points.length === 0) {
            return { charts: [], tokens: 0 };
        }

        const prompt = VISUALIZATION_PROMPT
            .replace('{data_points}', JSON.stringify(analysis.data_points))
            .replace('{insights}', analysis.insights.join('\n'));

        const result = await this.llm.json<Array<{
            type: 'bar' | 'line' | 'pie' | 'scatter' | 'table';
            title: string;
            description?: string;
            x_axis?: string;
            y_axis?: string;
        }>>(prompt, {
            system: ANALYST_SYSTEM_PROMPT,
            temperature: 0.2,
        });

        const charts: ChartConfig[] = [];
        if (result.data && Array.isArray(result.data)) {
            for (const viz of result.data.slice(0, 3)) {
                charts.push({
                    type: viz.type,
                    title: viz.title,
                    data: analysis.data_points.map(d => ({ label: d.label, value: d.value })),
                    config: {
                        x_axis: viz.x_axis || 'label',
                        y_axis: viz.y_axis || 'value',
                        description: viz.description,
                    },
                });
            }
        }

        return { charts, tokens: result.raw.tokens_used.total };
    }

    private formatReport(analysis: AnalysisResult): string {
        return `# Analysis Report

## Executive Summary
${analysis.summary}

## Confidence Level
**${analysis.confidence.toUpperCase()}**

## Key Insights
${analysis.insights.map((i, idx) => `${idx + 1}. ${i}`).join('\n')}

## Data Points
| Metric | Value | Trend |
|--------|-------|-------|
${analysis.data_points.map(d => `| ${d.label} | ${d.value} | ${d.trend ?? '-'} |`).join('\n')}

${analysis.trends && analysis.trends.length > 0 ? `## Trends\n${analysis.trends.map(t => `- ${t}`).join('\n')}` : ''}

${analysis.recommendations && analysis.recommendations.length > 0 ? `## Recommendations\n${analysis.recommendations.map(r => `- ${r}`).join('\n')}` : ''}
`;
    }
}
