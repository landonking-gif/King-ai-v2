/**
 * Update Agent
 * Discovers and recommends software updates with impact analysis
 */

import type { AgentType } from '@mother-harness/shared';
import { getLLMClient } from '@mother-harness/shared';
import { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';

/** Software inventory item */
export interface SoftwareItem {
    name: string;
    type: 'dependency' | 'tool' | 'runtime' | 'os' | 'service';
    current_version: string;
    source: string;
}

/** Update recommendation */
export interface UpdateRecommendation {
    item: SoftwareItem;
    latest_version: string;
    update_urgency: 'critical' | 'high' | 'medium' | 'low';
    breaking_changes: boolean;
    security_related: boolean;
    summary: string;
    migration_notes: string[];
    rollback_plan: string;
}

const UPDATE_SYSTEM_PROMPT = `You are a software update specialist. Your role is to:
1. Analyze software dependencies and tools for updates
2. Assess the impact and urgency of updates
3. Identify security-related updates
4. Provide clear migration guidance
5. Suggest rollback strategies

Prioritize security updates. Be realistic about breaking changes.`;

const ANALYZE_UPDATES_PROMPT = `Analyze these software items for updates:

Items:
{items}

Context: {context}

For each item, return a JSON array:
[
  {
    "name": "package name",
    "current_version": "X.Y.Z",
    "latest_version": "A.B.C",
    "update_urgency": "critical" | "high" | "medium" | "low",
    "breaking_changes": true/false,
    "security_related": true/false,
    "summary": "brief summary of what changed",
    "key_changes": ["notable changes"],
    "migration_notes": ["steps needed to update"],
    "rollback_plan": "how to rollback if needed"
  }
]

Be realistic about version numbers based on your knowledge.`;

export class UpdateAgent extends BaseAgent {
    readonly agentType: AgentType = 'update';
    private llm = getLLMClient();

    protected async run(inputs: string, context: AgentContext): Promise<AgentResult> {
        const startTime = Date.now();
        let totalTokens = 0;

        // Discover software inventory
        const inventory = await this.discoverInventory(inputs, context);
        totalTokens += inventory.tokens;

        if (inventory.items.length === 0) {
            return {
                success: true,
                outputs: {
                    update_recommendations: [],
                    inventory: [],
                    recommendations: [],
                    summary: {
                        total_items: 0,
                        updates_available: 0,
                        critical_updates: 0,
                        security_updates: 0,
                        breaking_updates: 0,
                    },
                    priority_updates: [],
                    message: 'No software items found to check',
                },
                explanation: 'No items in inventory',
                tokens_used: totalTokens,
                duration_ms: Date.now() - startTime,
            };
        }

        // Check for updates
        const updates = await this.checkUpdates(inventory.items, context);
        totalTokens += updates.tokens;

        // Categorize by urgency
        const critical = updates.recommendations.filter(u => u.update_urgency === 'critical');
        const security = updates.recommendations.filter(u => u.security_related);
        const breaking = updates.recommendations.filter(u => u.breaking_changes);

        const migrationSteps = updates.recommendations
            .flatMap(u => u.migration_notes)
            .filter((note, idx, arr) => arr.indexOf(note) === idx);

        return {
            success: true,
            outputs: {
                inventory: inventory.items,
                recommendations: updates.recommendations,
                update_recommendations: updates.recommendations,
                summary: {
                    total_items: inventory.items.length,
                    updates_available: updates.recommendations.length,
                    critical_updates: critical.length,
                    security_updates: security.length,
                    breaking_updates: breaking.length,
                },
                priority_updates: critical.concat(
                    security.filter(s => !critical.includes(s))
                ),
                breaking_changes: breaking.map(b => ({
                    name: b.item.name,
                    current_version: b.item.current_version,
                    latest_version: b.latest_version,
                    summary: b.summary,
                })),
                migration_steps: migrationSteps,
            },
            explanation: `Found ${updates.recommendations.length} updates (${critical.length} critical, ${security.length} security-related)`,
            tokens_used: totalTokens,
            duration_ms: Date.now() - startTime,
        };
    }

    private async discoverInventory(
        inputs: string,
        context: AgentContext
    ): Promise<{ items: SoftwareItem[]; tokens: number }> {
        // Try to parse input as inventory
        try {
            const parsed = JSON.parse(inputs) as SoftwareItem[];
            if (Array.isArray(parsed)) {
                return { items: parsed, tokens: 0 };
            }
        } catch {
            // Not JSON, use LLM to parse
        }

        const prompt = `Extract software items from this input:

Input: ${inputs}
Context: ${context.recent_context ?? 'No context'}

Return a JSON array:
[
  {
    "name": "package/tool name",
    "type": "dependency" | "tool" | "runtime" | "os" | "service",
    "current_version": "version if known, or 'unknown'",
    "source": "where it comes from (npm, pip, system, etc.)"
  }
]`;

        const result = await this.llm.json<SoftwareItem[]>(prompt, {
            system: UPDATE_SYSTEM_PROMPT,
            temperature: 0.2,
        });

        return {
            items: result.data ?? [],
            tokens: result.raw.tokens_used.total,
        };
    }

    private async checkUpdates(
        items: SoftwareItem[],
        context: AgentContext
    ): Promise<{ recommendations: UpdateRecommendation[]; tokens: number }> {
        const itemsJson = JSON.stringify(items, null, 2);

        const prompt = ANALYZE_UPDATES_PROMPT
            .replace('{items}', itemsJson)
            .replace('{context}', context.recent_context ?? 'No additional context');

        const result = await this.llm.json<Array<{
            name: string;
            current_version: string;
            latest_version: string;
            update_urgency: 'critical' | 'high' | 'medium' | 'low';
            breaking_changes: boolean;
            security_related: boolean;
            summary: string;
            key_changes?: string[];
            migration_notes: string[];
            rollback_plan: string;
        }>>(prompt, {
            system: UPDATE_SYSTEM_PROMPT,
            temperature: 0.3,
            max_tokens: 4096,
        });

        if (!result.data || !Array.isArray(result.data)) {
            return { recommendations: [], tokens: result.raw.tokens_used.total };
        }

        // Map back to items
        const recommendations: UpdateRecommendation[] = result.data
            .filter(update => update.current_version !== update.latest_version)
            .map(update => {
                const item = items.find(i => i.name === update.name) ?? {
                    name: update.name,
                    type: 'dependency' as const,
                    current_version: update.current_version,
                    source: 'unknown',
                };

                return {
                    item,
                    latest_version: update.latest_version,
                    update_urgency: update.update_urgency,
                    breaking_changes: update.breaking_changes,
                    security_related: update.security_related,
                    summary: update.summary,
                    migration_notes: update.migration_notes,
                    rollback_plan: update.rollback_plan,
                };
            });

        return {
            recommendations,
            tokens: result.raw.tokens_used.total,
        };
    }
}
