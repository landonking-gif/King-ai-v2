/**
 * Designer Agent
 * Creates architecture diagrams, UI/UX designs, system designs
 */

import type { AgentType } from '@mother-harness/shared';
import { getLLMClient } from '@mother-harness/shared';
import { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';

/** Diagram types that can be generated */
export type DiagramType =
    | 'architecture'
    | 'sequence'
    | 'flowchart'
    | 'entity_relationship'
    | 'class_diagram'
    | 'state_machine'
    | 'component';

/** Design alternative */
interface DesignAlternative {
    name: string;
    description: string;
    pros: string[];
    cons: string[];
}

/** Diagram output */
interface Diagram {
    type: DiagramType;
    mermaid: string;
    description: string;
}

const DESIGNER_SYSTEM_PROMPT = `You are an expert software architect and system designer. Your role is to:
1. Analyze requirements and design robust, scalable systems
2. Create clear architecture diagrams using Mermaid syntax
3. Consider trade-offs and present alternatives
4. Follow industry best practices for the given domain
5. Ensure designs are practical and implementable

Be thorough in your analysis but focused in your output. Provide valid Mermaid diagrams.`;

const REQUIREMENTS_PROMPT = `Analyze this design request:

Request: {request}
Context: {context}

Return a JSON object:
{
  "design_type": "system" | "component" | "ui" | "api",
  "scope": "brief description of what to design",
  "constraints": ["list of constraints to consider"],
  "diagrams_needed": ["architecture", "sequence", "flowchart", etc.],
  "key_requirements": ["list of key requirements"],
  "technologies_suggested": ["relevant technologies"]
}`;

const DESIGN_PROMPT = `Create a comprehensive design document for:

Type: {design_type}
Scope: {scope}
Key Requirements: {requirements}
Constraints: {constraints}
Technologies: {technologies}

Return a JSON object:
{
  "document": "Full markdown design document with sections for Overview, Architecture, Components, Data Flow, API Contracts (if applicable), Security, Scalability",
  "alternatives": [
    {
      "name": "Alternative name",
      "description": "Brief description",
      "pros": ["advantage 1", "advantage 2"],
      "cons": ["disadvantage 1", "disadvantage 2"]
    }
  ],
  "decisions": ["Key architectural decisions and rationale"]
}`;

const DIAGRAM_PROMPT = `Generate a {diagram_type} diagram in Mermaid syntax for:

Design Context: {context}

Return a JSON object:
{
  "mermaid": "valid mermaid diagram code (no code fence, just the diagram)",
  "description": "what this diagram shows"
}

Rules for Mermaid:
- Use proper Mermaid syntax
- Keep diagrams readable (not too complex)
- Use meaningful node/actor names`;

export class DesignerAgent extends BaseAgent {
    readonly agentType: AgentType = 'design';
    private llm = getLLMClient();

    protected async run(inputs: string, context: AgentContext): Promise<AgentResult> {
        const startTime = Date.now();
        let totalTokens = 0;

        // Analyze design requirements
        const requirements = await this.analyzeRequirements(inputs, context);
        totalTokens += requirements.tokens;

        // Generate design document
        const design = await this.generateDesign(requirements, context);
        totalTokens += design.tokens;

        // Generate diagrams
        const diagrams = await this.generateDiagrams(
            requirements.diagrams_needed,
            requirements.scope
        );
        totalTokens += diagrams.tokens;

        return {
            success: true,
            outputs: {
                design_document: design.document,
                diagrams: diagrams.diagrams,
                alternatives: design.alternatives,
                decisions: design.decisions,
            },
            explanation: `Created ${requirements.design_type} design with ${diagrams.diagrams.length} diagrams`,
            artifacts: diagrams.diagrams.map((d, i) => `diagram_${d.type}_${i + 1}`),
            tokens_used: totalTokens,
            duration_ms: Date.now() - startTime,
        };
    }

    private async analyzeRequirements(
        inputs: string,
        context: AgentContext
    ): Promise<{
        design_type: 'system' | 'component' | 'ui' | 'api';
        scope: string;
        constraints: string[];
        diagrams_needed: DiagramType[];
        key_requirements: string[];
        technologies: string[];
        tokens: number;
    }> {
        const prompt = REQUIREMENTS_PROMPT
            .replace('{request}', inputs)
            .replace('{context}', context.recent_context ?? 'No previous context');

        const result = await this.llm.json<{
            design_type: 'system' | 'component' | 'ui' | 'api';
            scope: string;
            constraints: string[];
            diagrams_needed: DiagramType[];
            key_requirements: string[];
            technologies_suggested: string[];
        }>(prompt, {
            system: DESIGNER_SYSTEM_PROMPT,
            temperature: 0.3,
        });

        if (result.data) {
            return {
                ...result.data,
                technologies: result.data.technologies_suggested || [],
                tokens: result.raw.tokens_used.total,
            };
        }

        return {
            design_type: 'system',
            scope: inputs,
            constraints: [],
            diagrams_needed: ['architecture'],
            key_requirements: [inputs],
            technologies: [],
            tokens: result.raw.tokens_used.total,
        };
    }

    private async generateDesign(
        requirements: Awaited<ReturnType<typeof this.analyzeRequirements>>,
        _context: AgentContext
    ): Promise<{
        document: string;
        alternatives: DesignAlternative[];
        decisions: string[];
        tokens: number;
    }> {
        const prompt = DESIGN_PROMPT
            .replace('{design_type}', requirements.design_type)
            .replace('{scope}', requirements.scope)
            .replace('{requirements}', requirements.key_requirements.join(', '))
            .replace('{constraints}', requirements.constraints.join(', ') || 'None specified')
            .replace('{technologies}', requirements.technologies.join(', ') || 'To be determined');

        const result = await this.llm.json<{
            document: string;
            alternatives: DesignAlternative[];
            decisions: string[];
        }>(prompt, {
            system: DESIGNER_SYSTEM_PROMPT,
            temperature: 0.4,
            max_tokens: 4096,
        });

        if (result.data) {
            return {
                ...result.data,
                tokens: result.raw.tokens_used.total,
            };
        }

        return {
            document: `# Design Document\n\nDesign for: ${requirements.scope}\n\n*Design generation failed*`,
            alternatives: [],
            decisions: [],
            tokens: result.raw.tokens_used.total,
        };
    }

    private async generateDiagrams(
        types: DiagramType[],
        designContext: string
    ): Promise<{ diagrams: Diagram[]; tokens: number }> {
        const diagrams: Diagram[] = [];
        let totalTokens = 0;

        for (const type of types.slice(0, 3)) { // Limit to 3 diagrams
            const prompt = DIAGRAM_PROMPT
                .replace('{diagram_type}', type)
                .replace('{context}', designContext);

            const result = await this.llm.json<{
                mermaid: string;
                description: string;
            }>(prompt, {
                system: DESIGNER_SYSTEM_PROMPT,
                temperature: 0.2,
            });

            totalTokens += result.raw.tokens_used.total;

            if (result.data && result.data.mermaid) {
                diagrams.push({
                    type,
                    mermaid: result.data.mermaid,
                    description: result.data.description || `${type} diagram`,
                });
            }
        }

        return { diagrams, tokens: totalTokens };
    }
}
