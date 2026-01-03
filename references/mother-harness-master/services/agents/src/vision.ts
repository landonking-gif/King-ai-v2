/**
 * Vision Agent
 * Analyzes images using vision-capable LLMs
 */

import type { AgentType } from '@mother-harness/shared';
import { getLLMClient } from '@mother-harness/shared';
import { BaseAgent, type AgentContext, type AgentResult } from './base-agent.js';

/** Vision analysis types */
type AnalysisType = 'describe' | 'extract_text' | 'identify_objects' | 'analyze_diagram' | 'compare';

/** Visual Element (Object) */
export interface VisualElement {
    name: string;
    confidence: number;
    location: string;
}

const VISION_SYSTEM_PROMPT = `You are an expert image analyst. Your role is to:
1. Analyze images accurately and thoroughly
2. Extract relevant information based on the task
3. Identify objects, text, patterns, and relationships
4. Provide structured, actionable outputs
5. Be clear about what you can and cannot determine from the image

Be precise and objective in your analysis.`;

const VISION_PROMPT = `Analyze this image:

Task: {task}
Analysis Type: {analysis_type}

Additional Context: {context}

Return a JSON object:
{
  "description": "detailed description of what's in the image",
  "analysis_type_results": {
    // Results specific to the analysis type
  },
  "extracted_text": ["any text visible in the image"],
  "objects_identified": [
    { "name": "object", "confidence": 0.9, "location": "description" }
  ],
  "colors_dominant": ["primary colors in the image"],
  "recommendations": ["any recommendations based on analysis"],
  "confidence": "high" | "medium" | "low"
}`;

export class VisionAgent extends BaseAgent {
    readonly agentType: AgentType = 'vision';
    private llm = getLLMClient();

    protected async run(inputs: string, context: AgentContext): Promise<AgentResult> {
        const startTime = Date.now();

        // Parse input to extract image data and task
        const { imageData, task, analysisType } = this.parseInput(inputs);

        if (!imageData) {
            return {
                success: false,
                outputs: { error: 'No image data provided' },
                explanation: 'Vision analysis requires image data',
                tokens_used: 0,
                duration_ms: Date.now() - startTime,
            };
        }

        // Analyze the image
        const analysis = await this.analyzeImage(imageData, task, analysisType, context);

        return {
            success: true,
            outputs: {
                description: analysis.description,
                analysis_results: analysis.analysis_type_results,
                extracted_text: analysis.extracted_text,
                objects: analysis.objects_identified,
                colors: analysis.colors_dominant,
                recommendations: analysis.recommendations,
                confidence: analysis.confidence,
            },
            explanation: `Image analysis complete (${analysis.confidence} confidence)`,
            tokens_used: analysis.tokens,
            duration_ms: Date.now() - startTime,
        };
    }

    private parseInput(inputs: string): {
        imageData: string | null;
        task: string;
        analysisType: AnalysisType;
    } {
        // Try to parse as JSON first
        try {
            const parsed = JSON.parse(inputs) as {
                image?: string;
                task?: string;
                type?: AnalysisType;
            };
            return {
                imageData: parsed.image ?? null,
                task: parsed.task ?? 'Describe this image',
                analysisType: parsed.type ?? 'describe',
            };
        } catch {
            // Treat as plain text task
            return {
                imageData: null,
                task: inputs,
                analysisType: 'describe',
            };
        }
    }

    private async analyzeImage(
        imageData: string,
        task: string,
        analysisType: AnalysisType,
        context: AgentContext
    ): Promise<{
        description: string;
        analysis_type_results: Record<string, unknown>;
        extracted_text: string[];
        objects_identified: VisualElement[];
        colors_dominant: string[];
        recommendations: string[];
        confidence: 'high' | 'medium' | 'low';
        tokens: number;
    }> {
        const prompt = VISION_PROMPT
            .replace('{task}', task)
            .replace('{analysis_type}', analysisType)
            .replace('{context}', context.recent_context ?? 'No additional context');

        // Use chat with images for vision models
        const result = await this.llm.chat(
            [
                { role: 'system', content: VISION_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: prompt,
                    images: [imageData] // Base64 image data
                }
            ],
            {
                model: 'llava:13b', // Vision-capable model
                temperature: 0.3,
                max_tokens: 2048,
            }
        );

        // Try to parse JSON from response
        try {
            let content = result.content.trim();
            if (content.startsWith('```json')) {
                content = content.slice(7);
            }
            if (content.startsWith('```')) {
                content = content.slice(3);
            }
            if (content.endsWith('```')) {
                content = content.slice(0, -3);
            }

            const parsed = JSON.parse(content.trim()) as {
                description: string;
                analysis_type_results: Record<string, unknown>;
                extracted_text: string[];
                objects_identified: Array<{ name: string; confidence: number; location: string }>;
                colors_dominant: string[];
                recommendations: string[];
                confidence: 'high' | 'medium' | 'low';
            };

            return {
                ...parsed,
                tokens: result.tokens_used.total,
            };
        } catch {
            // Return raw description if JSON parsing fails
            return {
                description: result.content,
                analysis_type_results: {},
                extracted_text: [],
                objects_identified: [],
                colors_dominant: [],
                recommendations: [],
                confidence: 'medium',
                tokens: result.tokens_used.total,
            };
        }
    }
}
