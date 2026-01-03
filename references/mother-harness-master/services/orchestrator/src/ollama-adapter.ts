/**
 * Ollama Adapter
 * Interfaces with local and cloud Ollama instances
 */

import { config } from './config.js';

/** Generation options */
export interface GenerationOptions {
    model?: string;
    temperature?: number;
    max_tokens?: number;
    stop?: string[];
    system?: string;
}

/** Generation result */
export interface GenerationResult {
    content: string;
    model: string;
    tokens_used: {
        prompt: number;
        completion: number;
        total: number;
    };
    duration_ms: number;
    finish_reason: 'stop' | 'length' | 'error';
}

/** Chat message */
export interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
    images?: string[]; // Base64 images for vision models
}

export class OllamaAdapter {
    private localUrl: string;
    // private cloudUrl?: string; // Unused
    // private cloudApiKey?: string; // Unused
    private defaultModel = 'gpt-oss:20b';

    constructor() {
        this.localUrl = config.ollamaLocalUrl;
        // Cloud URL and key would be configured if available
    }

    /**
     * Generate completion (single prompt)
     */
    async generate(
        prompt: string,
        options: GenerationOptions = {}
    ): Promise<GenerationResult> {
        const startTime = Date.now();
        const model = options.model ?? this.defaultModel;

        try {
            const response = await fetch(`${this.localUrl}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model,
                    prompt,
                    stream: false,
                    options: {
                        temperature: options.temperature ?? 0.7,
                        num_predict: options.max_tokens ?? 2048,
                        stop: options.stop,
                    },
                    system: options.system,
                }),
            });

            if (!response.ok) {
                throw new Error(`Ollama error: ${response.status}`);
            }

            const result = await response.json() as {
                response: string;
                model: string;
                prompt_eval_count?: number;
                eval_count?: number;
                done_reason?: string;
            };

            return {
                content: result.response,
                model: result.model,
                tokens_used: {
                    prompt: result.prompt_eval_count ?? 0,
                    completion: result.eval_count ?? 0,
                    total: (result.prompt_eval_count ?? 0) + (result.eval_count ?? 0),
                },
                duration_ms: Date.now() - startTime,
                finish_reason: result.done_reason === 'stop' ? 'stop' : 'length',
            };
        } catch (error) {
            return {
                content: '',
                model,
                tokens_used: { prompt: 0, completion: 0, total: 0 },
                duration_ms: Date.now() - startTime,
                finish_reason: 'error',
            };
        }
    }

    /**
     * Chat completion (multi-turn)
     */
    async chat(
        messages: ChatMessage[],
        options: GenerationOptions = {}
    ): Promise<GenerationResult> {
        const startTime = Date.now();
        const model = options.model ?? this.defaultModel;

        try {
            const response = await fetch(`${this.localUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model,
                    messages,
                    stream: false,
                    options: {
                        temperature: options.temperature ?? 0.7,
                        num_predict: options.max_tokens ?? 2048,
                        stop: options.stop,
                    },
                }),
            });

            if (!response.ok) {
                throw new Error(`Ollama error: ${response.status}`);
            }

            const result = await response.json() as {
                message: { content: string };
                model: string;
                prompt_eval_count?: number;
                eval_count?: number;
                done_reason?: string;
            };

            return {
                content: result.message.content,
                model: result.model,
                tokens_used: {
                    prompt: result.prompt_eval_count ?? 0,
                    completion: result.eval_count ?? 0,
                    total: (result.prompt_eval_count ?? 0) + (result.eval_count ?? 0),
                },
                duration_ms: Date.now() - startTime,
                finish_reason: result.done_reason === 'stop' ? 'stop' : 'length',
            };
        } catch (error) {
            return {
                content: '',
                model,
                tokens_used: { prompt: 0, completion: 0, total: 0 },
                duration_ms: Date.now() - startTime,
                finish_reason: 'error',
            };
        }
    }

    /**
     * Generate embeddings
     */
    async embed(
        text: string | string[],
        model: string = 'nomic-embed-text'
    ): Promise<number[][]> {
        const inputs = Array.isArray(text) ? text : [text];
        const embeddings: number[][] = [];

        for (const input of inputs) {
            try {
                const response = await fetch(`${this.localUrl}/api/embeddings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model,
                        prompt: input,
                    }),
                });

                if (!response.ok) {
                    throw new Error(`Ollama error: ${response.status}`);
                }

                const result = await response.json() as { embedding: number[] };
                embeddings.push(result.embedding);
            } catch (error) {
                // Return zero vector on error
                embeddings.push(new Array(768).fill(0));
            }
        }

        return embeddings;
    }

    /**
     * List available models
     */
    async listModels(): Promise<Array<{ name: string; size: number; modified_at: string }>> {
        try {
            const response = await fetch(`${this.localUrl}/api/tags`);

            if (!response.ok) {
                throw new Error(`Ollama error: ${response.status}`);
            }

            const result = await response.json() as {
                models: Array<{ name: string; size: number; modified_at: string }>;
            };

            return result.models;
        } catch (error) {
            return [];
        }
    }

    /**
     * Check if Ollama is available
     */
    async isAvailable(): Promise<boolean> {
        try {
            const response = await fetch(`${this.localUrl}/api/tags`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000),
            });
            return response.ok;
        } catch {
            return false;
        }
    }
}

// Singleton
let adapterInstance: OllamaAdapter | null = null;

export function getOllamaAdapter(): OllamaAdapter {
    if (!adapterInstance) {
        adapterInstance = new OllamaAdapter();
    }
    return adapterInstance;
}
