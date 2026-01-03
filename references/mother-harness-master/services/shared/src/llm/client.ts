/**
 * LLM Client
 * Shared Ollama/LLM adapter for all services and agents
 */

/** Generation options */
export interface LLMOptions {
    model?: string;
    temperature?: number;
    max_tokens?: number;
    stop?: string[];
    system?: string;
    format?: 'json' | 'text';
}

/** Generation result */
export interface LLMResult {
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

/** Embedding result */
export interface EmbeddingResult {
    embeddings: number[][];
    model: string;
    dimensions: number;
}

/**
 * LLM Client - interfaces with Ollama (local) and cloud providers
 */
export class LLMClient {
    private baseUrl: string;
    private defaultModel = 'gpt-oss:20b';
    private embeddingModel = 'nomic-embed-text';

    constructor(config?: { baseUrl?: string; defaultModel?: string }) {
        this.baseUrl = config?.baseUrl ?? process.env.OLLAMA_URL ?? 'http://localhost:11434';
        if (config?.defaultModel) {
            this.defaultModel = config.defaultModel;
        }
    }

    /**
     * Chat completion (multi-turn conversation)
     */
    async chat(
        messages: ChatMessage[],
        options: LLMOptions = {}
    ): Promise<LLMResult> {
        const startTime = Date.now();
        const model = options.model ?? this.defaultModel;

        try {
            const response = await fetch(`${this.baseUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model,
                    messages,
                    stream: false,
                    format: options.format,
                    options: {
                        temperature: options.temperature ?? 0.7,
                        num_predict: options.max_tokens ?? 4096,
                        stop: options.stop,
                    },
                }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Ollama error ${response.status}: ${errorText}`);
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
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[LLMClient] Chat error: ${errorMessage}`);
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
     * Simple completion with system prompt
     */
    async complete(
        prompt: string,
        options: LLMOptions = {}
    ): Promise<LLMResult> {
        const messages: ChatMessage[] = [];

        if (options.system) {
            messages.push({ role: 'system', content: options.system });
        }
        messages.push({ role: 'user', content: prompt });

        return this.chat(messages, options);
    }

    /**
     * Generate JSON response
     */
    async json<T = unknown>(
        prompt: string,
        options: LLMOptions = {}
    ): Promise<{ data: T | null; raw: LLMResult }> {
        const result = await this.complete(prompt, {
            ...options,
            format: 'json',
        });

        if (result.finish_reason === 'error') {
            return { data: null, raw: result };
        }

        try {
            // Clean up potential markdown code blocks
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

            const data = JSON.parse(content.trim()) as T;
            return { data, raw: result };
        } catch (parseError) {
            console.error('[LLMClient] JSON parse error:', parseError);
            return { data: null, raw: result };
        }
    }

    /**
     * Generate embeddings for text
     */
    async embed(
        texts: string | string[],
        model?: string
    ): Promise<EmbeddingResult> {
        const inputs = Array.isArray(texts) ? texts : [texts];
        const embeddingModel = model ?? this.embeddingModel;
        const embeddings: number[][] = [];

        for (const input of inputs) {
            try {
                const response = await fetch(`${this.baseUrl}/api/embeddings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: embeddingModel,
                        prompt: input,
                    }),
                });

                if (!response.ok) {
                    throw new Error(`Ollama error: ${response.status}`);
                }

                const result = await response.json() as { embedding: number[] };
                embeddings.push(result.embedding);
            } catch (error) {
                console.error('[LLMClient] Embedding error:', error);
                // Return zero vector on error (768 dims for nomic-embed-text)
                embeddings.push(new Array(768).fill(0));
            }
        }

        return {
            embeddings,
            model: embeddingModel,
            dimensions: embeddings[0]?.length ?? 768,
        };
    }

    /**
     * Check if LLM service is available
     */
    async isAvailable(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseUrl}/api/tags`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000),
            });
            return response.ok;
        } catch {
            return false;
        }
    }

    /**
     * List available models
     */
    async listModels(): Promise<Array<{ name: string; size: number }>> {
        try {
            const response = await fetch(`${this.baseUrl}/api/tags`);
            if (!response.ok) return [];

            const result = await response.json() as {
                models: Array<{ name: string; size: number }>;
            };
            return result.models;
        } catch {
            return [];
        }
    }
}

// Singleton instance
let clientInstance: LLMClient | null = null;

/**
 * Get shared LLM client instance
 */
export function getLLMClient(): LLMClient {
    if (!clientInstance) {
        clientInstance = new LLMClient();
    }
    return clientInstance;
}
