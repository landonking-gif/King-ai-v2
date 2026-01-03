/**
 * Template Library
 * Reusable prompt templates with variables
 */

import { getRedisJSON } from '../redis/index.js';
import { nanoid } from 'nanoid';

/** Template variable */
export interface TemplateVariable {
    name: string;
    type: 'string' | 'number' | 'boolean' | 'select' | 'text';
    description: string;
    required: boolean;
    default?: string | number | boolean;
    options?: string[];  // For select type
}

/** Prompt template */
export interface PromptTemplate {
    id: string;
    name: string;
    description: string;
    category: string;

    // Template content with {{variable}} placeholders
    template: string;
    variables: TemplateVariable[];

    // Targeting
    target_agents?: string[];  // Agents this template is designed for

    // Stats
    use_count: number;
    last_used_at?: string;
    avg_rating?: number;

    // Access
    owner_id: string;
    shared: boolean;

    // Metadata
    created_at: string;
    updated_at: string;
}

export class TemplateLibrary {
    private redis = getRedisJSON();
    private readonly prefix = 'template:';

    /**
     * Create a new template
     */
    async createTemplate(
        ownerId: string,
        data: {
            name: string;
            description: string;
            category: string;
            template: string;
            variables?: TemplateVariable[];
            target_agents?: string[];
            shared?: boolean;
        }
    ): Promise<PromptTemplate> {
        const now = new Date().toISOString();

        // Auto-detect variables from template
        const detectedVars = this.detectVariables(data.template);
        const variables = data.variables ?? detectedVars.map(v => ({
            name: v,
            type: 'string' as const,
            description: '',
            required: true,
        }));

        const template: PromptTemplate = {
            id: `tmpl-${nanoid()}`,
            name: data.name,
            description: data.description,
            category: data.category,
            template: data.template,
            variables,
            ...(data.target_agents !== undefined && { target_agents: data.target_agents }),
            use_count: 0,
            owner_id: ownerId,
            shared: data.shared ?? false,
            created_at: now,
            updated_at: now,
        };

        await this.redis.set(`${this.prefix}${template.id}`, '$', template);

        return template;
    }

    /**
     * Get a template by ID
     */
    async getTemplate(templateId: string): Promise<PromptTemplate | null> {
        return await this.redis.get<PromptTemplate>(`${this.prefix}${templateId}`);
    }

    /**
     * Render a template with variable values
     */
    async renderTemplate(
        templateId: string,
        variables: Record<string, string | number | boolean>
    ): Promise<{ success: boolean; result?: string; error?: string }> {
        const template = await this.getTemplate(templateId);
        if (!template) {
            return { success: false, error: 'Template not found' };
        }

        // Check required variables
        for (const v of template.variables) {
            if (v.required && !(v.name in variables) && v.default === undefined) {
                return { success: false, error: `Missing required variable: ${v.name}` };
            }
        }

        // Apply defaults and validate
        const values: Record<string, string | number | boolean> = {};
        for (const v of template.variables) {
            if (v.name in variables) {
                values[v.name] = variables[v.name]!;
            } else if (v.default !== undefined) {
                values[v.name] = v.default;
            }

            // Validate select options
            if (v.type === 'select' && v.options) {
                const val = values[v.name];
                if (val !== undefined && !v.options.includes(String(val))) {
                    return { success: false, error: `Invalid option for ${v.name}: ${val}` };
                }
            }
        }

        // Render template
        let result = template.template;
        for (const [key, value] of Object.entries(values)) {
            result = result.replace(new RegExp(`{{${key}}}`, 'g'), String(value));
        }

        // Update stats
        template.use_count++;
        template.last_used_at = new Date().toISOString();
        await this.redis.set(`${this.prefix}${templateId}`, '$', template);

        return { success: true, result };
    }

    /**
     * Find templates
     */
    async findTemplates(options: {
        owner_id?: string;
        category?: string;
        target_agent?: string;
        include_shared?: boolean;
        search?: string;
    }): Promise<PromptTemplate[]> {
        const keys = await this.redis.keys(`${this.prefix}*`);
        const templates: PromptTemplate[] = [];

        for (const key of keys) {
            const template = await this.redis.get<PromptTemplate>(key);
            if (!template) continue;

            // Filter by access
            const canAccess = template.owner_id === options.owner_id ||
                (options.include_shared && template.shared);
            if (!canAccess) continue;

            // Filter by category
            if (options.category && template.category !== options.category) continue;

            // Filter by target agent
            if (options.target_agent && template.target_agents) {
                if (!template.target_agents.includes(options.target_agent)) continue;
            }

            // Search filter
            if (options.search) {
                const searchLower = options.search.toLowerCase();
                const matchesName = template.name.toLowerCase().includes(searchLower);
                const matchesDesc = template.description.toLowerCase().includes(searchLower);
                if (!matchesName && !matchesDesc) continue;
            }

            templates.push(template);
        }

        return templates.sort((a, b) => b.use_count - a.use_count);
    }

    /**
     * Get categories
     */
    async getCategories(ownerId: string): Promise<string[]> {
        const templates = await this.findTemplates({ owner_id: ownerId, include_shared: true });
        const categories = new Set<string>();

        for (const t of templates) {
            categories.add(t.category);
        }

        return [...categories].sort();
    }

    /**
     * Detect variables in template string
     */
    private detectVariables(template: string): string[] {
        const pattern = /{{(\w+)}}/g;
        const variables: string[] = [];
        let match;

        while ((match = pattern.exec(template)) !== null) {
            const varName = match[1];
            if (varName && !variables.includes(varName)) {
                variables.push(varName);
            }
        }

        return variables;
    }

    /**
     * Built-in templates
     */
    static readonly BUILTIN_TEMPLATES = [
        {
            name: 'Research Summary',
            description: 'Comprehensive research on a topic with sources',
            category: 'research',
            template: `Research the following topic thoroughly: {{topic}}

Focus areas:
{{focus_areas}}

Requirements:
- Provide at least 5 authoritative sources
- Include recent developments (last {{timeframe}})
- Highlight key findings and implications
- Note any controversies or debates

Format the response with clear sections and citations.`,
            variables: [
                { name: 'topic', type: 'string' as const, description: 'Main research topic', required: true },
                { name: 'focus_areas', type: 'text' as const, description: 'Specific areas to focus on', required: false, default: 'General overview' },
                { name: 'timeframe', type: 'select' as const, description: 'Time range for recent developments', required: false, default: '1 year', options: ['6 months', '1 year', '2 years', '5 years'] },
            ],
            target_agents: ['researcher'],
        },
        {
            name: 'Code Review',
            description: 'Thorough code review with security and quality checks',
            category: 'development',
            template: `Review the following {{language}} code:

\`\`\`{{language}}
{{code}}
\`\`\`

Check for:
1. Security vulnerabilities
2. Performance issues
3. Code quality and maintainability
4. Best practices adherence
5. Potential bugs

Suggest specific improvements with code examples.`,
            variables: [
                { name: 'language', type: 'select' as const, description: 'Programming language', required: true, options: ['typescript', 'python', 'go', 'rust', 'java'] },
                { name: 'code', type: 'text' as const, description: 'Code to review', required: true },
            ],
            target_agents: ['critic', 'coder'],
        },
    ];
}

// Singleton
let libraryInstance: TemplateLibrary | null = null;

export function getTemplateLibrary(): TemplateLibrary {
    if (!libraryInstance) {
        libraryInstance = new TemplateLibrary();
    }
    return libraryInstance;
}
