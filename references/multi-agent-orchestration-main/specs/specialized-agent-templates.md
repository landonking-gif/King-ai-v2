# Plan: Specialized Agent Templates

## Task Description

Enhance the multi-agent orchestration system to support specialized agent templates. This feature allows the orchestrator to create agents based on predefined templates (subagents) stored in `.claude/agents/*.md` files. These templates define specific roles, capabilities, allowed tools, and system prompts, enabling rapid creation of purpose-built agents without manually specifying all configuration details.

## Objective

Implement a template-based agent creation system where:
1. Subagent templates are discovered and loaded from `.claude/agents/` directory
2. Orchestrator can reference templates by name when creating agents
3. Template metadata (tools, model, system prompt) is automatically applied to created agents
4. Frontend displays which template an agent was created from
5. Orchestrator system prompt dynamically includes available templates

## Problem Statement

Currently, creating specialized agents requires manually specifying:
- Complete system prompt text
- Explicit tool allowlist
- Model selection

This is repetitive, error-prone, and doesn't leverage reusable agent configurations. We need a template library system where pre-configured agent roles can be instantiated by name, similar to how Claude Code's subagent system works.

## Solution Approach

Implement a **Subagent Template Registry** that:
1. **Discovers** templates from `.claude/agents/*.md` files in the orchestrator's working directory
2. **Parses** frontmatter metadata (name, description, tools, model) using a dedicated parser
3. **Validates** template structure using Pydantic models
4. **Injects** available templates into orchestrator system prompt as `SUBAGENT_MAP` variable
5. **Applies** template configuration when `subagent_template` parameter is provided to `create_agent`
6. **Tracks** template usage in agent metadata for UI display

This approach maintains backward compatibility (templates are optional) while adding powerful composition capabilities.

## Relevant Files

### Existing Files to Modify

- **apps/orchestrator_3_stream/backend/modules/agent_manager.py** (lines 98-572)
  - Update `create_agent_tool` to accept `subagent_template` parameter
  - Modify `create_agent` method to handle template loading and application
  - Apply template's tools, model, and system prompt when template is specified

- **apps/orchestrator_3_stream/backend/modules/orchestrator_service.py** (lines 120-143)
  - Update `_load_system_prompt` to inject SUBAGENT_MAP into prompt template
  - Implement template replacement logic for dynamic subagent listing

- **apps/orchestrator_3_stream/backend/prompts/orchestrator_agent_system_prompt.md**
  - Add SUBAGENT_MAP section with template placeholder
  - Update create_agent tool documentation to include `subagent_template` parameter
  - Add usage instructions for template-based agent creation

- **apps/orchestrator_3_stream/frontend/src/components/AgentList.vue** (lines 47-61)
  - Add template badge display beneath agent name
  - Show "templated from: <name>" tag when agent uses template

- **apps/orchestrator_3_stream/frontend/src/types.d.ts**
  - Update Agent interface to include template_name field in metadata

### New Files to Create

- **apps/orchestrator_3_stream/backend/modules/subagent_loader.py**
  - SubagentTemplate Pydantic model (name, description, tools, model, color, prompt_body)
  - Frontmatter parser for `.claude/agents/*.md` files
  - SubagentRegistry class for template discovery, loading, and retrieval
  - Template validation and error handling

- **apps/orchestrator_3_stream/backend/modules/subagent_models.py**
  - Pydantic models for subagent template structure
  - Frontmatter metadata model
  - Template validation schemas

## Implementation Phases

### Phase 1: Foundation - Subagent Parsing Infrastructure
Build the core template loading system:
- Create Pydantic models for template structure
- Implement frontmatter parser for `.claude/agents/*.md` files
- Build SubagentRegistry for template discovery and caching
- Add template validation and error handling

### Phase 2: Core Implementation - Template-Based Agent Creation
Integrate templates into agent creation workflow:
- Update `create_agent` tool signature and logic
- Implement template application (tools, model, system prompt override)
- Store template metadata in agent records
- Update orchestrator system prompt with SUBAGENT_MAP

### Phase 3: Integration & Polish - UI and Documentation
Complete the feature with user-facing enhancements:
- Add template badge to AgentList.vue
- Update frontend types for template metadata
- Document template format and usage
- Test end-to-end template workflow

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom.

### 1. Create Subagent Pydantic Models

- Create `apps/orchestrator_3_stream/backend/modules/subagent_models.py`
- Define `SubagentFrontmatter` model with fields: name, description, tools (list), model (optional), color (optional)
- Define `SubagentTemplate` model with fields: frontmatter (SubagentFrontmatter), prompt_body (str), file_path (Path)
- Add validation for required fields and tool name format
- Include docstrings explaining each model's purpose

### 2. Create Frontmatter Parser and SubagentRegistry

- Create `apps/orchestrator_3_stream/backend/modules/subagent_loader.py`
- Implement `parse_subagent_file(file_path: Path) -> SubagentTemplate` function
  - Parse YAML frontmatter (lines between `---` delimiters)
  - Extract prompt body (everything after second `---`)
  - Return validated SubagentTemplate instance
  - Handle parsing errors gracefully with clear error messages
  - Log: "Parsing template from {file_path}"
  - Warn on errors: "Failed to parse template {file_path}: {error}"
- Implement `SubagentRegistry` class with methods:
  - `__init__(working_dir: str | Path, logger: OrchestratorLogger)`:
    - Initialize with orchestrator's CWD and logger instance
    - Store templates_dir as `Path(working_dir) / ".claude" / "agents"`
    - Log: "Initializing SubagentRegistry with working_dir: {working_dir}"
  - `discover_templates() -> Dict[str, SubagentTemplate]`:
    - **Check if templates_dir exists first**
    - If directory doesn't exist:
      - Warn: "⚠️  Subagent templates directory not found: {templates_dir}"
      - Warn: "💡 Create .claude/agents/ directory and add *.md template files to enable specialized agents"
      - Return empty dict
    - Scan `.claude/agents/*.md` files using glob pattern
    - Log: "Scanning for templates in {templates_dir}"
    - For each .md file found:
      - Try to parse with `parse_subagent_file()`
      - On success: Add to templates dict, log: "✓ Loaded template: {name}"
      - On failure: Warn and skip: "✗ Skipping invalid template {file}: {error}"
    - After scan complete:
      - If templates found: Log: "✅ Discovered {count} subagent template(s): {names}"
      - If no templates found: Warn: "⚠️  No valid templates found in {templates_dir}"
    - Return templates dict
  - `get_template(name: str) -> Optional[SubagentTemplate]`:
    - Retrieve template by name from cache
    - If not found, log: "Template '{name}' not found in registry"
  - `list_templates() -> List[Dict[str, str]]`:
    - Return list of {name, description} for SUBAGENT_MAP
    - If no templates: Return empty list
  - `get_available_names() -> List[str]`:
    - Return sorted list of template names (for error messages)
  - Cache templates in memory for performance
- Add comprehensive error handling for missing directories, files, invalid frontmatter, etc.

### 3. Update Agent Manager to Support Templates

- In `apps/orchestrator_3_stream/backend/modules/agent_manager.py`:
  - Add `SubagentRegistry` instance to `AgentManager.__init__`:
    - Initialize with `self.working_dir` and `self.logger`
    - Call `discover_templates()` immediately after creation
    - Store in `self.subagent_registry`
  - Update `create_agent_tool` signature (line 106):
    - Add `subagent_template` parameter: `{"name": str, "system_prompt": str, "model": str, "subagent_template": str}`
    - Update tool description to explain template parameter usage
    - Add note: "If subagent_template is provided, system_prompt can be empty (will use template's prompt)"
  - Modify `create_agent` method (line 574):
    - Add `subagent_template: Optional[str] = None` parameter
    - If `subagent_template` is provided:
      - Log: "Creating agent '{name}' using template '{subagent_template}'"
      - Fetch template: `template = self.subagent_registry.get_template(subagent_template)`
      - If template not found:
        - Get available templates: `available = self.subagent_registry.get_available_names()`
        - Return error with helpful message:
          ```python
          {
            "ok": False,
            "error": f"Template '{subagent_template}' not found. Available templates: {', '.join(available) or 'None'}"
          }
          ```
      - If template found:
        - Log: "Applying template configuration: tools={len(template.frontmatter.tools)}, model={template.frontmatter.model}"
        - Override: `system_prompt = template.prompt_body`, `model = template.frontmatter.model or model`, `allowed_tools = template.frontmatter.tools`
        - Add to metadata: `{"template_name": template.frontmatter.name, "template_color": template.frontmatter.color}`
    - Pass updated parameters to agent creation

### 4. Inject SUBAGENT_MAP into Orchestrator System Prompt

- In `apps/orchestrator_3_stream/backend/modules/orchestrator_service.py`:
  - Update `_load_system_prompt` method (line 120):
    - After loading prompt file, check for `{{SUBAGENT_MAP}}` placeholder
    - If placeholder found:
      - Initialize SubagentRegistry with `self.working_dir` and `self.logger`
      - Generate template list: `templates = registry.list_templates()`
      - Format replacement text:
        - If templates exist:
          - Format as markdown list: `- **{name}**: {description}`
          - Log: "✅ Injecting {count} subagent template(s) into orchestrator prompt"
        - If no templates:
          - Use message: "No subagent templates available. Create templates in `.claude/agents/` directory to enable specialized agents."
          - Log: "⚠️  No templates available for SUBAGENT_MAP injection"
      - Replace `{{SUBAGENT_MAP}}` with formatted text
    - Return modified prompt

### 5. Update Orchestrator System Prompt Template

- Edit `apps/orchestrator_3_stream/backend/prompts/orchestrator_agent_system_prompt.md`:
  - Add new section after Variables (around line 17):
    ```markdown
    ## Available Subagent Templates

    {{SUBAGENT_MAP}}

    Use these templates with the `subagent_template` parameter when creating agents.
    ```
  - Update `create_agent` tool documentation (around line 23):
    - Add fourth parameter: `subagent_template` (optional)
    - Explain: "Name of a subagent template to use. If provided, the template's system prompt, tools, and model will be applied automatically."
  - Add usage example:
    ```markdown
    **Example with template:**
    create_agent(
      name="code-scout",
      system_prompt="",  # Will be overridden by template
      model="sonnet",    # Can be overridden if template specifies
      subagent_template="scout-report-suggest"
    )
    ```

### 6. Update Frontend Types for Template Metadata

- Edit `apps/orchestrator_3_stream/frontend/src/types.d.ts`:
  - Update `Agent` interface to include optional template fields:
    ```typescript
    export interface Agent {
      // ... existing fields ...
      metadata?: {
        template_name?: string;
        template_color?: string;
        [key: string]: any;
      };
    }
    ```

### 7. Add Template Badge to AgentList.vue

- Edit `apps/orchestrator_3_stream/frontend/src/components/AgentList.vue`:
  - Add template badge row after agent name (around line 54):
    ```vue
    <div class="agent-name-row">
      <span class="agent-name" :style="{ borderColor: getAgentColor(agent) }">
        {{ agent.name }}
      </span>
    </div>
    <!-- NEW: Template badge -->
    <div v-if="agent.metadata?.template_name" class="template-badge">
      <span class="template-icon">📋</span>
      <span class="template-text">templated from: {{ agent.metadata.template_name }}</span>
    </div>
    ```
  - Add CSS styling in `<style scoped>` section:
    ```css
    .template-badge {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.7rem;
      color: #9ca3af;
      margin-top: 0.25rem;
      padding: 2px 6px;
      background: rgba(139, 92, 246, 0.1);
      border-radius: 3px;
      border: 1px solid rgba(139, 92, 246, 0.2);
      width: fit-content;
    }

    .template-icon {
      font-size: 0.75rem;
      line-height: 1;
    }

    .template-text {
      font-weight: 500;
      font-style: italic;
    }
    ```

### 8. Add Comprehensive Logging and Error Handling

**Implementation Checklist:**

- **In `subagent_loader.py`:**
  - `__init__`:
    - Log: "Initializing SubagentRegistry with working_dir: {working_dir}"
    - Log: "Templates directory: {templates_dir}"
  - `discover_templates()`:
    - If `.claude/agents/` doesn't exist:
      - Warn: "⚠️  Subagent templates directory not found: {templates_dir}"
      - Warn: "💡 Create .claude/agents/ directory and add *.md template files to enable specialized agents"
      - Info: "Run: mkdir -p {templates_dir}"
    - Log: "Scanning for templates in {templates_dir}"
    - For each file:
      - Debug: "Parsing template file: {file_path.name}"
      - On success: Info: "✓ Loaded template: {name} (tools: {tool_count}, model: {model})"
      - On failure: Warn: "✗ Skipping invalid template {file_path.name}: {error}"
    - After discovery:
      - If templates found: Info: "✅ Discovered {count} subagent template(s): {', '.join(names)}"
      - If no files found: Warn: "⚠️  No .md files found in {templates_dir}"
      - If files found but none valid: Error: "❌ Found {file_count} files but no valid templates"
  - `get_template()`:
    - If not found: Debug: "Template '{name}' not found in registry"
  - `parse_subagent_file()`:
    - Debug: "Reading template file: {file_path}"
    - On YAML error: Error: "Invalid YAML frontmatter in {file_path}: {error}"
    - On validation error: Error: "Template validation failed for {file_path}: {error}"

- **In `agent_manager.py`:**
  - `__init__`:
    - After registry creation: Info: "Subagent registry initialized with {count} template(s)"
    - If no templates: Warn: "⚠️  No subagent templates available. Agents must be created manually."
  - `create_agent` with template:
    - Start: Info: "Creating agent '{name}' using template '{subagent_template}'"
    - Template found: Info: "Applying template '{template.name}': {len(tools)} tools, model={model}"
    - Template not found:
      - Error: "❌ Template '{subagent_template}' not found"
      - Info: "Available templates: {', '.join(available) or 'None - create templates in .claude/agents/'}"
  - Return error format:
    ```python
    {
      "ok": False,
      "error": f"Template '{subagent_template}' not found. Available: {', '.join(available) or 'None'}",
      "suggestion": "Create templates in .claude/agents/ directory or use manual agent creation"
    }
    ```

- **In `orchestrator_service.py`:**
  - `_load_system_prompt` with SUBAGENT_MAP:
    - If placeholder found: Debug: "Found {{SUBAGENT_MAP}} placeholder in system prompt"
    - After registry init: Info: "SubagentRegistry initialized for prompt injection"
    - If templates available: Info: "✅ Injecting {count} subagent template(s) into orchestrator prompt"
    - If no templates: Warn: "⚠️  No templates available for SUBAGENT_MAP - using fallback message"

- **General Error Handling:**
  - All exceptions should be caught and logged with context
  - File operations should include try/except with clear error messages
  - YAML parsing errors should show line number if possible
  - Path errors should show absolute paths for debugging

### 9. Test Template Discovery and Loading

- Create test template: `apps/orchestrator_3_stream/.claude/agents/test-builder.md`
  ```markdown
  ---
  name: test-builder
  description: Test agent for building features
  tools: [Read, Write, Edit, Bash]
  model: haiku
  color: blue
  ---

  You are a test builder agent. Write tests for the codebase.
  ```
- Start orchestrator backend and verify:
  - Templates are discovered in logs
  - SUBAGENT_MAP appears in orchestrator prompt
  - No errors in startup logs

### 10. Test Template-Based Agent Creation

- Use orchestrator chat to create agent with template:
  ```
  "Create a scout agent using the scout-report-suggest template"
  ```
- Verify in logs:
  - Template is retrieved successfully
  - Agent is created with template's tools and system prompt
  - Agent metadata includes template_name
- Check frontend:
  - Agent appears in AgentList with template badge
  - Badge shows "templated from: scout-report-suggest"
- Test error case:
  ```
  "Create an agent with template nonexistent-template"
  ```
  - Verify error message lists available templates

### 11. Validate End-to-End Workflow

- Create multiple agents from different templates
- Verify each shows correct template badge
- Test that templated agents work correctly (send commands, check responses)
- Verify metadata persists across backend restarts (database storage)
- Test creating agent without template (backward compatibility)

## Testing Strategy

### Unit Tests
- **Frontmatter Parser**: Test parsing valid frontmatter, handling missing fields, invalid YAML
- **SubagentRegistry**: Test discovery with no templates, single template, multiple templates
- **Template Application**: Test system prompt override, tools override, model override, metadata storage

### Integration Tests
- **Agent Creation Flow**: Create agent with template, verify database record, check metadata
- **Orchestrator Prompt Injection**: Verify SUBAGENT_MAP replacement, template list formatting
- **Frontend Display**: Test template badge rendering, handling agents without templates

### Edge Cases
- **Missing .claude/agents/ directory** → Log warning, suggest creation command, continue with empty templates
- **Template file with invalid frontmatter** → Skip with warning, continue discovery
- **Template with missing required fields** → Validation error, clear message
- **Requesting nonexistent template** → Return available templates in error, suggest alternatives
- **No templates in directory** → Empty SUBAGENT_MAP with helpful message, no errors
- **Empty .claude/agents/ directory** → Warn about no .md files found
- **All templates invalid** → Warn about file count vs valid template count
- **Template with conflicting tool names** → Validate tool names exist in Claude SDK (future enhancement)

## Acceptance Criteria

- ✅ SubagentRegistry discovers and loads templates from `.claude/agents/*.md`
- ✅ Frontmatter parser correctly extracts name, description, tools, model, color
- ✅ `create_agent` tool accepts `subagent_template` parameter
- ✅ Template's system prompt, tools, and model override defaults when template is used
- ✅ Agent metadata stores template_name for reference
- ✅ Orchestrator system prompt includes SUBAGENT_MAP with available templates
- ✅ Frontend displays template badge beneath agent name
- ✅ Template badge shows "templated from: <template_name>"
- ✅ Creating agent without template still works (backward compatibility)
- ✅ Clear error messages when template not found (includes available alternatives)
- ✅ Comprehensive logging for template operations (discovery, loading, errors)
- ✅ Warnings displayed when .claude/agents/ directory doesn't exist
- ✅ Helpful suggestions provided when templates unavailable (mkdir command, setup instructions)
- ✅ Graceful degradation when no templates available (system continues to function)

## Validation Commands

Execute these commands to validate the task is complete:

1. **Start Backend and Check Template Discovery**
   ```bash
   cd apps/orchestrator_3_stream
   uv run python backend/main.py
   # Verify log output shows: "Found X subagent templates"
   ```

2. **Verify Subagent Files Exist**
   ```bash
   ls apps/orchestrator_3_stream/.claude/agents/
   # Should show: scout-report-suggest.md and other template files
   ```

3. **Test Python Modules Compile**
   ```bash
   uv run python -m py_compile apps/orchestrator_3_stream/backend/modules/subagent_loader.py
   uv run python -m py_compile apps/orchestrator_3_stream/backend/modules/subagent_models.py
   ```

4. **Test Frontend Type Checking**
   ```bash
   cd apps/orchestrator_3_stream/frontend
   npm run type-check
   # OR
   npx vue-tsc --noEmit
   ```

5. **Integration Test via UI**
   - Start frontend: `./start_fe.sh`
   - Create agent via orchestrator chat: "Create a scout agent using the scout-report-suggest template"
   - Verify agent appears with template badge in AgentList
   - Check agent metadata in browser DevTools

6. **Database Validation**
   ```bash
   # Connect to database and verify agent metadata includes template_name
   uv run python -c "
   from apps.orchestrator_3_stream.backend.modules.database import get_agent
   import asyncio
   async def check():
       agent = await get_agent('agent-uuid-here')
       print(agent.metadata)
   asyncio.run(check())
   "
   ```

7. **Validate Logging Output**

   Start backend and check logs for expected messages:

   **When .claude/agents/ exists with templates:**
   ```bash
   cd apps/orchestrator_3_stream
   uv run python backend/main.py 2>&1 | grep -E "(SubagentRegistry|template|SUBAGENT_MAP)"
   ```

   Expected log messages:
   - ✓ "Initializing SubagentRegistry with working_dir: ..."
   - ✓ "Templates directory: .../claude/agents"
   - ✓ "Scanning for templates in ..."
   - ✓ "✓ Loaded template: scout-report-suggest"
   - ✓ "✅ Discovered N subagent template(s): ..."
   - ✓ "✅ Injecting N subagent template(s) into orchestrator prompt"

   **When .claude/agents/ directory doesn't exist:**
   ```bash
   # Rename directory temporarily to test warnings
   mv apps/orchestrator_3_stream/.claude/agents apps/orchestrator_3_stream/.claude/agents.backup
   uv run python backend/main.py 2>&1 | grep -E "(template|warning|⚠️)"
   # Restore directory
   mv apps/orchestrator_3_stream/.claude/agents.backup apps/orchestrator_3_stream/.claude/agents
   ```

   Expected warning messages:
   - ✓ "⚠️  Subagent templates directory not found: ..."
   - ✓ "💡 Create .claude/agents/ directory and add *.md template files"
   - ✓ "⚠️  No subagent templates available"
   - ✓ "⚠️  No templates available for SUBAGENT_MAP"

   **When invalid template exists:**
   ```bash
   # Create invalid template
   echo "---\ninvalid yaml:\n  broken\n---" > apps/orchestrator_3_stream/.claude/agents/invalid.md
   uv run python backend/main.py 2>&1 | grep -E "(invalid|Skipping|✗)"
   # Remove test file
   rm apps/orchestrator_3_stream/.claude/agents/invalid.md
   ```

   Expected error messages:
   - ✓ "✗ Skipping invalid template invalid.md: ..."
   - ✓ "Invalid YAML frontmatter" OR "Template validation failed"

8. **Test Missing Template Error Message**
   ```bash
   # In orchestrator chat, try to create agent with nonexistent template
   # Command: "Create an agent named test using template nonexistent-template"
   ```

   Expected in tool response:
   - ❌ Error message: "Template 'nonexistent-template' not found"
   - 💡 Suggestion: "Available templates: scout-report-suggest, ..." OR "None"
   - 📝 Helpful message about creating templates in .claude/agents/

## Notes

### Dependencies
- Use Python's built-in `yaml` library for frontmatter parsing: `import yaml`
- If not available, add with: `uv add pyyaml`
- Use pathlib for cross-platform file operations

### Template Format Reference
Templates follow this structure:
```markdown
---
name: agent-name
description: Brief description
tools: [Read, Write, Edit, Bash, Grep, Glob]
model: sonnet  # or haiku, opus
color: blue    # optional, for UI theming
---

# Agent System Prompt

Detailed instructions for the agent...
```

### Backward Compatibility
- `subagent_template` parameter is optional (defaults to None)
- Existing agent creation workflows continue to work unchanged
- If both system_prompt and template are provided, template takes precedence (log warning)

### Future Enhancements (Not in Scope)
- Template versioning
- Template validation UI
- Template marketplace/sharing
- Template override parameters (e.g., add extra tools to template)
- Hot-reload templates without backend restart
