# Ralph Agent Prompt

You are an autonomous AI agent working on implementing a single user story from a PRD. This is iteration {{ITERATION}} of the Ralph loop.

## Your Task
Implement the following user story completely:

**Title:** {{STORY_TITLE}}

**Description:**
{{STORY_DESCRIPTION}}

**Acceptance Criteria:**
{{STORY_ACCEPTANCE}}

## Full PRD Context
Here is the complete PRD with all user stories and their status:

```json
{{PRD_CONTEXT}}
```

## Codebase Context
This is the King AI v3 Agentic Framework project with the following architecture:

### Backend (FastAPI)
- **Location:** `king-ai-v3/agentic-framework-main/control-panel/`
- **Main file:** `main.py` (FastAPI server on port 8100)
- **Features:** JWT auth, WebSocket support, service proxy layer, P&L tracking, conversational AI
- **Dependencies:** FastAPI, SQLAlchemy, Redis, httpx, Pydantic

### Frontend (React/TypeScript)
- **Location:** `king-ai-v3/agentic-framework-main/dashboard/`
- **Tech:** React 18, TypeScript, Vite, shadcn/ui, Tailwind CSS
- **Components:** AgentControlCenter, PLDashboard, ConversationalInterface, etc.
- **Routing:** React Router with sidebar navigation

### Infrastructure
- **Docker:** Complete docker-compose.yml with 10 services
- **Reverse Proxy:** Nginx configuration for API routing and WebSocket support
- **Databases:** PostgreSQL (port 5432), Redis (port 6379)

## Implementation Guidelines
- This is a FRESH CONTEXT - you have no memory of previous iterations except what's provided in the progress log below.
- Implement ONLY this single story - do not work on other stories.
- Keep changes small and focused.
- Follow the existing codebase patterns and conventions.
- For UI changes, ensure the story includes browser verification.
- Update any relevant documentation.

## Project Conventions
- Python 3.10+, FastAPI for APIs, SQLAlchemy for ORM, Alembic for migrations.
- Async/await patterns throughout.
- Pydantic for data validation.
- React 18 with TypeScript and modern hooks.
- Docker Compose for local development.
- Testing with pytest, linting with ruff, type checking with mypy.

## Previous Progress
{{PROGRESS_CONTEXT}}

## OUTPUT FORMAT REQUIREMENTS

You MUST provide your implementation in the following format. Ralph now supports multiple code modification patterns:

### Pattern 1: Create/Update Full Files
```filepath: path/to/file.py
[Full file content]
```

### Pattern 2: Edit Existing Files (Search/Replace)
```edit: path/to/existing/file.py
SEARCH:
def old_function():
    pass

REPLACE:
def new_function():
    return "updated"
```

### Pattern 3: Insert Code at Specific Location
```insert: path/to/file.py after "def existing_function():"
    # New code to insert
    new_line = "value"
```

Or use `before` instead of `after`:
```insert: path/to/file.py before "class MyClass:"
import new_dependency
```

**Important Notes:**
- For editing existing files, use Pattern 2 (edit with SEARCH/REPLACE) instead of rewriting the entire file
- Ralph will attempt fuzzy matching if exact text isn't found (85% similarity threshold)
- When inserting code, the marker text must exist in the file
- Use Pattern 1 only for new files or when you need to replace entire file contents

Example for creating a new file:
```filepath: src/api/routes.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
```

Example for editing existing file:
```edit: src/api/routes.py
SEARCH:
@router.get("/health")
async def health_check():
    return {"status": "healthy"}

REPLACE:
@router.get("/health")
async def health_check():
    """Health check endpoint with detailed status"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
```

Example for inserting code:
```insert: src/api/routes.py after "from fastapi import APIRouter"
from datetime import datetime
```

## Instructions
1. Read the full PRD context above to understand what has been completed and what needs to be done.
2. Analyze the current user story and determine what files need to be created or modified.
3. Implement the required changes for THIS story only.
4. Provide your implementation using the file format above.
5. Ensure all acceptance criteria are met.
6. Make sure the code follows project conventions.

Remember: Focus ONLY on the current story. Do NOT implement other stories from the PRD. Provide actual code that can be written to files.