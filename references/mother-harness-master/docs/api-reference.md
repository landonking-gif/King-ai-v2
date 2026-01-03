# API Reference

## Overview

The Mother-Harness Orchestrator API provides comprehensive endpoints for managing tasks, projects, approvals, and document libraries.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://orchestrator.production.com`

## Interactive Documentation

The API includes interactive Swagger UI documentation available at:

```
http://localhost:8000/documentation
```

The Swagger UI provides:
- Complete API schema
- Interactive request/response testing
- Authentication testing
- Example requests and responses
- Schema definitions

## Authentication

All API endpoints (except `/health`) require JWT authentication.

### Getting a Token

Obtain a JWT token via Google OAuth:

```http
POST /auth/google
Content-Type: application/json

{
  "id_token": "<Google ID Token>"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "user_id": "user-abc123",
    "email": "user@example.com",
    "roles": ["user"]
  }
}
```

### Using the Token

Include the token in the `Authorization` header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

## Core Endpoints

### Tasks

#### Create Task
```http
POST /api/tasks
Content-Type: application/json
Authorization: Bearer <token>

{
  "query": "Research best practices for React testing and create a summary",
  "project_id": "project-abc123"  // optional
}
```

#### List Tasks
```http
GET /api/tasks?status=executing
Authorization: Bearer <token>
```

#### Get Task Details
```http
GET /api/tasks/task-abc123
Authorization: Bearer <token>
```

#### Execute Task
```http
POST /api/tasks/task-abc123/execute
Authorization: Bearer <token>
```

### Projects

#### Create Project
```http
POST /api/projects
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "E-commerce Platform",
  "description": "Online shopping platform development"
}
```

#### List Projects
```http
GET /api/projects
Authorization: Bearer <token>
```

### Approvals

#### Get Pending Approvals
```http
GET /api/approvals/pending
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "id": "approval-abc123",
    "task_id": "task-xyz",
    "type": "code_execution",
    "description": "Implement user authentication\n\nRisk factors:\n- High-risk agent: coder",
    "risk_level": "medium",
    "preview": {
      "files": ["src/auth.ts"],
      "commands": ["npm test"]
    },
    "status": "pending",
    "created_at": "2024-12-22T10:00:00Z",
    "expires_at": "2024-12-22T18:00:00Z"
  }
]
```

#### Respond to Approval
```http
POST /api/approvals/approval-abc123/respond
Content-Type: application/json
Authorization: Bearer <token>

{
  "approved": true,
  "notes": "Reviewed and approved after security scan"
}
```

**Permissions**: Requires `approver` or `admin` role

### Libraries

#### List Libraries
```http
GET /api/libraries?search=technical
Authorization: Bearer <token>
```

#### Create Library
```http
POST /api/libraries
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "Technical Documentation",
  "folder_path": "/libraries/tech-docs",
  "description": "Internal technical documentation repository",
  "auto_scan": true
}
```

## WebSocket

Real-time task updates via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Task update:', data);
};

// Send authentication
ws.send(JSON.stringify({
  type: 'auth',
  token: '<JWT token>'
}));

// Subscribe to task updates
ws.send(JSON.stringify({
  type: 'subscribe',
  task_id: 'task-abc123'
}));
```

## Response Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad Request - Invalid parameters |
| 401  | Unauthorized - Missing or invalid authentication |
| 403  | Forbidden - Insufficient permissions |
| 404  | Not Found - Resource doesn't exist |
| 500  | Internal Server Error |

## Rate Limiting

Currently no rate limiting is enforced, but it may be added in future versions.

## Pagination

List endpoints currently return all results. Pagination will be added in future versions for large result sets.

## Error Responses

All errors follow this format:

```json
{
  "error": "Description of the error",
  "code": "ERROR_CODE",
  "details": {
    // Additional error context
  }
}
```

## Data Models

### Task

```typescript
{
  id: string;
  project_id: string;
  user_id: string;
  description: string;
  status: 'planning' | 'executing' | 'approval_needed' | 'completed' | 'failed';
  todo_list: TodoItem[];
  steps_completed: string[];
  created_at: string;
  updated_at: string;
  completed_at?: string;
}
```

### TodoItem (Step)

```typescript
{
  id: string;
  description: string;
  agent: AgentType;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  depends_on: string[];
  require_approval?: boolean;
  approval_type?: ApprovalType;
  risk?: 'low' | 'medium' | 'high';
  result?: unknown;
  error?: string;
  started_at?: string;
  completed_at?: string;
}
```

### Agent Types

- `orchestrator` - Task planning and coordination
- `researcher` - Information gathering and research
- `coder` - Code implementation
- `design` - Architecture and design
- `analyst` - Data analysis and metrics
- `critic` - Review and validation
- `skeptic` - Challenge assumptions
- `rag` - Document retrieval
- `librarian` - Document management
- `vision` - Image and visual analysis
- `update` - Dependency updates
- `toolsmith` - Tool and integration creation

### Approval Types

- `file_write` - Writing or modifying files
- `code_execution` - Executing code
- `git_push` - Pushing to git repository
- `workflow_creation` - Creating automation workflows
- `api_call` - External API calls

## Examples

### Complete Task Workflow

```bash
# 1. Create a task
TASK_ID=$(curl -X POST http://localhost:8000/api/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze our API performance metrics"}' \
  | jq -r '.id')

# 2. Execute the task
curl -X POST http://localhost:8000/api/tasks/$TASK_ID/execute \
  -H "Authorization: Bearer $TOKEN"

# 3. Check task status
curl http://localhost:8000/api/tasks/$TASK_ID \
  -H "Authorization: Bearer $TOKEN"

# 4. If approval needed, check pending approvals
APPROVAL_ID=$(curl http://localhost:8000/api/approvals/pending \
  -H "Authorization: Bearer $TOKEN" \
  | jq -r '.[0].id')

# 5. Approve the request
curl -X POST http://localhost:8000/api/approvals/$APPROVAL_ID/respond \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "notes": "Looks good"}'
```

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:

- **YAML**: `/services/orchestrator/openapi.yaml`
- **JSON**: `http://localhost:8000/documentation/json`
- **Interactive**: `http://localhost:8000/documentation`

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-org/mother-harness/issues
- Documentation: https://docs.mother-harness.com
