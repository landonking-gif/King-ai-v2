# Approval Workflow Documentation

## Overview

The Mother-Harness approval system provides intelligent risk assessment and gating for operations that could have significant impact. The system automatically analyzes each step's risk level and determines whether manual approval is required before proceeding.

## Features

### 1. Dynamic Risk Assessment

The approval service analyzes multiple factors to determine risk level:

- **Agent Type**: Operations from high-risk agents (coder, toolsmith) receive increased scrutiny
- **Operation Type**: Different approval types (file_write, code_execution, git_push, etc.) have different base risk scores
- **Content Analysis**: Step descriptions and results are scanned for risky patterns
- **Environment Context**: Production environments trigger higher risk scores
- **Result Inspection**: Actual execution results are analyzed for risky content

### 2. Risk Levels

#### Low Risk
- Simple read operations
- Research tasks
- Documentation queries
- Analysis without modification
- **Auto-approvable**: Yes (by default)
- **Expiration**: 4 hours

#### Medium Risk
- File write operations (non-sensitive files)
- Code execution in development environments
- API calls to internal services
- **Auto-approvable**: Configurable
- **Expiration**: 8 hours

#### High Risk
- Modification of configuration files (.env, credentials)
- Commands with sudo or elevated privileges
- Destructive operations (delete, remove, drop)
- Production environment changes
- External API calls with side effects
- **Auto-approvable**: No
- **Expiration**: 24 hours

### 3. Risky Pattern Detection

The system automatically detects dangerous patterns:

#### File Operations
- `.env`, `.git`, `.ssh` files
- Files containing: `password`, `secret`, `token`, `credentials`, `private key`

#### Commands
- `rm -rf`, `sudo`, `chmod 777`
- `curl ... | sh`, `wget ... | bash`
- `docker run`, `eval()`, `exec()`

#### Code Patterns
- `process.exit`, `child_process`, `fs.unlink`
- Dynamic code execution

#### API Endpoints
- DELETE, DESTROY, DROP, TRUNCATE operations
- Payment/billing-related endpoints

### 4. Auto-Approval Policy

Configure auto-approval rules:

```typescript
{
  enabled: true,                    // Enable auto-approval
  max_risk_level: 'low',           // Maximum risk level to auto-approve
  allowed_types: [],               // Specific approval types to auto-approve
  require_security_scan: false     // Require security scan before auto-approval
}
```

## Usage

### API Endpoints

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
    "run_id": "run-xyz",
    "task_id": "task-456",
    "type": "code_execution",
    "description": "Implement user authentication\n\nRisk factors:\n- High-risk agent: coder\n- Operation type: code_execution",
    "risk_level": "medium",
    "preview": {
      "files": ["src/auth.ts", "src/middleware/auth.ts"],
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
POST /api/approvals/:id/respond
Authorization: Bearer <token>
Content-Type: application/json

{
  "approved": true,
  "notes": "Reviewed and approved after security scan"
}
```

**Permissions**: Requires `approver` or `admin` role

**Response:**
```json
{
  "success": true
}
```

### Workflow

1. **Step Execution**: Agent completes a step
2. **Risk Assessment**: System analyzes the step and its result
3. **Approval Check**: Determines if approval is needed
4. **Auto-Approval** (if applicable): Low-risk operations proceed automatically
5. **Manual Approval** (if required): Task pauses and waits for approver
6. **Notification**: Approver receives approval request
7. **Review**: Approver reviews risk factors and preview
8. **Decision**: Approver approves or rejects
9. **Continuation**: If approved, task execution resumes

### Examples

#### Example 1: Auto-Approved Research

```typescript
{
  description: "Research best practices for React hooks",
  agent: "researcher"
}
```

**Risk Assessment**:
- Level: Low
- Auto-approvable: Yes
- **Result**: Proceeds without approval

#### Example 2: Code Execution Requiring Approval

```typescript
{
  description: "Implement user authentication with JWT",
  agent: "coder",
  result: {
    outputs: {
      files: ["src/auth.ts", "src/middleware/auth.ts"],
      commands: ["npm test", "npm run build"]
    }
  }
}
```

**Risk Assessment**:
- Level: Medium
- Factors: High-risk agent (coder), code execution
- Auto-approvable: No
- **Result**: Requires approval

#### Example 3: High-Risk Production Change

```typescript
{
  description: "Update production .env file with new API keys",
  project_id: "production-api",
  agent: "coder"
}
```

**Risk Assessment**:
- Level: High
- Factors: .env file, production environment, high-risk agent
- Auto-approvable: No
- **Result**: Requires approval with 24-hour expiration

## Configuration

### Environment-Specific Policies

You can configure different policies for different environments:

**Development**:
```typescript
{
  enabled: true,
  max_risk_level: 'medium',
  allowed_types: ['file_write', 'code_execution']
}
```

**Staging**:
```typescript
{
  enabled: true,
  max_risk_level: 'low',
  allowed_types: []
}
```

**Production**:
```typescript
{
  enabled: false,  // All operations require approval
  max_risk_level: 'low',
  allowed_types: []
}
```

## Monitoring

### Activity Stream Events

The approval system logs detailed events:

- `approval_requested`: When approval is requested
- `approval_approved`: When an approver approves
- `approval_rejected`: When an approver rejects
- `approval_expired`: When approval request expires

### Metrics

Track approval metrics:
- Approval request rate by risk level
- Average time to approval
- Auto-approval rate
- Rejection rate by type

## Security Considerations

1. **Principle of Least Privilege**: Only users with `approver` or `admin` roles can approve requests
2. **Audit Trail**: All approval decisions are logged with user identity and reasoning
3. **Expiration**: Approval requests expire based on risk level to prevent stale requests
4. **Preview Generation**: Approvers see exactly what will be executed
5. **Risk Transparency**: All risk factors are clearly communicated

## Best Practices

1. **Review Risk Factors**: Always read the risk factors before approving
2. **Check Previews**: Inspect files, commands, and API calls in the preview
3. **Add Notes**: Document your approval decision for audit purposes
4. **Time-Sensitive**: Respond to high-risk approvals within 24 hours
5. **Environment Awareness**: Apply stricter scrutiny to production changes

## Troubleshooting

### Approval Not Appearing

**Issue**: Created approval request not showing up

**Solutions**:
- Check user_id matches the approver
- Verify approval status is 'pending'
- Check if approval has expired

### False Positives

**Issue**: Low-risk operations requiring approval

**Solutions**:
- Adjust auto-approval policy
- Review step description for risky keywords
- Add explicit `require_approval: false` if appropriate

### Auto-Approval Not Working

**Issue**: Expected auto-approval but received manual approval request

**Solutions**:
- Check auto-approval policy is enabled
- Verify risk level is within max_risk_level
- Review risk assessment factors
- Confirm operation type is in allowed_types

## Future Enhancements

1. **ML-Based Risk Scoring**: Use machine learning to improve risk detection
2. **Approval Delegation**: Allow approvers to delegate to others
3. **Conditional Auto-Approval**: More sophisticated policy rules
4. **Security Scanning Integration**: Automatic vulnerability scanning before approval
5. **Approval Templates**: Pre-defined approval workflows for common scenarios
