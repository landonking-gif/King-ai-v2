# User Onboarding Quickstart Guide

## Welcome to Mother-Harness!

This guide will help you get started with the Mother-Harness multi-agent orchestration platform in under 10 minutes.

**Last Updated:** December 22, 2024
**Estimated Time:** 5-10 minutes

---

## What is Mother-Harness?

Mother-Harness is an AI-powered orchestration system that helps you:
- 🔍 **Research** complex topics across web and documents
- 💻 **Generate code** and perform software engineering tasks
- 📊 **Analyze data** and create visualizations
- 🎨 **Design** architecture and UI/UX
- ✅ **Review and validate** work with critic agents

### How It Works

1. **You ask a question** or describe a task
2. **Mother Orchestrator** creates an intelligent plan
3. **Specialist agents** execute the work (Researcher, Coder, Analyst, etc.)
4. **You review and approve** risky actions
5. **Get results** with full transparency

---

## Prerequisites

- **Web Browser** - Chrome, Firefox, Safari, or Edge
- **Google Account** - For authentication
- **Access Granted** - Your email must be in the allowed domains list

---

## Step 1: Access the Platform (2 minutes)

### 1.1 Navigate to Dashboard

Open your web browser and go to:
```
http://[your-server-address]:3000
```

**Examples:**
- Local: `http://localhost:3000`
- Server: `http://core1.example.com:3000`

### 1.2 Sign In with Google

1. Click **"Sign in with Google"**
2. Select your Google account
3. Grant permissions when prompted
4. You'll be redirected to the dashboard

**Troubleshooting:**
- **"Domain not allowed"** - Contact your administrator to add your email domain
- **"Authentication failed"** - Refresh the page and try again
- **Stuck on redirect** - Clear browser cache and cookies

---

## Step 2: Explore the Dashboard (2 minutes)

### Dashboard Overview

Once logged in, you'll see:

**Left Sidebar:**
- 🏠 **Home** - Overview and recent activity
- 📋 **Tasks** - View all your tasks
- 📁 **Projects** - Organize work into projects
- 📚 **Libraries** - Access document collections
- ✅ **Approvals** - Review pending actions

**Main Panel:**
- Task creation form
- Recent tasks list
- System activity metrics

**Top Bar:**
- Your profile
- Notification bell
- Settings (if admin)

---

## Step 3: Create Your First Task (3 minutes)

### 3.1 Simple Research Task

Let's start with a basic research task:

1. **In the task creation form**, enter:
   ```
   What are the key benefits of using TypeScript over JavaScript for large applications?
   ```

2. **Click "Create Task"**

3. **Watch the orchestrator work:**
   - Task appears in "Active Tasks"
   - Status updates in real-time
   - Orchestrator creates a plan
   - Researcher agent gathers information

4. **View results** (appears in 30-60 seconds):
   - Click on your task to see details
   - Read the research summary
   - Review sources and findings

**What Just Happened?**
- Mother Orchestrator analyzed your query
- Created a step-by-step plan
- Delegated to the Researcher agent
- Researcher gathered and synthesized information
- Results saved to your task

---

### 3.2 Try a Coding Task

Now let's try something more advanced:

1. **Create a new task:**
   ```
   Write a Python function that calculates the Fibonacci sequence up to n terms
   ```

2. **Click "Create Task" and execute**

3. **The Coder agent will:**
   - Analyze requirements
   - Generate Python code
   - Include documentation and examples
   - Provide usage instructions

4. **Review the code:**
   - Click on the task
   - See the generated function
   - Copy and use in your project

---

### 3.3 Understanding Approvals

Some tasks require your approval before execution:

**Example Task Requiring Approval:**
```
Create a git repository and push my code to GitHub
```

**What happens:**
1. Task is created and planned
2. Orchestrator identifies risky action (git operations)
3. **Approval request appears** in notifications
4. **You review:**
   - What action will be taken
   - Risk level (low/medium/high)
   - Preview of command/operation
5. **You decide:**
   - ✅ Approve - Action proceeds
   - ❌ Reject - Task is canceled
   - ⏸️ Modify - Edit and resubmit

**Approval Types:**
- **Low Risk** - Auto-approved (reading files)
- **Medium Risk** - Requires approval (writing files, API calls)
- **High Risk** - Requires approval with warning (system commands, deletions)

---

## Step 4: Working with Projects (2 minutes)

### 4.1 Create a Project

Projects help organize related tasks:

1. Go to **Projects** in sidebar
2. Click **"New Project"**
3. Enter project details:
   - **Name:** "My Web App"
   - **Description:** "Building a user authentication system"
4. Click **Create**

### 4.2 Add Tasks to Project

1. When creating a task, select your project from dropdown
2. Or, drag existing tasks into project folder
3. All project tasks appear together

**Benefits:**
- Keep related work organized
- Track project progress
- Share context between tasks
- Easy to find past work

---

## Step 5: Using Document Libraries (2 minutes)

### 5.1 Upload Documents

Libraries let you work with your own documents:

1. Go to **Libraries** in sidebar
2. Click **"New Library"**
3. Enter library name: "Company Docs"
4. **Upload documents:**
   - PDFs, Word docs, text files
   - Technical documentation
   - Research papers

5. **Documents are automatically:**
   - Parsed and indexed
   - Made searchable
   - Available to agents

### 5.2 Query Your Documents

Create a task that references your library:

```
Based on our company documentation, what is our refund policy?
```

**The RAG agent will:**
- Search your library
- Find relevant sections
- Provide answers with citations

---

## Best Practices

### ✅ Do's

- **Be specific** in task descriptions
- **Review approvals carefully** before confirming
- **Organize tasks** into projects
- **Check task status** regularly
- **Review agent outputs** thoroughly
- **Provide feedback** if results aren't right

### ❌ Don'ts

- **Don't share sensitive data** without redaction
- **Don't approve risky actions** without understanding them
- **Don't create duplicate tasks** (check existing first)
- **Don't expect instant results** (complex tasks take time)
- **Don't ignore approval requests** (they need your decision)

---

## Common Task Examples

### Research Tasks

```
Research the latest trends in serverless architecture
Summarize the key features of React 18
Find best practices for API security
```

### Coding Tasks

```
Write a REST API endpoint for user authentication
Create a React component for a login form
Generate SQL query to find duplicate records
```

### Analysis Tasks

```
Analyze this dataset and create a summary report
Identify patterns in user behavior from these logs
Compare performance metrics between two implementations
```

### Design Tasks

```
Design a database schema for a blog application
Create a system architecture for a microservices platform
Propose a UI layout for a dashboard
```

---

## Understanding Task Status

| Status | Meaning | Next Steps |
|--------|---------|------------|
| **Pending** | Waiting to start | No action needed |
| **Planning** | Creating execution plan | No action needed |
| **Running** | Actively executing | Monitor progress |
| **Approval Needed** | Waiting for your approval | Review and approve/reject |
| **Completed** | Successfully finished | Review results |
| **Failed** | Encountered an error | Check logs, retry if needed |

---

## Tips for Great Results

### 1. Provide Context

**Instead of:**
```
Write a function
```

**Try:**
```
Write a TypeScript function that validates email addresses using regex,
includes unit tests, and handles edge cases like plus-addressing
```

### 2. Break Down Complex Tasks

**Instead of:**
```
Build me a complete web application
```

**Try:**
1. "Design database schema for user management system"
2. "Create API endpoints for user CRUD operations"
3. "Build frontend login component"
4. "Add authentication middleware"

### 3. Specify Output Format

**Examples:**
```
Summarize this article in 3 bullet points
Generate code with inline comments and examples
Create a markdown report with charts and tables
```

### 4. Use Projects for Context

Tasks within a project share context:
```
Project: "E-commerce Platform"
Task 1: "Design product catalog schema"
Task 2: "Create API for product search"  # Can reference Task 1
Task 3: "Add product filtering UI"       # Can reference Tasks 1 & 2
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + K` | Quick task creation |
| `Ctrl/Cmd + P` | Open project switcher |
| `Ctrl/Cmd + /` | Show command palette |
| `Escape` | Close modal/dialog |
| `?` | Show help |

---

## Getting Help

### In-App Help

- **Help Icon (?)** in top right corner
- **Tooltips** - Hover over elements for explanations
- **Task Examples** - Click "Examples" in task creation form

### Documentation

- [API Reference](api-reference.md) - For programmatic access
- [Troubleshooting Guide](troubleshooting-runbook.md) - Common issues
- [Architecture Overview](architecture.md) - How it works

### Support

- **Email:** support@example.com
- **Slack:** #mother-harness-support
- **Status Page:** status.example.com

---

## Frequently Asked Questions

### How long do tasks take?

- **Simple research:** 30-60 seconds
- **Code generation:** 1-3 minutes
- **Complex analysis:** 5-10 minutes
- **Multi-step projects:** 10-30 minutes

### Can I cancel a running task?

Currently, tasks cannot be canceled mid-execution. They will either complete or fail. Future versions will support cancellation.

### What happens to my data?

- **Tasks:** Stored in Redis database, retained for 90 days
- **Documents:** Stored locally, never sent to external services
- **PII:** Automatically redacted in logs and analytics

### Can I use this programmatically?

Yes! See [API Reference](api-reference.md) for REST API documentation.

### What languages/frameworks are supported?

Agents support:
- **Languages:** Python, TypeScript, JavaScript, Go, Rust, Java
- **Frameworks:** React, Node.js, FastAPI, Django, Flask
- **Tools:** Git, Docker, npm, pip, SQL

### Is my work private?

- Tasks are visible only to you (and admins if needed)
- Projects can be shared with team members
- Document libraries are private by default

---

## Next Steps

### Intermediate Users

Once comfortable with basics, try:
- Creating multi-step workflows
- Using agent chaining (output of one task feeds next)
- Uploading your own documentation libraries
- Collaborating on projects with team members

### Advanced Users

Explore:
- REST API integration
- Custom agent configurations
- Batch task processing
- Metrics and analytics dashboards

---

## Onboarding Validation Checklist

Use this checklist to verify successful onboarding:

- [ ] Successfully logged in with Google authentication
- [ ] Dashboard loaded and navigated all sections
- [ ] Created first research task and viewed results
- [ ] Created coding task and reviewed generated code
- [ ] Understood approval workflow
- [ ] Created a project and added tasks to it
- [ ] Uploaded document to library (optional)
- [ ] Reviewed task status meanings
- [ ] Bookmarked documentation links
- [ ] Tested keyboard shortcuts

**Completion Time:** ________

**User Feedback:**
- What was easiest? ________________
- What was confusing? ________________
- What would you like to learn next? ________________

---

## Feedback and Improvement

We want to make onboarding better!

**Please share feedback:**
- What worked well?
- What was confusing?
- What's missing?
- How long did it take?

**Send feedback to:** product@example.com

---

## Welcome Aboard!

You're now ready to use Mother-Harness to accelerate your work. Start with simple tasks and gradually explore more advanced features.

**Remember:**
- Start simple, build complexity
- Review all approvals carefully
- Organize work into projects
- Check documentation when stuck
- Reach out for help anytime

**Happy orchestrating! 🚀**

---

*Last Updated: December 22, 2024*
*Quickstart Version: 1.0*
*Next Review: After first 10 user onboardings*
