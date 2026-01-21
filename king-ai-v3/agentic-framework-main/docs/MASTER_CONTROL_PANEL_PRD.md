# Product Requirements Document (PRD)
# King AI v3 - Agentic Framework Master Control Panel

**Version:** 1.0.0  
**Date:** January 20, 2026  
**Status:** Draft  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [System Architecture](#3-system-architecture)
4. [Core Dashboards & Features](#4-core-dashboards--features)
5. [API Integration Requirements](#5-api-integration-requirements)
6. [Data Models & State Management](#6-data-models--state-management)
7. [User Interface Specifications](#7-user-interface-specifications)
8. [Security & Authentication](#8-security--authentication)
9. [Performance Requirements](#9-performance-requirements)
10. [Deployment & Infrastructure](#10-deployment--infrastructure)
11. [Implementation Phases](#11-implementation-phases)

---

## 1. Executive Summary

### 1.1 Vision Statement

Create a **Master Control Panel** - a comprehensive, real-time web-based dashboard that serves as the single pane of glass for the King AI v3 Agentic Framework. This control panel will expose 100% of the framework's capabilities through an intuitive, visually rich interface.

### 1.2 Goals

| Goal | Description | Success Metric |
|------|-------------|----------------|
| **Complete Visibility** | Expose all 5 microservices, CLI, and SDK capabilities | 100% API coverage in UI |
| **Real-Time Monitoring** | Live updates for all system activities | < 500ms update latency |
| **Operational Efficiency** | Single interface for all operations | 80% reduction in CLI usage |
| **Decision Support** | AI-powered insights and recommendations | Actionable insights per workflow |
| **Governance Compliance** | Full audit trail and approval workflows | 100% provenance tracking |

### 1.3 Target Services

| Service | Port | Purpose |
|---------|------|---------|
| **Orchestrator** | 8000 | Workflow execution, subagent coordination |
| **Subagent Manager** | 8001 | Agent lifecycle, governance |
| **Memory Service** | 8002 | Multi-tier storage (Redis/Postgres/Vector/S3) |
| **MCP Gateway** | 8080 | External tool catalog and proxy |
| **Code Executor** | 8004 | Sandboxed skill execution |

---

## 2. Project Overview

### 2.1 Problem Statement

1. No visual interface for monitoring multi-agent workflows
2. No centralized view of system health and performance
3. Approval workflows require manual API calls
4. No visualization of agent decision-making processes
5. No business metrics tracking for orchestrator-managed operations

### 2.2 Solution

A comprehensive **Master Control Panel** web application providing:
- 15+ specialized dashboards
- Real-time WebSocket updates
- Visual workflow builder
- AI-powered conversational interface
- Business intelligence and analytics
- Complete audit and compliance center

---

## 3. System Architecture

### 3.1 Technology Stack

#### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React 18 + TypeScript |
| State Management | Zustand + React Query |
| UI Library | shadcn/ui + Tailwind CSS |
| Charts | Recharts + D3.js |
| Workflow Visualization | React Flow |
| Real-Time | Socket.IO Client |
| Forms | React Hook Form + Zod |
| Tables | TanStack Table |

#### Backend (Control Panel Server)
| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| WebSocket | FastAPI WebSocket + Redis PubSub |
| ORM | SQLAlchemy 2.0 + asyncpg |
| Caching | Redis |
| Authentication | JWT + OAuth2 |

#### Infrastructure
| Component | Technology |
|-----------|------------|
| Database | PostgreSQL 16 |
| Cache/PubSub | Redis 7 |
| Reverse Proxy | Nginx |
| Containerization | Docker + Docker Compose |

---

## 4. Core Dashboards & Features

### 4.1 Dashboard 1: Command Center (Main Hub)
- System Health Bar for all 5 services
- KPI Cards (active workflows, pending approvals, token usage)
- Live Activity Feed via WebSocket
- Quick Actions panel
- Workflow Throughput Chart (24h)
- P&L Summary widget
- Agent Utilization metrics
- Model Costs breakdown

### 4.2 Dashboard 2: Workflow Studio
- Drag-and-Drop workflow builder
- Visual step configuration
- Model selection per step
- Artifact flow visualization
- Live execution overlay
- YAML import/export
- Template library
- Dry run mode

### 4.3 Dashboard 3: Approval Center
- Pending approvals sorted by priority
- Risk assessment display
- Thinking trace from AI
- Provenance chain visualization
- Bulk approve/reject
- Expiration countdown
- Audit trail

### 4.4 Dashboard 4: Business P&L Tracker
- Revenue tracking by workflow type
- Expense breakdown (LLM, infra, tools)
- Margin analysis
- Historical trends
- Cost attribution per workflow
- ROI calculations
- Budget alerts

### 4.5 Dashboard 5: Agent Control Center
- Active agents list
- Agent spawn configuration
- Capability assignment
- Performance metrics
- Utilization charts
- Pause/restart/destroy actions

### 4.6 Dashboard 6: Memory Explorer
- 4-tier memory visualization (Session/Vector/Structured/Cold)
- Semantic search
- Session browser
- Artifact timeline
- Compaction controls
- Storage statistics

### 4.7 Dashboard 7: Model Hub
- Active models status
- Model routing rules
- Usage statistics by provider
- Cost tracking
- Configuration editor
- Budget controls

### 4.8 Dashboard 8: Tool Catalog (MCP)
- Registered servers list
- Tool browser with schemas
- Test tool interface
- Provenance log
- Enable/disable controls

### 4.9 Dashboard 9: Skill Manager
- Available skills list
- Skill testing interface
- Safety flags display
- Execution logs

### 4.10 Dashboard 10: Provenance Viewer
- Artifact lineage graphs
- Record details
- Hash verification
- Compliance reports

### 4.11 Dashboard 11: Talk to King AI
- Conversational interface
- Auto mode detection (brainstorm vs command)
- Inline workflow creation
- Execution confirmation
- Live progress display

### 4.12 Dashboard 12: Summary Hub
- Report templates
- Scheduled reports
- Export options (PDF, Excel, CSV)
- Automated insights

### 4.13 Dashboard 13: Activity Monitor
- Real-time event stream
- Filterable by service/agent/workflow
- Event details expansion
- Pause/resume stream

### 4.14 Dashboard 14: History Archive
- Completed workflows browser
- Search and filtering
- Comparison view
- Replay functionality

### 4.15 Dashboard 15: Settings & Configuration
- System settings
- Security configuration
- LLM provider setup
- Memory settings
- Notification preferences
- User management

---

## 5. API Integration Requirements

### 5.1 Control Panel Backend Endpoints

```
# Authentication
POST /api/auth/login
POST /api/auth/logout
GET /api/auth/me

# Dashboard
GET /api/dashboard/overview
GET /api/dashboard/health

# Business P&L
GET /api/business/pl/summary
GET /api/business/pl/trend
POST /api/business/transactions

# Analytics
GET /api/analytics/workflows/throughput
GET /api/analytics/agents/utilization
GET /api/analytics/models/usage

# Conversational
POST /api/chat/message
GET /api/chat/history

# WebSocket
WS /ws/activity-feed
WS /ws/approvals
WS /ws/workflows/{id}
```

### 5.2 Service Proxy Routes

```
/api/orchestrator/*     -> http://localhost:8000
/api/subagent-manager/* -> http://localhost:8001
/api/memory/*           -> http://localhost:8002
/api/mcp/*              -> http://localhost:8080
/api/code-exec/*        -> http://localhost:8004
```

---

## 6. Data Models

### 6.1 Business P&L Models

```python
class BusinessUnit:
    id: str
    name: str
    description: str

class FinancialTransaction:
    id: str
    business_unit_id: str
    workflow_id: Optional[str]
    transaction_type: Literal["revenue", "expense"]
    category: str
    amount: Decimal
    timestamp: datetime

class PLSummary:
    period: str
    total_revenue: Decimal
    total_expenses: Decimal
    net_profit: Decimal
    margin_percent: float
```

---

## 7. User Interface Specifications

### 7.1 Design System
- Dark mode primary, light mode secondary
- Inter font for UI, JetBrains Mono for code
- 4px spacing system
- Lucide React icons
- Consistent chart colors

### 7.2 Responsive Breakpoints
- Mobile: < 640px
- Tablet: 640-1024px
- Desktop: 1024-1440px
- Wide: > 1440px

---

## 8. Security & Authentication

### 8.1 RBAC Roles
| Role | Permissions |
|------|-------------|
| Admin | Full access |
| Operator | Manage workflows, agents, approvals |
| Analyst | Read-only + P&L access |
| Auditor | Read-only + provenance access |
| Developer | Workflow creation, testing |

---

## 9. Performance Requirements

| Operation | Target |
|-----------|--------|
| Page Load | < 1.5s |
| API Response | < 200ms |
| WebSocket Event | < 100ms |

---

## 10. Deployment

### 10.1 Ports
- Frontend: 3000
- Backend: 8100
- PostgreSQL: 5432
- Redis: 6379

---

## 11. Implementation Phases

### Phase 1: Foundation (Step 2-3)
- Backend setup with FastAPI
- Frontend setup with React/TypeScript
- Authentication system
- Service proxy layer

### Phase 2: Core Dashboards (Step 4)
- Command Center
- Workflow Studio
- Approval Center
- Agent Control

### Phase 3: Advanced Features (Step 5)
- All 15 dashboards complete
- Real-time WebSocket integration
- P&L tracking
- Conversational interface

### Phase 4: Integration (Step 6)
- Docker Compose configuration
- Service integration testing
- End-to-end testing

---

**END OF PRD**
