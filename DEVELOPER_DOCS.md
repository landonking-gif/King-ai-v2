# King AI v2 - Developer Documentation

## 1. System Architecture
King AI v2 is a hybrid "Dual-Brain" autonomous business empire. It is designed to run its heavy AI inference on a dedicated Cloud GPU while maintaining a lightweight dashboard accessible from anywhere.

### Component Overview
1.  **Backend (FastAPI)**:
    *   **Port**: 8000
    *   **Location**: `src/api`
    *   **Role**: Handles all business logic, database interactions, and AI orchestration.
2.  **Frontend (React/Vite)**:
    *   **Port**: 5173
    *   **Location**: `dashboard/`
    *   **Role**: Glassmorphic UI for the user to interact with the CEO and view empire stats.
3.  **Database (PostgreSQL & Redis)**:
    *   **Role**: PG stores persistent business data; Redis handles caching and task queues.
    *   **Deployment**: Runs via Docker Compose.
4.  **AI Brain (Hybrid)**:
    *   **Primary**: **Ollama** running Llama 3.1 8b on AWS `g4dn.xlarge`.
    *   **Fallback**: **Google Gemini 1.5 Flash** (via API) activates automatically if AWS is unreachable.
    *   **Orchestrator**: `src/master_ai/brain.py` manages the routing logic.

### current Capabilities
As of v2.0, the system supports:
1.  **Dual-Brain Intelligence**: Automatically switches between AWS Ollama (Cost-effective) and Gemini (High-intelligence fallback).
2.  **Business Lifecycle Verification**: Can take a business idea from "Discovery" to "Active" using the `LifecycleEngine` state machine.
3.  **Autonomous Chat**: The "CEO" can answer queries about the empire ("How much revenue?", "What is PetPal doing?") by querying the Postgres database.
4.  **Remote Persistance**: All data is stored in the remote PostgreSQL instance, preserving state even if the dashboard is closed.

---

## 2. System Logic & Data Flow
The following sequence diagram illustrates how a user request (e.g., "Start a business") flows through the system.

```mermaid
sequenceDiagram
    participant User
    participant Dash as Dashboard (React)
    participant API as FastAPI
    participant Brain as MasterAI
    participant LLM as Ollama/Gemini
    participant DB as Postgres

    User->>Dash: "Start a dropshipping business"
    Dash->>API: POST /api/chat
    API->>Brain: process_input()
    
    rect rgb(20, 20, 20)
        Note over Brain,LLM: Intent Classification
        Brain->>LLM: "Classify intent of: Start dropshipping..."
        LLM-->>Brain: {"type": "command", "domain": "commerce"}
    end

    Brain->>Brain: _handle_command()
    Brain->>DB: Create Business Unit (Status: Discovery)
    DB-->>Brain: Success (ID: 123)

    rect rgb(20, 20, 20)
        Note over Brain,LLM: Generation
        Brain->>LLM: "Generate welcome message for new business..."
        LLM-->>Brain: "Dropshipping unit initialized..."
    end

    Brain-->>API: Response + Actions Taken
    API-->>Dash: JSON Response
    Dash-->>User: Displays Message & Updates Empire View
```

---

## 3. Deployment Strategy (AWS)
The system is fully deployed on an **AWS EC2 g4dn.xlarge** instance.

### Instance Details
*   **IP Address**: `44.212.96.61`
*   **User**: `ubuntu`
*   **Key File**: `king-ai-studio.pem`
*   **Location**: `/home/ubuntu/king-ai-v2`

### Deployment Workflow
Use the provided `scripts/deploy.ps1` automation script. It performs the following:
1.  Syncs local changes to GitHub.
2.  Connects to AWS via SSH.
3.  Pulls the latest code.
4.  Restarts the Docker containers and Python/Node services.

### Manual Service Management (On Server)
```bash
# Connect
ssh -i "king-ai-studio.pem" ubuntu@44.212.96.61

# Restart App
cd king-ai-v2
./start.sh

# Restart Database
sudo docker-compose restart

# View Logs
tail -f backend.log
tail -f dashboard/frontend.log
```

---

## 3. Environment & Secrets
The `.env` file is critical. **DO NOT COMMIT .env TO GITHUB**.

### Required Variables
```ini
# Database
DATABASE_URL=postgresql+asyncpg://king:password@localhost:5432/kingai
REDIS_URL=redis://localhost:6379

# AI Configuration
OLLAMA_URL=http://localhost:11434  # Internal loopback on AWS
OLLAMA_MODEL=llama3.1:8b
GEMINI_API_KEY=AIzaSy...           # Fallback Key

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 4. Key Code Paths for the Next Developer
*   **AI Logic**: `src/master_ai/brain.py` - The core decision loop.
*   **Response Fallback**: `src/master_ai/brain.py -> _call_llm()` - Where the switch between Ollama and Gemini happens.
*   **Business Logic**: `src/business/lifecycle.py` - State machine for business evolution.
*   **Frontend Data**: `dashboard/src/App.jsx` - Main UI controller.

## 5. Troubleshooting
*   **"Connection to King AI Failed"**:
    *   Check if `uvicorn` is running on AWS (`ps aux | grep uvicorn`).
    *   Check if `OLLAMA_URL` is reachable from the backend (`curl localhost:11434`).
*   **Database Errors**:
    *   Ensure Docker containers are healthy (`sudo docker ps`).
    *   Check migration status (`alembic current`).
