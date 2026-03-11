# King AI v3 Master Control Panel

A comprehensive web-based dashboard for monitoring and controlling the King AI v3 Agentic Framework.

## Features

### 🖥️ **Command Center**
- Real-time system status overview
- Active agents and workflows monitoring
- System health indicators
- Agent status table with controls

### 🔄 **Workflow Studio**
- Visual workflow builder with drag-and-drop interface
- Support for LLM calls, tool execution, data processing, and conditional logic
- Workflow templates and management
- Real-time workflow execution monitoring

### ✅ **Approval Center**
- Centralized approval request management
- Priority-based approval workflows
- Risk assessment and categorization
- Bulk approval/rejection operations
- Audit trail and approval history

### 📊 **Analytics Dashboard**
- Performance metrics and KPIs
- Interactive charts and graphs
- Agent performance analytics
- Historical data analysis
- Custom reporting capabilities

### ⚙️ **Settings Panel**
- System configuration management
- AI model settings (Ollama, Claude, Gemini)
- Security and authentication settings
- API key management
- User preferences

### 🔴 **Real-time Updates**
- WebSocket-based live data streaming
- Real-time notifications
- Live activity feeds
- Instant status updates

## Tech Stack

### Backend
- **FastAPI** - High-performance async web framework
- **PostgreSQL** - Primary database
- **Redis** - Caching and session management
- **WebSockets** - Real-time communication
- **Pydantic** - Data validation

### Frontend
- **React 18** - Modern UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Modern component library
- **React Router** - Client-side routing
- **Zustand** - State management
- **TanStack Query** - Data fetching and caching
- **Socket.IO** - Real-time communication

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL
- Redis
- Ollama (optional, for local AI models)

### Development Setup

1. **Clone and navigate to the control panel directory:**
   ```bash
   cd king-ai-v3/agentic-framework-main/control-panel
   ```

2. **Backend Setup:**
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt

   # Set up environment variables
   cp .env.example .env
   # Edit .env with your configuration

   # Run the backend
   python main.py
   ```

3. **Frontend Setup:**
   ```bash
   cd frontend

   # Install Node dependencies
   npm install

   # Start development server
   npm run dev
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8100
   - API Documentation: http://localhost:8100/docs

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8100

## API Endpoints

### Dashboard
- `GET /api/dashboard/status` - System status summary
- `GET /api/dashboard/agents` - Agent list
- `GET /api/dashboard/overview` - Main KPIs
- `GET /api/dashboard/health` - Service health

### Workflows
- `GET /api/workflows` - List workflows
- `GET /api/workflows/{id}` - Get workflow details
- `POST /api/workflows` - Create workflow
- `PUT /api/workflows/{id}` - Update workflow

### Approvals
- `GET /api/approvals` - List approval requests
- `POST /api/approvals/{id}/approve` - Approve request
- `POST /api/approvals/{id}/reject` - Reject request

### Analytics
- `GET /api/analytics/metrics` - Performance metrics
- `GET /api/analytics/chart-data` - Chart data

### Settings
- `GET /api/settings` - Get system settings
- `PUT /api/settings` - Update settings

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Current user info

### WebSocket Endpoints
- `/ws/activity-feed` - Real-time activity updates
- `/ws/approvals` - Approval status updates
- `/ws/workflows/{id}` - Workflow-specific updates

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/control_panel
REDIS_URL=redis://localhost:6379

# AI Models
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key

# Security
SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_secret
```

## Development

### Project Structure

```
control-panel/
├── main.py                 # FastAPI backend
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── docker-compose.yml      # Docker orchestration
├── Dockerfile.backend      # Backend container
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/         # Page components
│   │   ├── lib/           # Utilities
│   │   └── App.tsx        # Main app component
│   ├── package.json       # Node dependencies
│   ├── vite.config.ts     # Vite configuration
│   └── Dockerfile.frontend # Frontend container
└── scripts/
    └── setup_db.sql       # Database initialization
```

### Running Tests

```bash
# Backend tests
pytest

# Frontend tests
cd frontend
npm test
```

### Building for Production

```bash
# Build frontend
cd frontend
npm run build

# Build backend container
docker build -f Dockerfile.backend -t king-ai-control-panel-backend .

# Build frontend container
docker build -f frontend/Dockerfile.frontend -t king-ai-control-panel-frontend .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the King AI v3 Agentic Framework. See the main project license for details.

## Support

For support and questions:
- Check the API documentation at `/docs`
- Review the logs in the `logs/` directory
- Open an issue in the main repository