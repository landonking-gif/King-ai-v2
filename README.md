# King AI v2 - Autonomous Business Empire

![King AI V2](https://img.shields.io/badge/Status-Operational-green) ![AI-Brain](https://img.shields.io/badge/AI_Model-Dual_Ollama_Gemini-blue) ![AWS](https://img.shields.io/badge/Deployed-AWS_Pro-orange)

King AI v2 is a sophisticated autonomous agent system designed to plan, launch, and manage digital businesses. It features a "Dual-Brain" architecture that runs powerful LLMs on a dedicated AWS GPU while maintaining a responsive, glassmorphic React dashboard for the user.

## 🚀 Quick Start (Master Controller)
We use a unified "Master Controller" script to handle deployments, server connections, and updates.

**Prerequisites:**
- Node.js installed locally.

**Usage:**
```bash
.\control.bat
```
(Or simply double-click the file in Explorer)

This interactive control center allows you to:
1.  **Full Deploy**: Sync secrets, code, and restart services.
2.  **Quick Sync**: Push code changes without restarting everything.
3.  **Monitor**: Stream live logs from the AWS empire.

## 📚 Documentation
*   [User Guide](USER_GUIDE.md): For operators (How to use the CEO Chat, Empire Overview).
*   [Developer Docs](DEVELOPER_DOCS.md): For engineers (Architecture, AWS Deployment, Secrets).

## 🧠 Features
*   **Dual-Brain AI**: Seamlessly switches between local/AWS Ollama and Google Gemini logic.
*   **Production Deployment**: Fully containerized Database (Postgres/Redis) on AWS EC2.
*   **Autonomous Loop**: Self-optimizing business logic that evolves overnight.
*   **Premium UI**: Glassmorphic dashboard built with React + Vite.
