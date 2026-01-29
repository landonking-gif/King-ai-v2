# MoltBot Integration Guide

## Overview

MoltBot is now integrated with King AI v3, providing multi-channel access to the AI system through various messaging platforms. This allows you to interact with King AI via WhatsApp, Telegram, Discord, Slack, Signal, Google Chat, Matrix, and more.

**🔥 NO API KEYS REQUIRED** - MoltBot uses the local DeepSeek R1 7B model running on the AWS server via Ollama. Everything runs locally without external API dependencies.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Messaging Platforms                  │
│  WhatsApp │ Telegram │ Discord │ Slack │ Signal     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  MoltBot Gateway      │
          │  Port: 18789          │
          │  ws://localhost:18789 │
          └──────────┬────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Ollama (Local LLM)    │
          │ Port: 11434           │
          │ deepseek-r1:7b        │
          │ NO API KEYS NEEDED    │
          └───────────────────────┘
```

## Installation Status

✅ **Completed:**
- MoltBot cloned to `/home/ubuntu/king-ai-v3/moltbot`
- Node.js upgraded to v22.22.0 (required)
- Dependencies installed via pnpm
- UI built and ready
- Configuration file created at `~/.moltbot/moltbot.json`
- **Configured to use local DeepSeek R1 7B (no API keys required)**
- Gateway running on port 18789

## Current Configuration

### MoltBot Config (`~/.moltbot/moltbot.json`)

**Default Model:** `ollama/deepseek-r1:7b` (Local, No API Keys)

```json
{
  gateway: {
    port: 18789,
    mode: "local",
    auth: {
      mode: "token",
      token: "kingai-moltbot-token-2026"
    }
  },
  
  agents: {
    defaults: {
      workspace: "~/king-ai-workspace",
      model: { 
        primary: "ollama/deepseek-r1:7b"  // Uses local Ollama - NO API KEY
      },
      models: {
        "ollama/deepseek-r1:7b": { 
          alias: "DeepSeek-Local"
        },
        "kingai/deepseek-r1": { 
          alias: "DeepSeek-via-Orchestrator"
        }
      }
    }
  },
  
  models: {
    mode: "merge",
    providers: {
      ollama: {
        baseUrl: "http://localhost:11434/v1",
        apiKey: "not-required-local-only",  // No real API key needed
        api: "openai-completions",
        models: [{
          id: "deepseek-r1:7b",
          name: "DeepSeek-R1-7B-Local",
          input: ["text"],
          contextWindow: 32000,
          maxTokens: 4096
        }]
      },
      kingai: {
        baseUrl: "http://localhost:8000/v1",
        apiKey: "not-required-local-only",  // No real API key needed
        api: "openai-completions",
        models: [{
          id: "deepseek-r1",
          name: "DeepSeek-R1-via-Orchestrator",
          input: ["text"],
          contextWindow: 32000,
          maxTokens: 4096
        }]
      }
    }
  }
}
```

**Note:** The `apiKey` fields are placeholders. Both providers use local services - no external API keys are required.

## Service Management

### Start All Services
```bash
# Unified startup script (starts everything)
bash /home/ubuntu/king-ai-v3/start_all_services.sh
```

### Start MoltBot Individually
```bash
cd /home/ubuntu/king-ai-v3/moltbot
pnpm moltbot gateway --port 18789
```

### Check Service Status
```bash
# Check if MoltBot is running
curl -I http://localhost:18789

# Check Ollama
curl http://localhost:11434/api/version

# Check King AI Orchestrator
curl http://localhost:8000/

# Check OpenAI-compatible endpoint
curl http://localhost:8000/v1/models
```

### View Logs
```bash
# MoltBot logs
tail -f /tmp/moltbot.log

# Orchestrator logs
tail -f /tmp/orchestrator.log

# Ollama logs
tail -f /tmp/ollama.log
```

## Channel Configuration

### Prerequisites
Each messaging platform requires specific API keys or setup:

#### Telegram (Easiest to Set Up)
1. Create a bot with [@BotFather](https://t.me/botfather)
2. Get the bot token
3. Add to config or environment:
   ```bash
   export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
   ```

#### Discord
1. Create application at [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a bot and get the token
3. Add to config:
   ```bash
   export DISCORD_BOT_TOKEN="your-discord-bot-token"
   ```

#### Slack
1. Create app at [Slack API](https://api.slack.com/apps)
2. Enable Socket Mode
3. Get Bot Token and App Token
4. Add to config:
   ```bash
   export SLACK_BOT_TOKEN="xoxb-your-token"
   export SLACK_APP_TOKEN="xapp-your-token"
   ```

#### WhatsApp
1. Run QR pairing:
   ```bash
   cd /home/ubuntu/king-ai-v3/moltbot
   pnpm moltbot whatsapp pair
   ```
2. Scan QR code with WhatsApp mobile app

#### Signal (Requires signal-cli)
1. Install signal-cli
2. Register phone number
3. Add to config

### Enable Channels

Update `~/.moltbot/moltbot.json` to enable channels. For example, to enable Telegram:

```json
{
  channels: {
    telegram: {
      token: "${TELEGRAM_BOT_TOKEN}",
      allowFrom: ["*"],
      groups: {
        "*": { requireMention: true }
      }
    }
  }
}
```

Or set environment variables in `~/.moltbot/.env`:
```bash
TELEGRAM_BOT_TOKEN=your-token-here
DISCORD_BOT_TOKEN=your-token-here
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_APP_TOKEN=xapp-your-token
```

## Access Points

### MoltBot Control UI
- **URL**: http://100.24.50.240:18789
- **Features**:
  - View connected channels
  - Manage sessions
  - Configure models
  - Monitor agent activity

### King AI Dashboard
- **URL**: http://100.24.50.240:8000
- **Features**:
  - Traditional web chat interface
  - Workflow management
  - Business monitoring

## API Endpoints

### King AI Orchestrator
```bash
# Traditional chat endpoint
POST http://localhost:8000/api/chat
Content-Type: application/json
{
  "message": "Hello, what can you help me with?",
  "user_id": "user123"
}

# OpenAI-compatible endpoint (used by MoltBot)
POST http://localhost:8000/v1/chat/completions
Content-Type: application/json
{
  "model": "deepseek-r1",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}

# List available models
GET http://localhost:8000/v1/models
```

## Testing the Integration

### 1. Test OpenAI-Compatible Endpoint
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "Hello, introduce yourself"}]
  }'
```

### 2. Test MoltBot Gateway
```bash
# Check health
curl http://localhost:18789

# Access control UI
open http://localhost:18789
# (or visit http://100.24.50.240:18789 from browser)
```

### 3. Test End-to-End via Telegram
```bash
# After configuring Telegram bot:
# 1. Open Telegram
# 2. Search for your bot
# 3. Send: /start
# 4. Send: "Hello, what can you help me with?"
# Bot should respond using King AI's DeepSeek model
```

## Troubleshooting

### MoltBot Won't Start
```bash
# Check Node.js version (must be 22+)
node --version

# Rebuild if needed
cd /home/ubuntu/king-ai-v3/moltbot
pnpm install
pnpm build

# Check for port conflicts
netstat -tlnp | grep 18789
```

### "Model not found" Error
```bash
# Verify model configuration
cat ~/.moltbot/moltbot.json | grep -A 10 "models"

# Test King AI endpoint directly
curl http://localhost:8000/v1/models
```

### Channels Not Connecting
```bash
# Run MoltBot doctor
cd /home/ubuntu/king-ai-v3/moltbot
pnpm moltbot doctor

# Fix configuration
pnpm moltbot doctor --fix

# Check logs for specific channel
tail -f /tmp/moltbot.log | grep -i telegram
```

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Check DeepSeek model is pulled
ollama list | grep deepseek-r1

# Pull model if missing
ollama pull deepseek-r1:7b
```

## Advanced Configuration

### Adding More Models
Edit `~/.moltbot/moltbot.json` to add fallback models:

```json
{
  agents: {
    defaults: {
      model: {
        primary: "kingai/deepseek-r1",
        fallbacks: ["ollama/deepseek-r1:7b", "ollama/llama3.2:3b"]
      }
    }
  }
}
```

### Multi-Agent Routing
Configure different agents for different channels:

```json
{
  agents: {
    list: [
      {
        id: "main",
        workspace: "~/king-ai-workspace",
        model: { primary: "kingai/deepseek-r1" }
      },
      {
        id: "telegram-bot",
        workspace: "~/telegram-workspace",
        model: { primary: "ollama/deepseek-r1:7b" }
      }
    ]
  }
}
```

## Security Considerations

### Gateway Authentication
MoltBot gateway uses token authentication. Update the token in config:

```json
{
  gateway: {
    auth: {
      mode: "token",
      token: "your-secure-token-here"
    }
  }
}
```

### DM Pairing
By default, MoltBot requires pairing for direct messages from unknown users:

```bash
# Approve a pairing request
cd /home/ubuntu/king-ai-v3/moltbot
pnpm moltbot pairing approve telegram <code>
```

### Channel Allowlists
Restrict who can use the bot:

```json
{
  channels: {
    telegram: {
      allowFrom: ["+1234567890", "username"]
    },
    discord: {
      guilds: {
        "guild-id": { requireMention: true }
      }
    }
  }
}
```

## Performance Optimization

### Model Selection
- **DeepSeek R1 7B**: Fast, 4.7GB, good for most tasks
- **Llama 3.2 3B**: Faster, 2GB, lightweight responses
- Consider pulling both for automatic fallback

### Memory Management
```bash
# Monitor memory usage
free -h

# If low on memory, use smaller model
ollama pull llama3.2:3b
```

### Concurrent Connections
MoltBot handles multiple channels simultaneously. Monitor with:

```bash
# View active sessions
curl http://localhost:18789/api/sessions

# Check WebSocket connections
netstat -an | grep 18789
```

## Support and Resources

### MoltBot Documentation
- Main docs: https://docs.molt.bot
- Channels: https://docs.molt.bot/channels
- Models: https://docs.molt.bot/concepts/models

### King AI Resources
- Dashboard: http://100.24.50.240:8000
- API Docs: http://100.24.50.240:8000/docs
- GitHub: [Your repo URL]

### Logs and Debugging
```bash
# All service logs
tail -f /tmp/*.log

# MoltBot verbose mode
cd /home/ubuntu/king-ai-v3/moltbot
pnpm moltbot gateway --port 18789 --verbose

# King AI orchestrator logs
tail -f /tmp/orchestrator.log
```

## Next Steps

1. **Configure Your First Channel**: Start with Telegram (easiest)
2. **Test Multi-Channel**: Set up Discord and Slack
3. **Customize Responses**: Adjust system prompts in orchestrator config
4. **Monitor Usage**: Check logs and MoltBot control UI
5. **Scale Up**: Add more channels and agents as needed

---

**Status**: ✅ Operational
**Version**: MoltBot 2026.1.27-beta.1 + King AI v3
**Last Updated**: January 29, 2026
