# MoltBot Configuration - No API Keys Required

## Overview
MoltBot is now configured to use the **local DeepSeek R1 7B model** running on your AWS server via Ollama. **No external API keys are needed.**

## How It Works

```
User Message (Telegram/Discord/etc)
         ↓
   MoltBot Gateway (Port 18789)
         ↓
   Ollama (Port 11434)
         ↓
   DeepSeek R1 7B Model (Local)
         ↓
   Response back to user
```

## Key Configuration Changes

### Primary Model
- **Before:** `kingai/deepseek-r1` (through orchestrator)
- **After:** `ollama/deepseek-r1:7b` (direct to local Ollama)

### API Keys
- **Before:** Required external API keys
- **After:** Uses placeholder `"not-required-local-only"` - no real keys needed

## Benefits

✅ **100% Local** - All processing happens on your AWS server
✅ **No API Costs** - No external API calls or charges
✅ **Private** - Your conversations never leave your server
✅ **Fast** - Direct connection to local model
✅ **No Rate Limits** - Use as much as you want

## Configuration File

Located at: `~/.moltbot/moltbot.json` on the server

```json
{
  agents: {
    defaults: {
      model: { 
        primary: "ollama/deepseek-r1:7b"  // 👈 Uses local model
      }
    }
  },
  
  models: {
    providers: {
      ollama: {
        baseUrl: "http://localhost:11434/v1",
        apiKey: "not-required-local-only",  // 👈 No real key needed
        models: [{
          id: "deepseek-r1:7b",
          name: "DeepSeek-R1-7B-Local"
        }]
      }
    }
  }
}
```

## Testing

Once deployed, test MoltBot:

```bash
# SSH into server
ssh -i <key> ubuntu@52.90.206.76

# Check MoltBot status
curl http://localhost:18789/

# Check Ollama status
curl http://localhost:11434/api/tags

# View MoltBot logs
tail -f /tmp/moltbot.log
```

## Channel Setup

To connect messaging platforms (still requires platform API keys, not LLM keys):

### Telegram
```bash
# Edit config
nano ~/.moltbot/moltbot.json

# Add your Telegram bot token
{
  channels: {
    telegram: {
      token: "YOUR_TELEGRAM_BOT_TOKEN"
    }
  }
}

# Restart MoltBot
pkill -f moltbot
cd /home/ubuntu/king-ai-v3/moltbot
pnpm moltbot gateway --port 18789 &
```

### Discord
Similar process - add Discord bot token to config

### WhatsApp
```bash
cd /home/ubuntu/king-ai-v3/moltbot
pnpm moltbot whatsapp pair
# Scan QR code
```

## Important Notes

1. **Messaging Platform Keys**: You still need API tokens for Telegram, Discord, etc. to connect to those platforms. This is different from LLM API keys.

2. **LLM API Keys**: NOT NEEDED - MoltBot uses the local DeepSeek R1 7B model

3. **Model Performance**: DeepSeek R1 7B provides excellent reasoning capabilities while running entirely on your server

4. **Fallback**: If you ever want to use external LLM APIs (OpenAI, Anthropic, etc.), you can add them to the config - but the default is local-only

## Deployment

Updated configuration deployed to: `52.90.206.76`

Use single deployment script:
```powershell
.\deploy.ps1 -IpAddress 52.90.206.76
```

## Support

- **MoltBot Logs**: `/tmp/moltbot.log`
- **Ollama Logs**: `/tmp/ollama.log`  
- **Integration Guide**: [MOLTBOT_INTEGRATION.md](MOLTBOT_INTEGRATION.md)
- **Full Deployment Status**: [DEPLOYMENT_STATUS.md](DEPLOYMENT_STATUS.md)
