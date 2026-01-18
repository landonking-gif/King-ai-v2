# MCP Server Quick Start Guide

**TLDR:** Just edit YAML files and restart `kautilya`. Everything syncs automatically! ✨

## The New Way (Automatic)

```bash
# 1. Edit your MCP server YAML
vim examples/firecrawl_mcp_server.yaml
# Change: name: Firecrawl MCP Server
# To:     name: Firecrawl Web Search

# 2. Restart kautilya
kautilya

# ✅ Done! Changes are automatically synced
```

## Where to Put YAML Files

Auto-sync scans these directories:

```
your-project/
├── examples/                  ← Example configs (checked into git)
├── .kautilya/mcp_servers/     ← Your custom servers (gitignored)
└── mcp_servers/               ← Project-specific servers
```

**Recommendation:** Put your custom MCP servers in `.kautilya/mcp_servers/`

## Quick Commands

### List All Servers
```bash
# CLI
kautilya mcp list

# Interactive
kautilya
> /mcp list
```

### Enable/Disable Servers
```bash
# CLI
kautilya mcp enable <tool_id>
kautilya mcp disable <tool_id>

# Interactive
kautilya
> /mcp enable <tool_id>
> /mcp disable <tool_id>
```

### See Sync Details
```bash
# Enable verbose mode
export KAUTILYA_VERBOSE_MODE=true

# Start kautilya
kautilya
```

**Output:**
```
Found 3 MCP YAML file(s) to sync...
Updating firecrawl_mcp from firecrawl_mcp_server.yaml
✓ MCP Sync: 0 new, 1 updated, 2 skipped
```

## Creating a New MCP Server

```bash
# 1. Create YAML file
cat > .kautilya/mcp_servers/my_server.yaml <<EOF
tool_id: my_custom_api
name: My Custom API
version: 1.0.0
owner: dev-team
contact: dev@example.com
endpoint: https://api.example.com/v1
auth_flow: api_key
classification:
  - safe
  - external_call
rate_limits:
  max_calls: 100
  window_seconds: 60
metadata:
  api_key_env: MY_API_KEY
tools:
  - name: fetch_data
    description: Fetch data from API
    parameters:
      - name: query
        type: string
        description: Search query
        required: true
    returns: API response data
EOF

# 2. Set API key (if needed)
export MY_API_KEY=your-api-key-here

# 3. Start kautilya
kautilya

# ✅ Server is automatically registered!
```

## Updating an Existing Server

```bash
# 1. Edit YAML file
vim .kautilya/mcp_servers/my_server.yaml
# Change any of: name, version, endpoint

# 2. Restart kautilya
kautilya

# ✅ Server is automatically updated!
```

## Troubleshooting

**Changes not appearing?**
```bash
# Enable verbose mode to see what's happening
export KAUTILYA_VERBOSE_MODE=true
kautilya
```

**Server not found?**
```bash
# Make sure YAML file is in a scanned directory
ls examples/*.yaml
ls .kautilya/mcp_servers/*.yaml
ls mcp_servers/*.yaml
```

**Need help?**
```bash
# Read full documentation
cat artifacts/MCP_AUTO_SYNC.md
```

## That's It!

No more manual imports. Just edit YAML and restart kautilya. 🎉

---

**Full Docs:** `artifacts/MCP_AUTO_SYNC.md`
