# Kautilya Helper Scripts

Utility scripts to simplify common kautilya operations.

## Available Scripts

### `update_mcp_server.sh`

**Purpose:** Update an MCP server registration from a YAML configuration file.

**Usage:**
```bash
./scripts/update_mcp_server.sh <yaml_file>
```

**Example:**
```bash
./scripts/update_mcp_server.sh examples/firecrawl_mcp_server.yaml
```

**What it does:**
1. Extracts tool_id from the YAML file
2. Shows current registration (if exists)
3. Asks for confirmation
4. Unregisters the old version
5. Imports the new version
6. Verifies the update

**When to use:**
- After editing an MCP server YAML file
- When you want to rename a server
- When updating server configuration (endpoint, tools, rate limits)
- When fixing registration issues

**Benefits:**
- ✅ Single command instead of 3 separate commands
- ✅ Shows confirmation before proceeding
- ✅ Verifies update completed successfully
- ✅ Handles errors gracefully

## Quick Workflow

### Update an MCP Server

```bash
# 1. Edit the YAML configuration
vim examples/my_server.yaml

# 2. Run the update script
./scripts/update_mcp_server.sh examples/my_server.yaml

# The script will:
# - Show current vs new registration
# - Ask for confirmation
# - Perform the update
# - Verify success
```

### Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MCP Server Update Workflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tool ID: firecrawl_mcp
Config File: examples/firecrawl_mcp_server.yaml

Current registration:
│ firecrawl_mcp   │ Firecrawl MCP Server │ 1.0.0   │ ...

Proceed with update? (y/N) y

Step 1: Unregistering firecrawl_mcp...
✓ Successfully unregistered tool 'firecrawl_mcp'

Step 2: Importing from examples/firecrawl_mcp_server.yaml...
✓ Server registered successfully!

Step 3: Verifying update...
│ firecrawl_mcp   │ Firecrawl Web Search │ 1.0.0   │ ...

✓ Update complete!
```

## Future Scripts (Planned)

- `sync_mcp_servers.sh` - Sync all servers from a directory
- `backup_mcp_registry.sh` - Export all server registrations
- `validate_mcp_yaml.sh` - Validate YAML before importing
- `list_mcp_changes.sh` - Show diff between YAML and registry

## Notes

- All scripts assume `kautilya` is in your PATH
- Scripts are designed to be safe - they ask for confirmation
- Check `artifacts/MCP_UPDATE_WORKFLOW.md` for detailed documentation
