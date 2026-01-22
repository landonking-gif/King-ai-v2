# Ralph - Standalone Autonomous AI Agent

This is a standalone version of Ralph that can be copied to any project.

## Files Required:
- `ralph-once.ps1` - Single iteration script
- `ralph.ps1` - Loop runner
- `prompts/default.txt` - AI prompt template
- `prd.json` - Your product requirements (create this)
- `progress.txt` - Auto-generated progress log

## Quick Setup:

1. Copy these Ralph files to your project root
2. Create a `prd.json` file with your requirements
3. Trust the folder: `copilot` then select option 2
4. Run: `.\ralph-once.ps1 -Prompt prompts\default.txt -AllowProfile safe`

## Model:
Ralph always uses `gpt-5-mini` for consistent, high-quality code generation.

## Independence:
Ralph works with any codebase and doesn't depend on specific frameworks or project structures.