# Ralph Setup Guide

## Step 1: Trust the Folder (One-Time Setup)

The GitHub Copilot CLI needs to trust your project folder before it can work autonomously.

```powershell
# Run copilot interactively
copilot
```

You'll see a prompt like this:
```
┌──────────────────────────────────────────────────────────────────┐
│ Confirm folder trust                                              │
│                                                                  │
│ /path/to/your/project                                            │
│                                                                  │
│ Do you trust the files in this folder?                          │
│                                                                  │
│ ❯ 1. Yes                                                         │
│   2. Yes, and remember this folder for future sessions           │
│   3. No (Esc)                                                    │
└──────────────────────────────────────────────────────────────────┘
```

**Select option 2: "Yes, and remember this folder for future sessions"**

Then press **Esc** or type `exit` to close Copilot CLI.

This only needs to be done once per project folder.

## Step 2: Test Ralph with a Single Run

```powershell
# Navigate to your project root (where prd.json is located)
cd /path/to/your/project

# Run one iteration
.\ralph-once.ps1 -Prompt prompts\default.txt -AllowProfile safe
```

This will:
- Read your `prd.json`
- Pick the highest priority incomplete story
- Implement it using Copilot CLI
- Update the PRD and progress.txt
- Commit the changes

## Step 3: Run Ralph in a Loop

Once you've verified it works:

```powershell
# Run 10 iterations
.\scripts\ralph\ralph.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe -Iterations 10
```

Ralph will:
- Implement stories one at a time
- Stop when all stories are complete
- Stop after 10 iterations (whichever comes first)
- Log everything to `progress.txt`

## Monitoring

### Watch progress in real-time
```powershell
# In another terminal
Get-Content progress.txt -Wait
```

### Check what was implemented
```powershell
git log --oneline -10
```

### See remaining stories
```powershell
Get-Content prd.json | ConvertFrom-Json | Where-Object { $_.passes -eq $false } | Format-Table id, title, priority
```

## Troubleshooting

### Copilot keeps asking for folder trust
You didn't select option 2 ("remember this folder"). Run `copilot` again and select option 2.

### "Copilot CLI not found"
```powershell
npm i -g @github/copilot
copilot --version
```

### Scripts won't run (execution policy)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Copilot asks for authentication
```powershell
# Use GitHub CLI
gh auth login

# Or authenticate through Copilot
copilot
# Follow the login prompts
```

## Next Steps

Once Ralph is running successfully:

1. **Create custom prompts** for your specific needs
2. **Add skills** for reusable instruction sets
3. **Adjust permission profiles** as needed
4. **Monitor and iterate** on your PRD

See [README-NEW.md](README-NEW.md) for full documentation.
