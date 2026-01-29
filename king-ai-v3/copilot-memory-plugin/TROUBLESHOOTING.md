# Troubleshooting Guide

## Common Issues and Solutions

### Extension Won't Load

**Symptoms:**
- Commands don't appear in Copilot Chat
- Extension shows error in VS Code

**Solutions:**
1. Check VS Code version (must be 1.89+)
2. Run `npm run compile` in extension directory
3. Check Debug Console for errors
4. Reload VS Code window (`Ctrl+Shift+P` → "Reload Window")

### Commands Not Working

**Symptoms:**
- `/diary` and `/reflect` don't respond

**Solutions:**
1. Ensure Copilot Chat is open (`Ctrl+Alt+I`)
2. Try typing just `diary` or `reflect` (without slash)
3. Check if extension is activated in status bar
4. Restart VS Code

### Memory Files Not Created

**Symptoms:**
- No `.copilot/memory/` directory
- Diary entries not saved

**Solutions:**
1. Check workspace permissions
2. Ensure Git repository exists
3. Try creating directory manually: `mkdir -p .copilot/memory/diary`
4. Check file system write access

### Reflection Finds No Patterns

**Symptoms:**
- `/reflect` shows "No new diary entries"

**Solutions:**
1. Run `/diary` first to create entries
2. Check `.copilot/memory/diary/` for files
3. Ensure entries aren't already processed
4. Check `processed.log` file

### Performance Issues

**Symptoms:**
- Extension slows down VS Code
- Memory usage is high

**Solutions:**
1. Limit diary entries (delete old ones)
2. Run reflection less frequently
3. Clear processed entries: `rm .copilot/memory/processed.log`
4. Restart VS Code

## Debug Steps

### Check Extension Status
1. Open Command Palette (`Ctrl+Shift+P`)
2. Type "Show Running Extensions"
3. Look for "GitHub Copilot Memory Plugin"

### View Debug Logs
1. Open Debug Console (`Ctrl+Shift+Y`)
2. Look for extension-related messages
3. Check for TypeScript compilation errors

### Manual Testing
1. Open VS Code Developer Tools (`Ctrl+Shift+I`)
2. Go to Console tab
3. Look for extension activation messages

## Getting Help

### Log Files to Check
- `.copilot/memory/diary/*.md` - Diary entries
- `.copilot/memory/processed.log` - Processed entries
- VS Code Debug Console - Extension logs

### Test Commands
```bash
# Check if extension loaded
# In VS Code: Ctrl+Shift+P → "Copilot Memory: Test"

# Manual diary creation
# In Copilot Chat: /diary

# Check file creation
ls -la .copilot/memory/
```

### Reset Extension
1. Close VS Code
2. Delete `.copilot/memory/` directory
3. Restart VS Code
4. Re-run `/diary` to test

## Advanced Troubleshooting

### API Compatibility
If using VS Code < 1.89, the chat API won't work. The extension requires:
- VS Code 1.89+
- GitHub Copilot Chat enabled
- Proposed API enabled in extension

### Network Issues
If reflection fails due to network:
- Check internet connection
- Verify Copilot service is available
- Try again later

### File System Issues
On Windows, ensure:
- Long path support enabled
- Proper NTFS permissions
- No antivirus blocking file creation

## Still Having Issues?

1. Check the main `MEMORY_PLUGIN_README.md` for detailed docs
2. Review VS Code extension logs
3. Try the King AI integration instead (works independently)
4. Report issues with full error logs and VS Code version