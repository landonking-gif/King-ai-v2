# GitHub Copilot Memory Plugin

A VS Code extension that implements long-term memory for GitHub Copilot, learning from your coding sessions to continuously improve AI assistance.

## Features

- **📝 Session Diary**: Generate detailed diary entries from coding sessions
- **🧠 Pattern Reflection**: Analyze multiple sessions to identify patterns and preferences
- **💾 Long-term Memory**: Store learnings in `COPILOT.md` for future reference
- **🔄 Continuous Learning**: Automatically improve Copilot's understanding over time

## Installation

### Prerequisites
- VS Code 1.89+
- Node.js 16+
- npm

### Setup Steps

1. **Open Extension Directory**:
   ```bash
   code copilot-memory-plugin/
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Compile Extension**:
   ```bash
   npm run compile
   ```

4. **Test Extension**:
   - Press `F5` to launch extension development host
   - Open GitHub Copilot Chat in the new window

## Usage

### Commands

#### `/diary`
Generates a diary entry for the current session.

**Captures**:
- Open workspace files
- Recent Git changes
- Session context and activities

#### `/reflect`
Analyzes unprocessed diary entries and updates memory.

**Does**:
- Identifies recurring patterns
- Extracts user preferences
- Updates `COPILOT.md` with learnings

### Memory File Structure

```
.copilot/memory/
├── diary/                    # Individual session diaries
│   └── YYYY-MM-DD-session-ID.md
├── processed.log             # Tracks analyzed entries
└── COPILOT.md               # Long-term memory file
```

## Example Diary Entry

```markdown
# Diary Entry: 2026-01-19 - Session abc123

## Task Summary
Implemented user authentication system

## Work Done
- Modified 3 files
- Added JWT token handling

## Design Decisions
- Used bcrypt for password hashing
- Implemented refresh tokens

## User Preferences
- Prefers functional components
- Uses TypeScript strict mode
```

## Configuration

### Extension Settings
Add to VS Code `settings.json`:
```json
{
  "copilot-memory": {
    "autoDiary": true,
    "reflectionInterval": 10,
    "memoryPath": ".copilot/memory"
  }
}
```

### API Proposals
The extension uses VS Code's proposed chat API:
```json
{
  "enabledApiProposals": ["chatParticipant"]
}
```

## Development

### Building
```bash
npm run compile
npm run watch  # For development
```

### Testing
```bash
npm run test
```

### Debugging
- Press `F5` to launch debug session
- Check Debug Console for extension logs
- Use `/copilot-memory.test` command to verify loading

## Architecture

### Components
- **Chat Participants**: Handle `/diary` and `/reflect` commands
- **Diary Generator**: Creates structured session logs
- **Reflection Engine**: Analyzes patterns across entries
- **Memory Storage**: Manages long-term learnings

### Data Flow
1. User interacts with Copilot
2. `/diary` captures session context
3. `/reflect` analyzes patterns
4. `COPILOT.md` updated with insights
5. Future sessions use accumulated knowledge

## Troubleshooting

### Extension Not Loading
- Verify VS Code version (1.89+ required)
- Check `npm run compile` succeeded
- Review Debug Console for errors

### Commands Not Working
- Ensure Copilot Chat is active
- Try reloading window: `Ctrl+Shift+P` → "Reload Window"
- Check extension is enabled

### Memory Not Saving
- Verify write permissions on workspace
- Check `.copilot/memory/` directory exists
- Ensure Git repository is initialized

## Contributing

### Code Structure
```
src/
├── extension.ts              # Main extension logic
├── diary.ts                  # Diary generation
├── reflection.ts             # Pattern analysis
└── memory.ts                 # Storage management
```

### Adding Features
1. Extend chat participants in `extension.ts`
2. Add new commands to `package.json`
3. Update memory storage as needed

## License

MIT License - see LICENSE file for details.

## Related

- **King AI Integration**: See `MEMORY_PLUGIN_README.md` for full system documentation
- **Claude Diary**: Original inspiration by rlancemartin
- **Generative Agents**: Memory architecture reference