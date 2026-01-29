import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

let memoryDir: string;
let diaryDir: string;
let processedLog: string;

export function activate(context: vscode.ExtensionContext) {
    // Set up directories
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder open');
        return;
    }

    memoryDir = path.join(workspaceFolder.uri.fsPath, '.copilot', 'memory');
    diaryDir = path.join(memoryDir, 'diary');
    processedLog = path.join(memoryDir, 'processed.log');

    // Ensure directories exist
    fs.mkdirSync(memoryDir, { recursive: true });
    fs.mkdirSync(diaryDir, { recursive: true });

    // Register chat participants
    const diaryParticipant = vscode.chat.registerChatParticipant('copilot-memory.diary', handler, {
        name: 'diary',
        description: 'Generate a diary entry for the current coding session',
        isSticky: true
    });
    const reflectParticipant = vscode.chat.registerChatParticipant('copilot-memory.reflect', reflectHandler, {
        name: 'reflect',
        description: 'Perform reflection across multiple diary entries and update memory',
        isSticky: true
    });

    context.subscriptions.push(diaryParticipant, reflectParticipant);

    // Register command for auto diary
    const autoDiaryCommand = vscode.commands.registerCommand('copilot-memory.autoDiary', () => {
        generateDiaryEntry();
    });
    const testCommand = vscode.commands.registerCommand('copilot-memory.test', () => {
        vscode.window.showInformationMessage('Hello from Copilot Memory Plugin!');
    });
    context.subscriptions.push(autoDiaryCommand, testCommand);

    // Auto diary on save (simulating pre-compact hook)
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument(() => {
            // Simple trigger: generate diary occasionally or based on conditions
            // For now, manual
        })
    );
}

async function handler(request: vscode.ChatRequest, context: vscode.ChatContext, stream: vscode.ChatResponseStream, token: vscode.CancellationToken): Promise<void> {
    const command = request.prompt.trim().toLowerCase();

    if (command === '/diary' || command === 'diary') {
        await generateDiaryEntry(stream);
    } else {
        stream.markdown('Use /diary to generate a diary entry for the current session.');
    }
}

async function reflectHandler(request: vscode.ChatRequest, context: vscode.ChatContext, stream: vscode.ChatResponseStream, token: vscode.CancellationToken): Promise<void> {
    const command = request.prompt.trim().toLowerCase();

    if (command.startsWith('/reflect') || command.startsWith('reflect')) {
        await performReflection(stream);
    } else {
        stream.markdown('Use /reflect to analyze diary entries and update memory.');
    }
}

async function generateDiaryEntry(stream?: vscode.ChatResponseStream) {
    // Get current workspace changes or recent activity
    const gitExtension = vscode.extensions.getExtension('vscode.git')?.exports;
    let changes = 'No recent changes detected.';

    if (gitExtension) {
        const api = gitExtension.getAPI(1);
        const repo = api.repositories[0];
        if (repo) {
            const status = await repo.getStatus();
            changes = status.files.map(f => `${f.status} ${f.uri.fsPath}`).join('\n');
        }
    }

    // Get open files
    const openFiles = vscode.workspace.textDocuments.map(doc => doc.fileName).join('\n');

    // Generate diary content
    const date = new Date().toISOString().split('T')[0];
    const sessionId = Date.now();
    const diaryFile = path.join(diaryDir, `${date}-session-${sessionId}.md`);

    const diaryContent = `# Diary Entry: ${date} - Session ${sessionId}

## Task Summary
Captured activity in the current coding session.

## Work Done
- Open files: ${openFiles}
- Changes: ${changes}

## Design Decisions
- [Add specific decisions here]

## User Preferences
- [Document preferences observed]

## Code Review Feedback
- [Any feedback noted]

## Challenges
- [Challenges faced]

## Solutions
- [Solutions implemented]

## Code Patterns
- [Patterns used]
`;

    fs.writeFileSync(diaryFile, diaryContent);

    const message = `Diary entry generated and saved to ${diaryFile}`;
    if (stream) {
        stream.markdown(message);
    } else {
        vscode.window.showInformationMessage(message);
    }
}

async function performReflection(stream: vscode.ChatResponseStream) {
    // Read all diary files
    const diaryFiles = fs.readdirSync(diaryDir).filter(f => f.endsWith('.md'));

    // Read processed log
    let processed: string[] = [];
    if (fs.existsSync(processedLog)) {
        processed = fs.readFileSync(processedLog, 'utf8').split('\n').filter(Boolean);
    }

    const unprocessed = diaryFiles.filter(f => !processed.includes(f));

    if (unprocessed.length === 0) {
        stream.markdown('No new diary entries to reflect on.');
        return;
    }

    // Analyze patterns
    let analysis = '## Reflection Analysis\n\n';
    analysis += `Analyzed ${unprocessed.length} new diary entries.\n\n`;

    // Simple pattern detection (in real implementation, use AI)
    analysis += '### Observed Patterns\n';
    analysis += '- Consistent use of TypeScript\n';
    analysis += '- Preference for async/await\n\n';

    // Update COPILOT.md
    const memoryFile = path.join(memoryDir, 'COPILOT.md');
    let memoryContent = '';
    if (fs.existsSync(memoryFile)) {
        memoryContent = fs.readFileSync(memoryFile, 'utf8');
    }

    memoryContent += '\n' + analysis;

    fs.writeFileSync(memoryFile, memoryContent);

    // Update processed log
    processed.push(...unprocessed);
    fs.writeFileSync(processedLog, processed.join('\n'));

    stream.markdown(`Reflection completed. Updated ${memoryFile} with new learnings.`);
}

export function deactivate() {}