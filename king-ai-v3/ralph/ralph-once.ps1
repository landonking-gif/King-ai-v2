<#
.SYNOPSIS
Ralph - Single run with GitHub Copilot CLI

.DESCRIPTION
Runs GitHub Copilot CLI once to implement features from a PRD.
Based on soderlind/ralph implementation.

.PARAMETER Prompt
Path to prompt file (required)

.PARAMETER AllowProfile
Permission profile: locked, safe, dev

.PARAMETER Skill
Comma-separated list of skills to prepend

.EXAMPLE
.\ralph-once.ps1 -Prompt prompts\default.txt -AllowProfile safe
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Prompt,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet('locked', 'safe', 'dev')]
    [string]$AllowProfile = 'safe',
    
    [Parameter(Mandatory=$false)]
    [string[]]$Skill,
    
    [Parameter(Mandatory=$false)]
    [string]$Model = $env:MODEL
)

# Set default model - always use gpt-5-mini
$Model = "gpt-5-mini"

# Check if copilot CLI is available
try {
    $copilotVersion = & copilot --version 2>&1
    Write-Host "Using Copilot CLI: $copilotVersion"
} catch {
    Write-Error "GitHub Copilot CLI not found. Install with: npm i -g @github/copilot"
    exit 1
}

# Check if prompt file exists
if (-not (Test-Path $Prompt)) {
    Write-Error "Prompt file not found: $Prompt"
    exit 1
}

# Check if PRD file exists
$Prd = "..\..\prd.json"
if (-not (Test-Path $Prd)) {
    Write-Error "PRD file not found: $Prd"
    exit 1
}

# Build context file
$contextFile = [System.IO.Path]::GetTempFileName()
Write-Host "Building context file: $contextFile"

# Add progress.txt if it exists
if (Test-Path "..\..\progress.txt") {
    Get-Content "..\..\progress.txt" | Add-Content $contextFile
    Add-Content $contextFile "`n`n---`n`n"
}

# Add PRD
Write-Host "Attaching PRD: $Prd"
Add-Content $contextFile "# Product Requirements Document"
Add-Content $contextFile ""
Get-Content $Prd | Add-Content $contextFile
Add-Content $contextFile "`n`n---`n`n"

# Add skills if specified
if ($Skill) {
    foreach ($skillName in $Skill) {
        $skillFile = "skills\$skillName\SKILL.md"
        if (Test-Path $skillFile) {
            Write-Host "Adding skill: $skillName"
            Get-Content $skillFile | Add-Content $contextFile
            Add-Content $contextFile "`n`n---`n`n"
        } else {
            Write-Warning "Skill file not found: $skillFile"
        }
    }
}

# Add main prompt
Write-Host "Adding prompt: $Prompt"
Get-Content $Prompt | Add-Content $contextFile

# Build copilot command with permissions
$copilotArgs = @()
$copilotArgs += "--model"
$copilotArgs += $Model
$copilotArgs += "--allow-all-tools"
$copilotArgs += "--silent"

# Run copilot with piped input
Write-Host "Piping context to Copilot CLI..."
try {
    $output = Get-Content $contextFile -Raw | & copilot @copilotArgs 2>&1
    $exitCode = $LASTEXITCODE
    Write-Host "Copilot exit code: $exitCode"
    Write-Host "Copilot output received:"
    Write-Host $output
} catch {
    Write-Error "Failed to run Copilot CLI: $_"
    Remove-Item $contextFile -ErrorAction SilentlyContinue
    exit 1
}

# Parse the JSON output and apply changes
try {
    $jsonOutput = $output | ConvertFrom-Json
    Write-Host "Parsed implementation: $($jsonOutput.feature)"
    
    foreach ($change in $jsonOutput.changes) {
        $filePath = $change.file
        $action = $change.action
        $content = $change.content
        
        if ($action -eq "create" -or $action -eq "edit") {
            Write-Host "Creating/Updating file: $filePath"
            $content | Set-Content -Path $filePath -Encoding UTF8
        } elseif ($action -eq "delete") {
            Write-Host "Deleting file: $filePath"
            if (Test-Path $filePath) {
                Remove-Item $filePath
            }
        }
    }
    
    Write-Host "Summary: $($jsonOutput.summary)"
    
    # Update progress.txt
    if (Test-Path "..\..\progress.txt") {
        $progressContent = Get-Content "..\..\progress.txt" -Raw
    } else {
        $progressContent = ""
    }
    $newProgress = $progressContent + "`n$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $($jsonOutput.feature) - $($jsonOutput.summary)"
    $newProgress | Set-Content -Path "..\..\progress.txt" -Encoding UTF8
    
} catch {
    Write-Warning "Failed to parse Copilot output as JSON or apply changes: $_"
    Write-Host "Raw output was:"
    Write-Host $output
}

# Clean up context file
Remove-Item $contextFile -ErrorAction SilentlyContinue

Write-Host "Ralph execution complete."
exit $exitCode
