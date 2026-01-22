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
$Prd = "prd.json"
if (-not (Test-Path $Prd)) {
    Write-Error "PRD file not found: $Prd"
    exit 1
}

# Build context file
$contextFile = [System.IO.Path]::GetTempFileName()
Write-Host "Building context file: $contextFile"

# Add progress.txt if it exists
if (Test-Path "progress.txt") {
    Get-Content "progress.txt" | Add-Content $contextFile
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
$copilotArgs += "-p"
$copilotArgs += "Follow the instructions below to implement the next feature from the PRD."

# Run copilot with piped input
Write-Host "Piping context to Copilot CLI..."
try {
    Get-Content $contextFile -Raw | & copilot @copilotArgs
    $exitCode = $LASTEXITCODE
} catch {
    Write-Error "Failed to run Copilot CLI: $_"
    Remove-Item $contextFile -ErrorAction SilentlyContinue
    exit 1
}
    
# Clean up context file
Remove-Item $contextFile -ErrorAction SilentlyContinue
    
exit $exitCode
