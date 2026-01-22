<#
.SYNOPSIS
Ralph - Looped runner with GitHub Copilot CLI

.DESCRIPTION
Runs GitHub Copilot CLI in a loop, implementing features until complete.
Based on soderlind/ralph implementation.

.PARAMETER Iterations
Maximum number of iterations to run

.PARAMETER Prompt
Path to prompt file (required)

.PARAMETER AllowProfile
Permission profile: locked, safe, dev

.PARAMETER Skill
Comma-separated list of skills to prepend

.EXAMPLE
.\ralph.ps1 -Prompt prompts\default.txt -AllowProfile safe -Iterations 10
#>

param(
    [Parameter(Mandatory=$false)]
    [int]$Iterations = 10,
    
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

Write-Host "Ralph Autonomous Loop Starting"
Write-Host "Max iterations: $Iterations"
Write-Host "Model: $Model"
Write-Host "Permissions: $AllowProfile"
Write-Host ""

# Initialize progress file if it doesn't exist
if (-not (Test-Path "progress.txt")) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    @"
# Ralph Progress Log
Started: $timestamp
Model: $Model

"@ | Set-Content "progress.txt"
}

for ($i = 1; $i -le $Iterations; $i++) {
    Write-Host ""
    Write-Host "=" * 60
    Write-Host "Ralph Iteration $i of $Iterations"
    Write-Host "=" * 60
    Write-Host ""
    
    # Build ralph-once parameters
    $ralphParams = @{
        Prompt = $Prompt
        AllowProfile = $AllowProfile
    }
    
    if ($Skill) {
        $ralphParams.Skill = $Skill
    }
    
    if ($Model) {
        $env:MODEL = $Model
    }
    
    # Run single iteration
    $scriptPath = Join-Path $PSScriptRoot "ralph-once.ps1"
    Write-Host "Script path: $scriptPath"
    Write-Host "Script exists: $(Test-Path $scriptPath)"
    
    if (Test-Path $scriptPath) {
        & $scriptPath @ralphParams
    } else {
        Write-Error "Script not found: $scriptPath"
        exit 1
    }
    
    $exitCode = $LASTEXITCODE
    
    # Log iteration result
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $status = if ($exitCode -eq 0) { "SUCCESS" } else { "FAILED" }
    "[$timestamp] Iteration $i : $status (exit code: $exitCode)" | Add-Content "progress.txt"
    
    # Check if Copilot signaled completion
    if (Test-Path "progress.txt") {
        $recentProgress = Get-Content "progress.txt" -Tail 50 | Out-String
        if ($recentProgress -match '<promise>COMPLETE</promise>') {
            Write-Host ""
            Write-Host "🎉 Copilot signaled COMPLETE - all stories done!"
            Write-Host ""
            break
        }
    }
    
    # Check PRD for completion if available
    if ($Prd -and (Test-Path $Prd)) {
        try {
            $prdContent = Get-Content $Prd -Raw | ConvertFrom-Json
            $incomplete = $prdContent | Where-Object { $_.passes -eq $false }
            
            if ($incomplete.Count -eq 0) {
                Write-Host ""
                Write-Host "🎉 All PRD items marked as passing!"
                Write-Host ""
                break
            } else {
                Write-Host "Remaining items: $($incomplete.Count)"
            }
        } catch {
            Write-Warning "Could not parse PRD file: $_"
        }
    }
    
    if ($exitCode -ne 0) {
        Write-Warning "Iteration $i failed with exit code $exitCode"
    }
    
    # Brief pause between iterations
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "Ralph loop completed after $i iterations"
Write-Host "Check progress.txt for details"
Write-Host ""
