#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Universal King AI v3 Deployment Script
    
.DESCRIPTION
    Deploys the complete King AI v3 system to an AWS EC2 instance with all components:
    - Orchestrator (FastAPI on port 8000)
    - Memory Service (port 8002)
    - MCP Gateway (port 8080)
    - Subagent Manager (port 8001)
    - Dashboard (port 3000)
    - MoltBot Multi-Channel Gateway (port 18789)
    - Ollama LLM Runtime with DeepSeek R1 7B
    
.PARAMETER IpAddress
    The IP address of the target EC2 instance (required)
    
.PARAMETER SkipBuild
    Skip local build steps if deploy.zip already exists
    
.EXAMPLE
    .\deploy.ps1 -IpAddress 52.90.206.76
    
.EXAMPLE
    .\deploy.ps1 -IpAddress 100.24.50.240
    
.EXAMPLE
    .\deploy.ps1 -IpAddress 52.90.206.76 -SkipBuild
#>

param(
    [Parameter(Mandatory=$true, HelpMessage="Enter the IP address of the EC2 instance")]
    [ValidatePattern('^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')]
    [string]$IpAddress,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipBuild
)

# Hardcoded SSH key path
$KeyPath = "C:\Users\dmilner.AGV-040318-PC\Downloads\landon\king-ai-v2\king-ai-v3\agentic-framework-main\king-ai-studio.pem"

# Error handling
$ErrorActionPreference = "Stop"

# Logging setup
$LogFile = "deployment_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$script:StartTime = Get-Date

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Add-Content -Path $LogFile -Value $logMessage
    
    switch ($Level) {
        "SUCCESS" { Write-Host $Message -ForegroundColor Green }
        "INFO" { Write-Host $Message -ForegroundColor Cyan }
        "WARNING" { Write-Host $Message -ForegroundColor Yellow }
        "ERROR" { Write-Host $Message -ForegroundColor Red }
        default { Write-Host $Message }
    }
}

function Write-Step {
    param([string]$Step, [string]$Message)
    $elapsed = [math]::Round(((Get-Date) - $script:StartTime).TotalSeconds, 1)
    Write-Log "[$Step] [Elapsed: ${elapsed}s] $Message" "INFO"
}

# Colors for output (legacy support)
function Write-Success { Write-Log $args "SUCCESS" }
function Write-Info { Write-Log $args "INFO" }
function Write-Warning { Write-Log $args "WARNING" }
function Write-Error { Write-Log $args "ERROR" }

# Banner
Clear-Host
Write-Log "================================================================" "INFO"
Write-Log "                                                                " "INFO"
Write-Log "           KING AI v3 - UNIVERSAL DEPLOYMENT SCRIPT             " "INFO"
Write-Log "                                                                " "INFO"
Write-Log "================================================================" "INFO"
Write-Host ""
Write-Log "Deployment started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "INFO"
Write-Log "Target Server: $IpAddress" "INFO"
Write-Log "SSH Key: $KeyPath" "INFO"
Write-Log "Log File: $LogFile" "INFO"
Write-Host ""

# Validation
Write-Step "INIT" "Validating prerequisites..."
if (-not (Test-Path $KeyPath)) {
    Write-Log "ERROR: SSH key not found at: $KeyPath" "ERROR"
    Write-Log "Please provide a valid key path using -KeyPath parameter" "ERROR"
    exit 1
}
Write-Step "INIT" "SSH key found and validated"

# Test SSH connectivity
Write-Step "INIT" "Testing SSH connectivity to $IpAddress..."
$sshTest = ssh -i $KeyPath -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$IpAddress "echo 'Connection successful'" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Cannot connect to $IpAddress" "ERROR"
    Write-Log "SSH output: $sshTest" "ERROR"
    Write-Host ""
    Write-Log "Troubleshooting:" "WARNING"
    Write-Log "  1. Verify security group allows SSH (port 22) from your IP" "WARNING"
    Write-Log "  2. Verify key file permissions (should not be too open)" "WARNING"
    Write-Log "  3. Verify instance is running: aws ec2 describe-instances" "WARNING"
    Write-Log "  4. Try: ssh -i $KeyPath ubuntu@$IpAddress" "WARNING"
    exit 1
}
Write-Step "INIT" "SSH connection successful"
Write-Host ""

# Create deployment package
if (-not $SkipBuild -or -not (Test-Path "deploy.zip")) {
    Write-Log "===============================================================" "INFO"
    Write-Log "STEP 1: Creating Deployment Package" "INFO"
    Write-Log "===============================================================" "INFO"
    Write-Step "BUILD" "Starting deployment package creation..."
    
    # Remove old package
    if (Test-Path "deploy.zip") {
        Write-Step "BUILD" "Removing old deploy.zip..."
        Remove-Item "deploy.zip" -Force
    }
    
    # Create staging directory
    $stagingDir = "deploy-staging"
    Write-Step "BUILD" "Creating staging directory: $stagingDir"
    if (Test-Path $stagingDir) {
        Write-Step "BUILD" "Cleaning existing staging directory..."
        Remove-Item $stagingDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stagingDir | Out-Null
    Write-Step "BUILD" "Staging directory ready"
    
    # Copy agentic-framework-main
    Write-Step "BUILD" "Copying agentic-framework-main directory..."
    $copyStart = Get-Date
    robocopy "king-ai-v3\agentic-framework-main" "$stagingDir\agentic-framework-main" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    $copyDuration = [math]::Round(((Get-Date) - $copyStart).TotalSeconds, 1)
    Write-Step "BUILD" "Copied agentic-framework-main (${copyDuration}s)"
    
    # Copy dashboard
    if (Test-Path "dashboard") {
        Write-Step "BUILD" "Copying dashboard directory..."
        robocopy "dashboard" "$stagingDir\dashboard" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
        Write-Step "BUILD" "Dashboard copied"
    } elseif (Test-Path "king-ai-v3\dashboard") {
        Write-Step "BUILD" "Copying dashboard directory from king-ai-v3..."
        robocopy "king-ai-v3\dashboard" "$stagingDir\dashboard" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
        Write-Step "BUILD" "Dashboard copied"
    } else {
        Write-Log "Dashboard not found - skipping" "WARNING"
    }
    
    # Create ZIP
    Write-Step "BUILD" "Creating deploy.zip archive..."
    $zipStart = Get-Date
    Compress-Archive -Path "$stagingDir\*" -DestinationPath "deploy.zip" -Force
    $zipDuration = [math]::Round(((Get-Date) - $zipStart).TotalSeconds, 1)
    
    # Cleanup
    Write-Step "BUILD" "Cleaning up staging directory..."
    Remove-Item $stagingDir -Recurse -Force
    
    $zipSize = (Get-Item "deploy.zip").Length / 1MB
    Write-Log "Deployment package created: $([math]::Round($zipSize, 2)) MB (compressed in ${zipDuration}s)" "SUCCESS"
} else {
    Write-Log "Using existing deploy.zip (SkipBuild enabled)" "INFO"
    $zipSize = (Get-Item "deploy.zip").Length / 1MB
    Write-Log "Existing package size: $([math]::Round($zipSize, 2)) MB" "INFO"
}
Write-Host ""

# Upload deployment package
Write-Log "===============================================================" "INFO"
Write-Log "STEP 2: Uploading to $IpAddress" "INFO"
Write-Log "===============================================================" "INFO"

Write-Step "UPLOAD" "Uploading deploy.zip ($(([math]::Round((Get-Item 'deploy.zip').Length / 1MB, 2))) MB)..."
$uploadStart = Get-Date
scp -i $KeyPath -o StrictHostKeyChecking=no deploy.zip ubuntu@${IpAddress}:/home/ubuntu/
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Failed to upload deploy.zip" "ERROR"
    exit 1
}
$uploadDuration = [math]::Round(((Get-Date) - $uploadStart).TotalSeconds, 1)
Write-Log "deploy.zip uploaded successfully (${uploadDuration}s)" "SUCCESS"

# Upload deployment script
Write-Step "UPLOAD" "Uploading enhanced-deploy.sh..."
scp -i $KeyPath -o StrictHostKeyChecking=no enhanced-deploy.sh ubuntu@${IpAddress}:/home/ubuntu/
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Failed to upload enhanced-deploy.sh" "ERROR"
    exit 1
}
Write-Step "UPLOAD" "enhanced-deploy.sh uploaded"

# Upload service startup script
if (Test-Path "start_all_services.sh") {
    Write-Step "UPLOAD" "Uploading start_all_services.sh..."
    scp -i $KeyPath -o StrictHostKeyChecking=no start_all_services.sh ubuntu@${IpAddress}:/home/ubuntu/king-ai-v3/ 2>&1 | Out-Null
    Write-Step "UPLOAD" "start_all_services.sh uploaded"
}
Write-Log "All files uploaded successfully" "SUCCESS"
Write-Host ""

# Execute deployment
Write-Log "===============================================================" "INFO"
Write-Log "STEP 3: Executing Remote Deployment" "INFO"
Write-Log "===============================================================" "INFO"
Write-Log "This will take 15-30 minutes..." "WARNING"
Write-Step "DEPLOY" "Starting remote deployment on $IpAddress..."
Write-Host ""

$deployStart = Get-Date

$deploymentScript = @'
#!/bin/bash
cd /home/ubuntu
chmod +x enhanced-deploy.sh
./enhanced-deploy.sh 2>&1 | tee /tmp/deployment.log
'@

# Execute deployment and stream output
Write-Step "DEPLOY" "Executing enhanced-deploy.sh on remote server..."
Write-Log "Streaming deployment output:" "INFO"
Write-Host ""
$deploymentScript | ssh -i $KeyPath -o StrictHostKeyChecking=no ubuntu@$IpAddress "bash -s"

if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Deployment script failed with exit code $LASTEXITCODE" "ERROR"
    Write-Host ""
    Write-Log "To troubleshoot, connect via SSH:" "WARNING"
    Write-Log "  ssh -i $KeyPath ubuntu@$IpAddress" "WARNING"
    Write-Log "  cat /tmp/deployment.log" "WARNING"
    exit 1
}

$deployDuration = [math]::Round(((Get-Date) - $deployStart).TotalMinutes, 1)
Write-Host ""
Write-Log "Remote deployment completed successfully (${deployDuration} minutes)" "SUCCESS"
Write-Host ""
Write-Log "===============================================================" "INFO"
Write-Log "STEP 4: Installing Ollama and DeepSeek R1 7B" "INFO"
Write-Log "===============================================================" "INFO"

$ollamaStart = Get-Date

$ollamaSetup = @'
#!/bin/bash
# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "[OLLAMA] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "[OLLAMA] Ollama installed successfully"
else
    echo "[OLLAMA] Ollama already installed: $(ollama --version)"
fi

# Start Ollama service
echo "[OLLAMA] Starting Ollama service..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

# Check if Ollama is running
if pgrep -x "ollama" > /dev/null; then
    echo "[OLLAMA] Service started successfully"
else
    echo "[OLLAMA] WARNING: Service may not have started properly"
fi

# Pull DeepSeek R1 7B model
echo "[OLLAMA] Pulling DeepSeek R1 7B model (this may take several minutes)..."
ollama pull deepseek-r1:7b

# Verify model
echo "[OLLAMA] Verifying installed models..."
ollama list | grep deepseek-r1

echo "[OLLAMA] Setup complete"
'@

Write-Step "OLLAMA" "Installing Ollama and pulling DeepSeek R1 7B model..."
Write-Log "This may take several minutes depending on network speed..." "INFO"
$ollamaSetup | ssh -i $KeyPath -o StrictHostKeyChecking=no ubuntu@$IpAddress "bash -s"

$ollamaDuration = [math]::Round(((Get-Date) - $ollamaStart).TotalMinutes, 1)
Write-Log "Ollama and DeepSeek R1 7B ready (${ollamaDuration} minutes)" "SUCCESS"
Write-Host ""

# Health check
Write-Log "===============================================================" "INFO"
Write-Log "STEP 5: Verifying Services" "INFO"
Write-Log "===============================================================" "INFO"
Write-Step "VERIFY" "Starting service health checks..."
Write-Host ""

Write-Step "VERIFY" "Waiting 10 seconds for services to stabilize..."
Start-Sleep -Seconds 10

$healthChecks = @(
    @{Name="Nginx Proxy"; Url="http://${IpAddress}/health"; Port=80}
    @{Name="Orchestrator"; Url="http://${IpAddress}:8000/health"; Port=8000}
    @{Name="Memory Service"; Url="http://${IpAddress}:8002/health"; Port=8002}
    @{Name="MoltBot Gateway"; Url="http://${IpAddress}:18789/"; Port=18789}
    @{Name="Ollama"; Url="http://${IpAddress}:11434/"; Port=11434}
)

$healthyServices = 0
$totalServices = $healthChecks.Count

Write-Step "VERIFY" "Checking $totalServices services..."

foreach ($check in $healthChecks) {
    Write-Step "VERIFY" "Testing $($check.Name) on port $($check.Port)..."
    try {
        $response = Invoke-WebRequest -Uri $check.Url -TimeoutSec 10 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Log "  [OK] $($check.Name) (port $($check.Port)) - HEALTHY" "SUCCESS"
            $healthyServices++
        } else {
            Write-Log "  [WARNING] $($check.Name) (port $($check.Port)) - Status: $($response.StatusCode)" "WARNING"
        }
    } catch {
        Write-Log "  [FAILED] $($check.Name) (port $($check.Port)) - Not responding" "ERROR"
        Write-Log "  Error: $($_.Exception.Message)" "ERROR"
    }
}

Write-Host ""
Write-Log "===============================================================" "INFO"
$totalTime = [math]::Round(((Get-Date) - $script:StartTime).TotalMinutes, 1)
if ($healthyServices -eq $totalServices) {
    Write-Log "DEPLOYMENT SUCCESSFUL!" "SUCCESS"
    Write-Log "All $healthyServices/$totalServices services are healthy" "SUCCESS"
    Write-Log "Total deployment time: $totalTime minutes" "SUCCESS"
} elseif ($healthyServices -ge 3) {
    Write-Log "DEPLOYMENT MOSTLY SUCCESSFUL" "WARNING"
    Write-Log "$healthyServices/$totalServices services are healthy" "WARNING"
    Write-Log "Total deployment time: $totalTime minutes" "WARNING"
} else {
    Write-Log "DEPLOYMENT COMPLETED WITH ERRORS" "ERROR"
    Write-Log "Only $healthyServices/$totalServices services are healthy" "ERROR"
    Write-Log "Total deployment time: $totalTime minutes" "ERROR"
}
Write-Log "===============================================================" "INFO"
Write-Host ""

# Access information
Write-Host "================================================================" -ForegroundColor Green
Write-Host "                   ACCESS INFORMATION                           " -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "API Endpoints:" -ForegroundColor Cyan
Write-Host "  Main API:      http://${IpAddress}/api" -ForegroundColor White
Write-Host "  API Docs:      http://${IpAddress}/docs" -ForegroundColor White
Write-Host "  Health Check:  http://${IpAddress}/health" -ForegroundColor White
Write-Host "  Chat API:      http://${IpAddress}/api/chat" -ForegroundColor White
Write-Host "  OpenAI API:    http://${IpAddress}:8000/v1/chat/completions" -ForegroundColor White
Write-Host ""
Write-Host "Services:" -ForegroundColor Cyan
Write-Host "  Orchestrator:  http://${IpAddress}:8000" -ForegroundColor White
Write-Host "  Memory:        http://${IpAddress}:8002" -ForegroundColor White
Write-Host "  MCP Gateway:   http://${IpAddress}:8080" -ForegroundColor White
Write-Host "  Subagents:     http://${IpAddress}:8001" -ForegroundColor White
Write-Host "  Dashboard:     http://${IpAddress}:3000" -ForegroundColor White
Write-Host "  MoltBot UI:    http://${IpAddress}:18789" -ForegroundColor White
Write-Host "  Ollama:        http://${IpAddress}:11434" -ForegroundColor White
Write-Host ""
Write-Host "AI Model:" -ForegroundColor Cyan
Write-Host "  DeepSeek R1 7B (via Ollama)" -ForegroundColor White
Write-Host ""
Write-Host "Multi-Channel Access (MoltBot):" -ForegroundColor Cyan
Write-Host "  Telegram, Discord, Slack, WhatsApp, Signal" -ForegroundColor White
Write-Host "  Configure: ssh ubuntu@$IpAddress" -ForegroundColor White
Write-Host "  Edit: ~/.moltbot/moltbot.json" -ForegroundColor White
Write-Host ""
Write-Host "SSH Access:" -ForegroundColor Cyan
Write-Host "  ssh -i $KeyPath ubuntu@$IpAddress" -ForegroundColor White
Write-Host ""
Write-Host "Logs:" -ForegroundColor Cyan
Write-Host "  ssh ubuntu@$IpAddress 'tail -f /tmp/orchestrator.log'" -ForegroundColor White
Write-Host "  ssh ubuntu@$IpAddress 'tail -f /tmp/moltbot.log'" -ForegroundColor White
Write-Host "  ssh ubuntu@$IpAddress 'tail -f /tmp/ollama.log'" -ForegroundColor White
Write-Host ""

# Test command
Write-Host "Quick Test:" -ForegroundColor Yellow
Write-Host @"
Invoke-WebRequest -Uri "http://${IpAddress}/api/chat" ``
  -Method POST ``
  -ContentType "application/json" ``
  -Body '{"message":"Hello, King AI!","user_id":"test"}'
"@ -ForegroundColor Gray
Write-Host ""

Write-Log "Deployment complete!" "SUCCESS"
Write-Log "Full deployment log saved to: $LogFile" "INFO"
Write-Log "Deployment completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "INFO"

# Summary statistics
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "                    DEPLOYMENT SUMMARY                          " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Log "Target Server: $IpAddress" "INFO"
Write-Log "Package Size: $([math]::Round((Get-Item 'deploy.zip').Length / 1MB, 2)) MB" "INFO"
Write-Log "Healthy Services: $healthyServices/$totalServices" "INFO"
Write-Log "Total Duration: $([math]::Round(((Get-Date) - $script:StartTime).TotalMinutes, 1)) minutes" "INFO"
Write-Log "Log File: $LogFile" "INFO"
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
