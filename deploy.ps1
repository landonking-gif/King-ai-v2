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

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Banner
Clear-Host
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "                                                                " -ForegroundColor Cyan
Write-Host "           KING AI v3 - UNIVERSAL DEPLOYMENT SCRIPT             " -ForegroundColor Cyan
Write-Host "                                                                " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Info "Target Server: $IpAddress"
Write-Info "SSH Key: $KeyPath"
Write-Host ""

# Validation
if (-not (Test-Path $KeyPath)) {
    Write-Error "ERROR: SSH key not found at: $KeyPath"
    Write-Host "Please provide a valid key path using -KeyPath parameter"
    exit 1
}

# Test SSH connectivity
Write-Info "Testing SSH connectivity..."
$sshTest = ssh -i $KeyPath -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$IpAddress "echo 'Connection successful'" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Cannot connect to $IpAddress"
    Write-Host "SSH output: $sshTest"
    Write-Host ""
    Write-Host "Troubleshooting:"
    Write-Host "  1. Verify security group allows SSH (port 22) from your IP"
    Write-Host "  2. Verify key file permissions (should not be too open)"
    Write-Host "  3. Verify instance is running: aws ec2 describe-instances"
    Write-Host "  4. Try: ssh -i $KeyPath ubuntu@$IpAddress"
    exit 1
}
Write-Success "[OK] SSH connection successful"
Write-Host ""

# Create deployment package
if (-not $SkipBuild -or -not (Test-Path "deploy.zip")) {
    Write-Info "==============================================================="
    Write-Info "STEP 1: Creating Deployment Package"
    Write-Info "==============================================================="
    
    # Remove old package
    if (Test-Path "deploy.zip") {
        Remove-Item "deploy.zip" -Force
    }
    
    # Create staging directory
    $stagingDir = "deploy-staging"
    if (Test-Path $stagingDir) {
        Remove-Item $stagingDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stagingDir | Out-Null
    
    # Copy agentic-framework-main
    Write-Info "Copying agentic-framework-main..."
    robocopy "king-ai-v3\agentic-framework-main" "$stagingDir\agentic-framework-main" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    
    # Copy dashboard
    if (Test-Path "dashboard") {
        Write-Info "Copying dashboard..."
        robocopy "dashboard" "$stagingDir\dashboard" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    } elseif (Test-Path "king-ai-v3\dashboard") {
        Write-Info "Copying dashboard..."
        robocopy "king-ai-v3\dashboard" "$stagingDir\dashboard" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    }
    
    # Create ZIP
    Write-Info "Creating deploy.zip..."
    Compress-Archive -Path "$stagingDir\*" -DestinationPath "deploy.zip" -Force
    
    # Cleanup
    Remove-Item $stagingDir -Recurse -Force
    
    $zipSize = (Get-Item "deploy.zip").Length / 1MB
    Write-Success "Deployment package created ($([math]::Round($zipSize, 2)) MB)"
} else {
    Write-Info "Using existing deploy.zip (SkipBuild enabled)"
}
Write-Host ""

# Upload deployment package
Write-Info "==============================================================="
Write-Info "STEP 2: Uploading to $IpAddress"
Write-Info "==============================================================="

Write-Info "Uploading deploy.zip..."
scp -i $KeyPath -o StrictHostKeyChecking=no deploy.zip ubuntu@${IpAddress}:/home/ubuntu/
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Failed to upload deploy.zip"
    exit 1
}
Write-Success "[OK] Upload complete"
Write-Host ""

# Upload deployment script
Write-Info "Uploading deployment script..."
scp -i $KeyPath -o StrictHostKeyChecking=no enhanced-deploy.sh ubuntu@${IpAddress}:/home/ubuntu/
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Failed to upload enhanced-deploy.sh"
    exit 1
}
Write-Success "[OK] Script uploaded"
Write-Host ""

# Upload service startup script
if (Test-Path "start_all_services.sh") {
    Write-Info "Uploading service startup script..."
    scp -i $KeyPath -o StrictHostKeyChecking=no start_all_services.sh ubuntu@${IpAddress}:/home/ubuntu/king-ai-v3/ 2>&1 | Out-Null
}
Write-Host ""

# Execute deployment
Write-Info "==============================================================="
Write-Info "STEP 3: Executing Remote Deployment"
Write-Info "==============================================================="
Write-Warning "This will take 15-30 minutes..."
Write-Host ""

$deploymentScript = @'
#!/bin/bash
cd /home/ubuntu
chmod +x enhanced-deploy.sh
./enhanced-deploy.sh 2>&1 | tee /tmp/deployment.log
'@

# Execute deployment and stream output
$deploymentScript | ssh -i $KeyPath -o StrictHostKeyChecking=no ubuntu@$IpAddress "bash -s"

if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Deployment script failed"
    Write-Host ""
    Write-Host "To troubleshoot, connect via SSH:"
    Write-Host "  ssh -i $KeyPath ubuntu@$IpAddress"
    Write-Host "  cat /tmp/deployment.log"
    exit 1
}

Write-Host ""
Write-Info "==============================================================="
Write-Info "STEP 4: Installing Ollama and DeepSeek R1 7B"
Write-Info "==============================================================="

$ollamaSetup = @'
#!/bin/bash
# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama service
echo "Starting Ollama..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

# Pull DeepSeek R1 7B model
echo "Pulling DeepSeek R1 7B model..."
ollama pull deepseek-r1:7b

# Verify model
echo "Verifying model..."
ollama list | grep deepseek-r1

echo "Ollama setup complete"
'@

Write-Info "Installing Ollama and pulling DeepSeek R1 7B model..."
$ollamaSetup | ssh -i $KeyPath -o StrictHostKeyChecking=no ubuntu@$IpAddress "bash -s"
Write-Success "[OK] Ollama and DeepSeek R1 7B ready"
Write-Host ""

# Health check
Write-Info "==============================================================="
Write-Info "STEP 5: Verifying Services"
Write-Info "==============================================================="
Write-Host ""

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

foreach ($check in $healthChecks) {
    try {
        $response = Invoke-WebRequest -Uri $check.Url -TimeoutSec 10 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Success "  [OK] $($check.Name) (port $($check.Port)) - HEALTHY"
            $healthyServices++
        } else {
            Write-Warning "  [WARNING] $($check.Name) (port $($check.Port)) - Status: $($response.StatusCode)"
        }
    } catch {
        Write-Error "  [FAILED] $($check.Name) (port $($check.Port)) - FAILED"
    }
}

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
if ($healthyServices -eq $totalServices) {
    Write-Success "DEPLOYMENT SUCCESSFUL!"
    Write-Success "All $healthyServices/$totalServices services are healthy"
} elseif ($healthyServices -ge 3) {
    Write-Warning "DEPLOYMENT MOSTLY SUCCESSFUL"
    Write-Warning "$healthyServices/$totalServices services are healthy"
} else {
    Write-Error "DEPLOYMENT COMPLETED WITH ERRORS"
    Write-Error "Only $healthyServices/$totalServices services are healthy"
}
Write-Host "===============================================================" -ForegroundColor Cyan
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

Write-Success "Deployment complete!"
Write-Host ""
