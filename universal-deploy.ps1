#!/usr/bin/env pwsh
<#
.SYNOPSIS
    King AI v3 - Universal Deployment Script

.DESCRIPTION
    The ultimate deployment solution for King AI v3 with multiple deployment methods:
    - TAR.GZ (fast, reliable for Unix paths)
    - ZIP (cross-platform compatibility)
    - RSYNC (incremental updates, efficient for development)

    Deploys complete system with all services:
    - Orchestrator (FastAPI on port 8000)
    - Subagent Manager (port 8001)
    - Memory Service (port 8002)
    - MCP Gateway (port 8080)
    - Dashboard (port 3000)
    - MoltBot Multi-Channel Gateway (port 18789)

.PARAMETER IpAddress
    Target EC2 instance IP address (required)

.PARAMETER Method
    Deployment method: TAR, ZIP, or RSYNC (default: TAR)

.PARAMETER SkipBuild
    Skip local packaging if deployment package exists

.PARAMETER SkipHealthCheck
    Skip final health verification

.PARAMETER IncludeDashboard
    Include dashboard deployment (default: true)

.PARAMETER IncludeMoltBot
    Include MoltBot deployment (default: true)

.EXAMPLE
    .\universal-deploy.ps1 -IpAddress 52.90.206.76

.EXAMPLE
    .\universal-deploy.ps1 -IpAddress 52.90.206.76 -Method RSYNC -SkipBuild

.EXAMPLE
    .\universal-deploy.ps1 -IpAddress 52.90.206.76 -Method ZIP -IncludeDashboard:$false
#>

param(
    [Parameter(Mandatory=$true, HelpMessage="Enter the IP address of the EC2 instance")]
    [ValidatePattern('^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')]
    [string]$IpAddress,

    [Parameter(Mandatory=$false)]
    [ValidateSet('TAR', 'ZIP', 'RSYNC')]
    [string]$Method = 'TAR',

    [Parameter(Mandatory=$false)]
    [switch]$SkipBuild,

    [Parameter(Mandatory=$false)]
    [switch]$SkipHealthCheck,

    [Parameter(Mandatory=$false)]
    [bool]$IncludeDashboard = $true,

    [Parameter(Mandatory=$false)]
    [bool]$IncludeMoltBot = $true
)

# Configuration
$KeyPath = "C:\Users\dmilner.AGV-040318-PC\Downloads\landon\king-ai-v2\king-ai-v3\agentic-framework-main\king-ai-studio.pem"
$ErrorActionPreference = "Stop"
$LogFile = "universal-deployment_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$script:StartTime = Get-Date

# Service definitions
$Services = @(
    @{Name="Orchestrator"; Port=8000; Url="http://${IpAddress}:8000/health"}
    @{Name="Subagent Manager"; Port=8001; Url="http://${IpAddress}:8001/health"}
    @{Name="Memory Service"; Port=8002; Url="http://${IpAddress}:8002/health"}
    @{Name="MCP Gateway"; Port=8080; Url="http://${IpAddress}:8080/health"}
)

if ($IncludeDashboard) {
    $Services += @{Name="Dashboard"; Port=3000; Url="http://${IpAddress}:3000"}
}

if ($IncludeMoltBot) {
    $Services += @{Name="MoltBot"; Port=18789; Url="http://${IpAddress}:18789"}
}

# Logging functions
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
    Write-Log "[$Step] [${elapsed}s] $Message" "INFO"
}

function Show-Progress {
    param([string]$Activity, [string]$Status, [int]$PercentComplete)
    Write-Progress -Activity $Activity -Status $Status -PercentComplete $PercentComplete
}

# SSH functions
function Invoke-SSHCommand {
    param([string]$Command, [int]$MaxRetries = 3, [string]$Description = "SSH command")

    for ($i = 1; $i -le $MaxRetries; $i++) {
        Write-Log "  Attempting $Description (try $i/$MaxRetries)..." "INFO"
        try {
            $result = ssh -i "$KeyPath" -o ConnectTimeout=30 -o StrictHostKeyChecking=no ubuntu@$IpAddress $Command 2>&1
            if ($LASTEXITCODE -eq 0) {
                return $result
            }
        } catch {
            # Continue to retry
        }

        if ($i -lt $MaxRetries) {
            Write-Log "  Failed, retrying in 5 seconds..." "WARNING"
            Start-Sleep -Seconds 5
        }
    }

    throw "SSH command failed after $MaxRetries attempts: $Description"
}

function Invoke-SCPWithVerification {
    param([string]$LocalPath, [string]$RemotePath, [int]$MaxRetries = 3)

    $fileName = Split-Path $LocalPath -Leaf
    $localHash = Get-FileHash $LocalPath -Algorithm SHA256 | Select-Object -ExpandProperty Hash

    for ($i = 1; $i -le $MaxRetries; $i++) {
        Write-Log "  Upload attempt $i/$MaxRetries..." "INFO"

        scp -i "$KeyPath" -o ConnectTimeout=30 -o StrictHostKeyChecking=no $LocalPath ubuntu@${IpAddress}:$RemotePath 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            if ($i -eq $MaxRetries) {
                throw "File upload failed after $MaxRetries attempts"
            }
            Start-Sleep -Seconds 5
            continue
        }

        Write-Log "  Verifying file integrity..." "INFO"
        $remoteHash = Invoke-SSHCommand "sha256sum $RemotePath/$fileName 2>/dev/null | cut -d' ' -f1" -Description "file verification"
        if ($remoteHash -and $remoteHash.Trim() -eq $localHash) {
            Write-Log "  [OK] File uploaded and verified successfully" "SUCCESS"
            return
        } else {
            Write-Log "  [ERROR] File verification failed, hash mismatch" "WARNING"
            if ($i -eq $MaxRetries) {
                throw "File verification failed after $MaxRetries attempts"
            }
            Start-Sleep -Seconds 5
        }
    }
}

# Deployment method implementations
function New-TarDeployment {
    param([bool]$IncludeDashboard, [bool]$IncludeMoltBot)

    Write-Step "BUILD" "Creating TAR.GZ deployment package..."

    $tarPath = "C:\Program Files\Git\usr\bin\tar.exe"
    if (!(Test-Path $tarPath)) {
        $tarPath = "C:\Program Files\Git\bin\tar.exe"
    }

    if (Test-Path "king-ai-deploy.tar.gz") {
        Write-Log "  Using existing king-ai-deploy.tar.gz..." "INFO"
        $fileSize = (Get-Item "king-ai-deploy.tar.gz").Length
        Write-Log "  File size: $([math]::Round($fileSize/1MB, 2)) MB" "INFO"
        return
    }

    if (!(Test-Path $tarPath)) {
        throw "Neither king-ai-deploy.tar.gz nor tar.exe found. Install Git for Windows."
    }

    Write-Log "  Creating fresh tar.gz archive..." "INFO"

    $excludeArgs = @(
        "--exclude=node_modules", "--exclude=__pycache__", "--exclude=.git",
        "--exclude=venv", "--exclude=.venv", "--exclude=.pytest_cache",
        "--exclude=*.pyc", "--exclude=*.pyo", "--exclude=*.log"
    )

    $tarArgs = $excludeArgs + @("-czf", "king-ai-deploy.tar.gz", "-C", "king-ai-v3", "agentic-framework-main")

    & $tarPath @tarArgs 2>&1 | Out-Null

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create deployment archive"
    }

    $fileSize = (Get-Item "king-ai-deploy.tar.gz").Length
    Write-Log "  [ERROR]� Archive created: $([math]::Round($fileSize/1MB, 2)) MB" "SUCCESS"
}

function New-ZipDeployment {
    param([bool]$IncludeDashboard, [bool]$IncludeMoltBot)

    Write-Step "BUILD" "Creating ZIP deployment package..."

    if (Test-Path "deploy.zip") {
        Write-Log "  Using existing deploy.zip..." "INFO"
        $fileSize = (Get-Item "deploy.zip").Length
        Write-Log "  File size: $([math]::Round($fileSize/1MB, 2)) MB" "INFO"
        return
    }

    Write-Log "  Creating ZIP archive..." "INFO"

    $compress = @{
        Path = "king-ai-v3\agentic-framework-main"
        CompressionLevel = "Optimal"
        DestinationPath = "deploy.zip"
    }

    Compress-Archive @compress

    $fileSize = (Get-Item "deploy.zip").Length
    Write-Log "  [ERROR]� Archive created: $([math]::Round($fileSize/1MB, 2)) MB" "SUCCESS"
}

function New-RsyncDeployment {
    param([bool]$IncludeDashboard, [bool]$IncludeMoltBot)

    Write-Step "SYNC" "Preparing RSYNC deployment..."

    $rsyncPath = "C:\Program Files\Git\usr\bin\rsync.exe"
    if (!(Test-Path $rsyncPath)) {
        throw "RSYNC not found. Install Git for Windows with RSYNC support."
    }

    Write-Log "  [ERROR]� RSYNC ready for deployment" "SUCCESS"
}

function Invoke-TarDeployment {
    Write-Step "UPLOAD" "Uploading TAR.GZ package..."
    Invoke-SCPWithVerification "king-ai-deploy.tar.gz" "/home/ubuntu"

    Write-Step "EXTRACT" "Extracting files on server..."
    Invoke-SSHCommand "cd /home/ubuntu; tar -xzf king-ai-deploy.tar.gz" -Description "TAR extraction"
    Write-Log "  [ERROR]� Files extracted successfully" "SUCCESS"
}

function Invoke-ZipDeployment {
    Write-Step "UPLOAD" "Uploading ZIP package..."
    Invoke-SCPWithVerification "deploy.zip" "/home/ubuntu"

    Write-Step "EXTRACT" "Extracting files on server..."
    Invoke-SSHCommand "cd /home/ubuntu; unzip -q deploy.zip; mv agentic-framework-main king-ai-v3/" -Description "ZIP extraction"
    Write-Log "  [ERROR]� Files extracted successfully" "SUCCESS"
}

function Invoke-RsyncDeployment {
    param([bool]$IncludeDashboard, [bool]$IncludeMoltBot)

    Write-Step "SYNC" "Syncing files with RSYNC..."

    $rsyncPath = "C:\Program Files\Git\usr\bin\rsync.exe"

    # Clean remote first
    Invoke-SSHCommand "sudo rm -rf /home/ubuntu/agentic-framework-main /home/ubuntu/dashboard 2>/dev/null; rm -rf /home/ubuntu/agentic-framework-main /home/ubuntu/dashboard 2>/dev/null; mkdir -p /home/ubuntu/agentic-framework-main /home/ubuntu/dashboard" -Description "remote cleanup"

    # Sync main framework
    Write-Log "  Syncing agentic-framework-main/..." "INFO"
    & $rsyncPath -avz --delete `
        --exclude=node_modules --exclude=__pycache__ --exclude=.git `
        --exclude=venv --exclude=.venv --exclude='*.pyc' --exclude='*.log' `
        -e "ssh -i '$KeyPath' -o StrictHostKeyChecking=no" `
        "./king-ai-v3/agentic-framework-main/" `
        "ubuntu@${IpAddress}:/home/ubuntu/agentic-framework-main/" | Out-Null

    # Sync dashboard if requested
    if ($IncludeDashboard -and (Test-Path "./dashboard")) {
        Write-Log "  Syncing dashboard/..." "INFO"
        & $rsyncPath -avz --delete `
            --exclude=node_modules --exclude=__pycache__ --exclude=.git `
            --exclude=dist --exclude=build --exclude='*.log' `
            -e "ssh -i '$KeyPath' -o StrictHostKeyChecking=no" `
            "./dashboard/" `
            "ubuntu@${IpAddress}:/home/ubuntu/dashboard/" | Out-Null
    }

    Write-Log "  [ERROR]� Files synced successfully" "SUCCESS"
}

function Invoke-ServerSetup {
    param([bool]$IncludeDashboard, [bool]$IncludeMoltBot)

    Write-Step "SETUP" "Setting up server environment..."

    # Copy the server setup template
    Write-Log "  Loading server setup script..." "INFO"
    if (!(Test-Path "server-setup-template.sh")) {
        throw "server-setup-template.sh not found"
    }
    
    $serverScript = Get-Content "server-setup-template.sh" -Raw
    $serverScript = $serverScript -replace "__AWS_IP__", $IpAddress

    # Add optional services - write directly to file
    $tempFile = "server-setup-generated.sh"
    $serverScript | Out-File -FilePath $tempFile -Encoding UTF8 -NoNewline
    
    if ($IncludeDashboard) {
        $amp = [char]38  # &
        $gt = [char]62   # >
        $redir = '2' + $gt + $amp + '1'
        $nohup = "    nohup npx serve -s dist -l 3000 $gt /tmp/dashboard.log $redir $amp"
        
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "`n`necho `"=== DASHBOARD SETUP ===`"`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "if [ -d `"/home/ubuntu/dashboard`" ]; then`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    cd /home/ubuntu/dashboard`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    npm install --legacy-peer-deps --quiet`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    NODE_ENV=production npm run build --quiet`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), $nohup + "`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    echo `$! $gt /tmp/dashboard.pid`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    cd ..`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    echo `"[OK] Dashboard started on port 3000`"`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    echo `"  3000 (Dashboard)`"`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "fi`n")
    }

    if ($IncludeMoltBot) {
        $amp = [char]38  # &
        $gt = [char]62   # >
        $redir = '2' + $gt + $amp + '1'
        $nohup = "    nohup python -m moltbot --config moltbot.json $gt /tmp/moltbot.log $redir $amp"
        
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "`n`necho `"=== MOLTBOT SETUP ===`"`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "if [ -f `"/home/ubuntu/agentic-framework-main/moltbot.json`" ]; then`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    cd /home/ubuntu/agentic-framework-main`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    source .venv/bin/activate`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    export PYTHONPATH=`"/home/ubuntu/king-ai-v3/agentic-framework-main:/home/ubuntu/agentic-framework-main:/home/ubuntu:`$PYTHONPATH`"`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), $nohup + "`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    echo `$! $gt /tmp/moltbot.pid`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    cd ..`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    echo `"[OK] MoltBot started on port 18789`"`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "    echo `"  18789 (MoltBot)`"`n")
        [IO.File]::AppendAllText((Resolve-Path $tempFile), "fi`n")
    }

    # Upload and execute server setup
    Invoke-SCPWithVerification $tempFile "/home/ubuntu"
    Remove-Item $tempFile

    Invoke-SSHCommand "chmod +x /home/ubuntu/server-setup-generated.sh" -Description "setting script permissions"
    Invoke-SSHCommand "/home/ubuntu/server-setup-generated.sh" -Description "server setup execution"

    Write-Log "[ERROR]� Server setup completed successfully" "SUCCESS"
}

function Test-ServiceHealth {
    Write-Step "HEALTH" "Verifying service health..."

    $healthy = 0
    $total = $Services.Count

    foreach ($svc in $Services) {
        try {
            if ($svc.Url -match "/health") {
                $response = Invoke-WebRequest -Uri $svc.Url -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
                Write-Log "  [OK] $($svc.Name) - HEALTHY" "SUCCESS"
                $healthy++
            } else {
                # For services without health endpoints, just check if port is listening
                $portCheck = Invoke-SSHCommand "netstat -tln 2>/dev/null | grep :$($svc.Port) `'||`' ss -tln | grep :$($svc.Port)" -Description "port check for $($svc.Name)"
                if ($portCheck) {
                    Write-Log "  [OK] $($svc.Name) - RUNNING (port $($svc.Port))" "SUCCESS"
                    $healthy++
                } else {
                    Write-Log "  [FAIL] $($svc.Name) - NOT LISTENING (port $($svc.Port))" "ERROR"
                }
            }
        } catch {
            Write-Log "  [FAIL] $($svc.Name) - NOT RESPONDING" "ERROR"
        }
    }

    Write-Host ""
    Write-Step "SUMMARY" "Deployment Results: $healthy/$total services healthy"

    if ($healthy -eq $total) {
        Write-Log "[SUCCESS] DEPLOYMENT SUCCESSFUL! All services are running." "SUCCESS"
    } elseif ($healthy -ge 4) {
        Write-Log "[WARNING] DEPLOYMENT MOSTLY SUCCESSFUL ($healthy/$total services)" "WARNING"
    } else {
        Write-Log "[ERROR] DEPLOYMENT INCOMPLETE ($healthy/$total services)" "ERROR"
        throw "Deployment failed: Only $healthy/$total services are healthy"
    }
}

# Main execution
try {
    Clear-Host
    Write-Log "=================================================================" "INFO"
    Write-Log "                                                                " "INFO"
    Write-Log "           KING AI v3 - UNIVERSAL DEPLOYMENT SCRIPT             " "INFO"
    Write-Log "                                                                " "INFO"
    Write-Log "=================================================================" "INFO"
    Write-Host ""
    Write-Log "Deployment started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "INFO"
    Write-Log "Target Server: $IpAddress" "INFO"
    Write-Log "Deployment Method: $Method" "INFO"
    Write-Log "Include Dashboard: $IncludeDashboard" "INFO"
    Write-Log "Include MoltBot: $IncludeMoltBot" "INFO"
    Write-Log "Log File: $LogFile" "INFO"
    Write-Host ""

    # Phase 1: Validation
    Show-Progress -Activity "Deployment" -Status "Validating prerequisites..." -PercentComplete 5
    Write-Step "INIT" "Validating prerequisites..."

    if (!(Test-Path $KeyPath)) {
        throw "SSH key not found at: $KeyPath"
    }
    Write-Log "[ERROR]� SSH key found" "SUCCESS"

    Write-Step "INIT" "Testing SSH connectivity..."
    $sshTest = Invoke-SSHCommand "echo 'SSH connection successful'" -Description "SSH connectivity test"
    Write-Log "[ERROR]� SSH connection established" "SUCCESS"

    # Phase 2: Build/Package
    Show-Progress -Activity "Deployment" -Status "Building deployment package..." -PercentComplete 20
    if (!$SkipBuild) {
        switch ($Method) {
            'TAR' { New-TarDeployment -IncludeDashboard $IncludeDashboard -IncludeMoltBot $IncludeMoltBot }
            'ZIP' { New-ZipDeployment -IncludeDashboard $IncludeDashboard -IncludeMoltBot $IncludeMoltBot }
            'RSYNC' { New-RsyncDeployment -IncludeDashboard $IncludeDashboard -IncludeMoltBot $IncludeMoltBot }
        }
    } else {
        Write-Step "BUILD" "Skipping build (SkipBuild specified)"
    }

    # Phase 3: Upload/Deploy
    Show-Progress -Activity "Deployment" -Status "Uploading files to server..." -PercentComplete 50
    switch ($Method) {
        'TAR' { Invoke-TarDeployment }
        'ZIP' { Invoke-ZipDeployment }
        'RSYNC' { Invoke-RsyncDeployment -IncludeDashboard $IncludeDashboard -IncludeMoltBot $IncludeMoltBot }
    }

    # Phase 4: Server Setup
    Show-Progress -Activity "Deployment" -Status "Setting up server environment..." -PercentComplete 80
    Invoke-ServerSetup -IncludeDashboard $IncludeDashboard -IncludeMoltBot $IncludeMoltBot

    # Phase 5: Health Check
    Show-Progress -Activity "Deployment" -Status "Verifying service health..." -PercentComplete 95
    if (!$SkipHealthCheck) {
        Test-ServiceHealth
    } else {
        Write-Step "HEALTH" "Skipping health check (SkipHealthCheck specified)"
    }

    Write-Progress -Activity "Deployment" -Completed

    # Final Summary
    $totalTime = [math]::Round(((Get-Date) - $script:StartTime).TotalMinutes, 1)
    Write-Host ""
    Write-Log "=================================================================" "SUCCESS"
    Write-Log "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!" "SUCCESS"
    Write-Log "Total deployment time: $totalTime minutes" "SUCCESS"
    Write-Log "=================================================================" "SUCCESS"
    Write-Host ""

    Write-Log "Access Information:" "INFO"
    Write-Log "  API: http://${IpAddress}/api" "INFO"
    Write-Log "  Docs: http://${IpAddress}/docs" "INFO"
    Write-Log "  Health: http://${IpAddress}/health" "INFO"
    if ($IncludeDashboard) { Write-Log "  Dashboard: http://${IpAddress}:3000" "INFO" }
    if ($IncludeMoltBot) { Write-Log "  MoltBot: http://${IpAddress}:18789" "INFO" }
    Write-Host ""

    Write-Log "Log Commands:" "INFO"
    Write-Log "  ssh ubuntu@${IpAddress} 'tail -f /tmp/orchestrator.log'" "INFO"
    Write-Log "  ssh ubuntu@${IpAddress} 'tail -f /tmp/subagent-manager.log'" "INFO"
    Write-Log "  ssh ubuntu@${IpAddress} 'tail -f /tmp/memory-service.log'" "INFO"
    Write-Log "  ssh ubuntu@${IpAddress} 'tail -f /tmp/mcp-gateway.log'" "INFO"
    if ($IncludeDashboard) { Write-Log "  ssh ubuntu@${IpAddress} 'tail -f /tmp/dashboard.log'" "INFO" }
    if ($IncludeMoltBot) { Write-Log "  ssh ubuntu@${IpAddress} 'tail -f /tmp/moltbot.log'" "INFO" }
    Write-Host ""

    Write-Log "Log file saved to: $LogFile" "INFO"

} catch {
    Write-Progress -Activity "Deployment" -Completed
    Write-Log "=================================================================" "ERROR"
    Write-Log "❌ DEPLOYMENT FAILED!" "ERROR"
    Write-Log "Error: $($_.Exception.Message)" "ERROR"
    Write-Log "=================================================================" "ERROR"
    Write-Host ""
    Write-Log "Check the log file for details: $LogFile" "WARNING"
    Write-Host ""
    throw
}
