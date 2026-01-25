#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Complete King AI v3 System Diagnostic
.DESCRIPTION
    Tests all services and identifies what's working/broken
#>

$ErrorActionPreference = "Continue"
$SERVER_IP = "3.236.144.91"
$SSH_KEY = "king-ai-v3\agentic-framework-main\king-ai-studio.pem"

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "KING AI v3 SYSTEM DIAGNOSTICS" -ForegroundColor Cyan
Write-Host "Server: $SERVER_IP" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Function to test HTTP endpoint
function Test-Endpoint {
    param($url, $name)
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Host "✅ $name - Status: $($response.StatusCode)" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ $name - NOT RESPONDING" -ForegroundColor Red
        Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

# 1. Test SSH Connectivity
Write-Host "`n[1/7] Testing SSH Connectivity..." -ForegroundColor Yellow
try {
    $sshTest = ssh -i $SSH_KEY ubuntu@$SERVER_IP "echo 'Connected'" 2>&1
    if ($sshTest -match "Connected") {
        Write-Host "✅ SSH Connection" -ForegroundColor Green
    } else {
        Write-Host "❌ SSH Failed: $sshTest" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ SSH Failed: $_" -ForegroundColor Red
    exit 1
}

# 2. Test All Service Ports
Write-Host "`n[2/7] Testing Service Ports..." -ForegroundColor Yellow
$services = @{
    "Dashboard (3000)" = "http://${SERVER_IP}:3000"
    "Orchestrator (8000)" = "http://${SERVER_IP}:8000/api/health"
    "Subagent Manager (8001)" = "http://${SERVER_IP}:8001/health"
    "Memory Service (8002)" = "http://${SERVER_IP}:8002/health"
    "MCP Gateway (8080)" = "http://${SERVER_IP}:8080/health"
}

$working = 0
$broken = 0
foreach ($service in $services.GetEnumerator()) {
    if (Test-Endpoint -url $service.Value -name $service.Key) {
        $working++
    } else {
        $broken++
    }
}

# 3. Check Process Status on Server
Write-Host "`n[3/7] Checking Running Processes..." -ForegroundColor Yellow
$processes = ssh -i $SSH_KEY ubuntu@$SERVER_IP "ps aux | grep -E 'ollama|python.*port|uvicorn|node.*vite' | grep -v grep"
if ($processes) {
    Write-Host $processes
} else {
    Write-Host "❌ No King AI processes found!" -ForegroundColor Red
}

# 4. Check Listening Ports
Write-Host "`n[4/7] Checking Listening Ports..." -ForegroundColor Yellow
$ports = ssh -i $SSH_KEY ubuntu@$SERVER_IP "ss -tuln | grep -E ':(3000|8000|8001|8002|8080|11434)'"
if ($ports) {
    Write-Host $ports
} else {
    Write-Host "❌ No services listening on expected ports!" -ForegroundColor Red
}

# 5. Check Ollama (LLM)
Write-Host "`n[5/7] Testing Ollama LLM..." -ForegroundColor Yellow
$ollama = ssh -i $SSH_KEY ubuntu@$SERVER_IP "curl -s http://localhost:11434/api/tags 2>&1"
if ($ollama -match "llama") {
    Write-Host "✅ Ollama is running with models" -ForegroundColor Green
} else {
    Write-Host "❌ Ollama may not be running properly" -ForegroundColor Red
}

# 6. Check Docker Containers
Write-Host "`n[6/7] Checking Docker Containers..." -ForegroundColor Yellow
$containers = ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker ps --format 'table {{.Names}}\t{{.Status}}'"
if ($containers -match "NAMES") {
    Write-Host $containers
} else {
    Write-Host "⚠️  No Docker containers running (services may be running as processes)" -ForegroundColor Yellow
}

# 7. System Summary
Write-Host "`n[7/7] System Summary" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Services Working: $working / $($services.Count)" -ForegroundColor $(if($working -eq $services.Count){"Green"}else{"Yellow"})
Write-Host "Services Broken: $broken / $($services.Count)" -ForegroundColor $(if($broken -eq 0){"Green"}else{"Red"})

# Final Diagnosis
Write-Host "`n================================" -ForegroundColor Cyan
if ($working -eq $services.Count) {
    Write-Host "✅ ALL SYSTEMS OPERATIONAL" -ForegroundColor Green
} elseif ($working -gt 0) {
    Write-Host "⚠️  PARTIAL SERVICE OUTAGE" -ForegroundColor Yellow
    Write-Host "`nTo fix, try:" -ForegroundColor Cyan
    Write-Host "1. Run deployment script: wsl bash run_service.sh" -ForegroundColor White
    Write-Host "2. Or SSH to server and start missing services" -ForegroundColor White
} else {
    Write-Host "❌ SYSTEM DOWN - NO SERVICES RESPONDING" -ForegroundColor Red
    Write-Host "`nTo fix:" -ForegroundColor Cyan
    Write-Host "1. SSH to server: ssh -i $SSH_KEY ubuntu@$SERVER_IP" -ForegroundColor White
    Write-Host "2. Check logs in /tmp/*.log" -ForegroundColor White
    Write-Host "3. Restart services" -ForegroundColor White
}
Write-Host "================================`n" -ForegroundColor Cyan
