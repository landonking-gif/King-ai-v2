#!/usr/bin/env pwsh
<#
.SYNOPSIS
    King AI v3 System Diagnostics - Comprehensive health check
.DESCRIPTION
    Checks all services, connections, and identifies what's not working
#>

$ErrorActionPreference = "Continue"
$SERVER_IP = "3.236.144.91"
$SSH_KEY = "king-ai-v3/agentic-framework-main/king-ai-studio.pem"

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "KING AI v3 SYSTEM DIAGNOSTICS" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Test 1: Network connectivity
Write-Host "[1/10] Testing network connectivity..." -ForegroundColor Yellow
try {
    $ping = Test-Connection -ComputerName $SERVER_IP -Count 2 -Quiet
    if ($ping) {
        Write-Host "✅ Server is reachable at $SERVER_IP" -ForegroundColor Green
    } else {
        Write-Host "❌ Cannot reach server at $SERVER_IP" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Network test failed: $_" -ForegroundColor Red
}

# Test 2: SSH connectivity
Write-Host "`n[2/10] Testing SSH connection..." -ForegroundColor Yellow
if (Test-Path $SSH_KEY) {
    Write-Host "✅ SSH key found: $SSH_KEY" -ForegroundColor Green
    
    $sshTest = ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$SERVER_IP "echo 'SSH OK'" 2>&1
    if ($sshTest -match "SSH OK") {
        Write-Host "✅ SSH connection successful" -ForegroundColor Green
    } else {
        Write-Host "❌ SSH connection failed" -ForegroundColor Red
        Write-Host "Error: $sshTest" -ForegroundColor Red
    }
} else {
    Write-Host "❌ SSH key not found: $SSH_KEY" -ForegroundColor Red
}

# Test 3: Docker containers
Write-Host "`n[3/10] Checking Docker containers on server..." -ForegroundColor Yellow
$dockerCheck = ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null" 2>&1
if ($dockerCheck) {
    Write-Host "Docker containers:" -ForegroundColor Cyan
    Write-Host $dockerCheck
} else {
    Write-Host "❌ Cannot check Docker containers" -ForegroundColor Red
}

# Test 4: Ollama service
Write-Host "`n[4/10] Testing Ollama LLM service..." -ForegroundColor Yellow
try {
    $ollamaTest = Invoke-RestMethod -Uri "http://${SERVER_IP}:11434/api/tags" -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($ollamaTest.models) {
        Write-Host "✅ Ollama is running" -ForegroundColor Green
        Write-Host "Models available:" -ForegroundColor Cyan
        $ollamaTest.models | ForEach-Object { Write-Host "  - $($_.name)" -ForegroundColor White }
    } else {
        Write-Host "❌ Ollama responded but no models found" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Ollama not responding on port 11434" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

# Test 5: Orchestrator service
Write-Host "`n[5/10] Testing Orchestrator API..." -ForegroundColor Yellow
try {
    $orchTest = Invoke-RestMethod -Uri "http://${SERVER_IP}:8000/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Host "✅ Orchestrator is running" -ForegroundColor Green
    Write-Host "Status: $($orchTest | ConvertTo-Json -Compress)" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Orchestrator not responding on port 8000" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

# Test 6: MCP Gateway
Write-Host "`n[6/10] Testing MCP Gateway..." -ForegroundColor Yellow
try {
    $mcpTest = Invoke-RestMethod -Uri "http://${SERVER_IP}:8080/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Host "✅ MCP Gateway is running" -ForegroundColor Green
} catch {
    Write-Host "❌ MCP Gateway not responding on port 8080" -ForegroundColor Red
}

# Test 7: Memory Service
Write-Host "`n[7/10] Testing Memory Service..." -ForegroundColor Yellow
try {
    $memTest = Invoke-RestMethod -Uri "http://${SERVER_IP}:8002/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Host "✅ Memory Service is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Memory Service not responding on port 8002" -ForegroundColor Red
}

# Test 8: Subagent Manager
Write-Host "`n[8/10] Testing Subagent Manager..." -ForegroundColor Yellow
try {
    $subTest = Invoke-RestMethod -Uri "http://${SERVER_IP}:8001/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Host "✅ Subagent Manager is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Subagent Manager not responding on port 8001" -ForegroundColor Red
}

# Test 9: Dashboard
Write-Host "`n[9/10] Testing Dashboard..." -ForegroundColor Yellow
try {
    $dashTest = Invoke-WebRequest -Uri "http://${SERVER_IP}:3000" -TimeoutSec 5 -UseBasicParsing -ErrorAction SilentlyContinue
    if ($dashTest.StatusCode -eq 200) {
        Write-Host "✅ Dashboard is accessible" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Dashboard not responding on port 3000" -ForegroundColor Red
}

# Test 10: Check service logs for errors
Write-Host "`n[10/10] Checking recent service logs..." -ForegroundColor Yellow
$logsCheck = ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker logs orchestrator --tail 20 2>&1 | grep -i 'error\|exception\|failed' || echo 'No recent errors'" 2>&1
if ($logsCheck -match "No recent errors") {
    Write-Host "✅ No recent errors in orchestrator logs" -ForegroundColor Green
} else {
    Write-Host "⚠️  Recent errors found:" -ForegroundColor Yellow
    Write-Host $logsCheck -ForegroundColor White
}

# Summary
Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "DIAGNOSIS SUMMARY" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

Write-Host "Run these commands to investigate further:`n" -ForegroundColor Yellow

Write-Host "1. Check all service logs:" -ForegroundColor White
Write-Host "   ssh -i $SSH_KEY ubuntu@$SERVER_IP 'docker logs orchestrator'" -ForegroundColor Gray

Write-Host "`n2. Restart services:" -ForegroundColor White
Write-Host "   ssh -i $SSH_KEY ubuntu@$SERVER_IP 'cd king-ai-v3/agentic-framework-main && docker-compose restart'" -ForegroundColor Gray

Write-Host "`n3. Check Ollama model:" -ForegroundColor White
Write-Host "   ssh -i $SSH_KEY ubuntu@$SERVER_IP 'docker exec ollama ollama list'" -ForegroundColor Gray

Write-Host "`n4. View real-time logs:" -ForegroundColor White
Write-Host "   ssh -i $SSH_KEY ubuntu@$SERVER_IP 'docker logs -f orchestrator'" -ForegroundColor Gray

Write-Host "`n5. Full system restart:" -ForegroundColor White
Write-Host "   ssh -i $SSH_KEY ubuntu@$SERVER_IP 'cd king-ai-v3/agentic-framework-main && docker-compose down && docker-compose up -d'" -ForegroundColor Gray

Write-Host "`n================================`n" -ForegroundColor Cyan
