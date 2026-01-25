#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Fix King AI Memory Service - Complete repair
.DESCRIPTION
    Starts the Memory Service and verifies all services are working
#>

$ErrorActionPreference = "Continue"
$SERVER_IP = "3.236.144.91"
$SSH_KEY = "king-ai-v3\agentic-framework-main\king-ai-studio.pem"

Write-Host "`n🔧 KING AI MEMORY SERVICE FIX" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Step 1: Upload fix script
Write-Host "[1/5] Uploading fix script..." -ForegroundColor Yellow
try {
    wsl bash -c "scp -o StrictHostKeyChecking=no -i '/mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/king-ai-studio.pem' '/mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/fix-memory-service.sh' ubuntu@3.236.144.91:~/"
    Write-Host "✅ Script uploaded" -ForegroundColor Green
} catch {
    Write-Host "❌ Upload failed: $_" -ForegroundColor Red
}

# Step 2: Make executable and run
Write-Host "`n[2/5] Running fix script on server..." -ForegroundColor Yellow
$fixOutput = wsl bash -c "ssh -o StrictHostKeyChecking=no -i '/mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/king-ai-studio.pem' ubuntu@3.236.144.91 'chmod +x ~/fix-memory-service.sh && bash ~/fix-memory-service.sh 2>&1'"
Write-Host $fixOutput

# Step 3: Wait for service to start
Write-Host "`n[3/5] Waiting for Memory Service to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

# Step 4: Test health endpoints
Write-Host "`n[4/5] Testing all service endpoints..." -ForegroundColor Yellow

$endpoints = @{
    "Memory Service" = "http://${SERVER_IP}:8002/health"
    "Orchestrator" = "http://${SERVER_IP}:8000/api/health"
}

$allHealthy = $true
foreach ($service in $endpoints.GetEnumerator()) {
    try {
        $response = Invoke-RestMethod -Uri $service.Value -TimeoutSec 5 -ErrorAction Stop
        Write-Host "  ✅ $($service.Key) - OK" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ $($service.Key) - FAILED" -ForegroundColor Red
        $allHealthy = $false
    }
}

# Step 5: Test chat endpoint
Write-Host "`n[5/5] Testing AI chat endpoint..." -ForegroundColor Yellow
try {
    $chatBody = @{
        text = "Hello"
        session_id = "health_check_$(Get-Date -Format 'yyyyMMddHHmmss')"
    } | ConvertTo-Json

    $chatResponse = Invoke-RestMethod -Uri "http://${SERVER_IP}:8000/api/chat/message" `
        -Method Post `
        -Body $chatBody `
        -ContentType "application/json" `
        -TimeoutSec 30 `
        -ErrorAction Stop

    if ($chatResponse.response) {
        Write-Host "  ✅ Chat is working!" -ForegroundColor Green
        Write-Host "  AI Response: $($chatResponse.response.Substring(0, [Math]::Min(100, $chatResponse.response.Length)))..." -ForegroundColor White
    } else {
        Write-Host "  ⚠️  Chat responded but format unexpected" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ❌ Chat endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
    $allHealthy = $false
}

# Final Status
Write-Host "`n================================" -ForegroundColor Cyan
if ($allHealthy) {
    Write-Host "✅ ALL SYSTEMS OPERATIONAL" -ForegroundColor Green
    Write-Host "`nYour King AI is ready at: http://${SERVER_IP}:3000" -ForegroundColor Cyan
} else {
    Write-Host "⚠️  SOME SERVICES STILL HAVE ISSUES" -ForegroundColor Yellow
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Check logs on server: ssh ubuntu@$SERVER_IP 'tail -50 /tmp/memory-service.log'" -ForegroundColor White
    Write-Host "2. Try manual restart: ssh ubuntu@$SERVER_IP 'cd agentic-framework-main && .venv/bin/python -m memory-service.service.main'" -ForegroundColor White
}
Write-Host "================================`n" -ForegroundColor Cyan
