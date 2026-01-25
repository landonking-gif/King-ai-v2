#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test the fixed Memory Service deployment
.DESCRIPTION
    Tests if the Memory Service starts correctly after the fix
#>

$SERVER_IP = "54.224.134.220"
$SSH_KEY = "king-ai-v3\agentic-framework-main\king-ai-studio.pem"

Write-Host "`n🔧 Testing Memory Service Fix" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Test 1: Check if Memory Service is running
Write-Host "[1/3] Checking Memory Service status..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://${SERVER_IP}:8002/health" -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
    Write-Host "✅ Memory Service is running!" -ForegroundColor Green
    Write-Host "   Status: $($response.StatusCode)" -ForegroundColor White
} catch {
    Write-Host "❌ Memory Service not responding" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Test 2: Test chat endpoint
Write-Host "`n[2/3] Testing AI chat..." -ForegroundColor Yellow
try {
    $chatBody = @{
        text = "Hello, are you working?"
        session_id = "test_$(Get-Date -Format 'yyyyMMddHHmmss')"
    } | ConvertTo-Json

    $chatResponse = Invoke-WebRequest -Uri "http://${SERVER_IP}:8000/api/chat/message" `
        -Method Post `
        -Body $chatBody `
        -ContentType "application/json" `
        -TimeoutSec 30 `
        -UseBasicParsing `
        -ErrorAction Stop

    Write-Host "✅ Chat endpoint working!" -ForegroundColor Green
    Write-Host "   Response received successfully" -ForegroundColor White
} catch {
    Write-Host "❌ Chat endpoint failed" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Test 3: Check all services
Write-Host "`n[3/3] Checking all services..." -ForegroundColor Yellow
$services = @{
    "Orchestrator" = "http://${SERVER_IP}:8000/api/health"
    "Memory Service" = "http://${SERVER_IP}:8002/health"
    "MCP Gateway" = "http://${SERVER_IP}:8080/health"
    "Subagent Manager" = "http://${SERVER_IP}:8001/health"
    "Dashboard" = "http://${SERVER_IP}:3000"
}

$working = 0
foreach ($service in $services.GetEnumerator()) {
    try {
        $response = Invoke-WebRequest -Uri $service.Value -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Host "  ✅ $($service.Key)" -ForegroundColor Green
        $working++
    } catch {
        Write-Host "  ❌ $($service.Key)" -ForegroundColor Red
    }
}

Write-Host "`n================================" -ForegroundColor Cyan
if ($working -eq $services.Count) {
    Write-Host "🎉 ALL SYSTEMS OPERATIONAL!" -ForegroundColor Green
    Write-Host "Your King AI is ready at: http://${SERVER_IP}:3000" -ForegroundColor Cyan
} else {
    Write-Host "⚠️  $working/$($services.Count) services working" -ForegroundColor Yellow
    Write-Host "Run the deployment script again if needed:" -ForegroundColor White
    Write-Host "wsl bash -c 'cd /mnt/c/Users/... && sed -i s/\r$// run_service.sh && echo 54.224.134.220 | bash run_service.sh'" -ForegroundColor Gray
}
Write-Host "================================`n" -ForegroundColor Cyan