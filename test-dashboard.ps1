#!/usr/bin/env pwsh

$IP = "52.90.242.99"
$KeyPath = "C:\Users\dmilner.AGV-040318-PC\.ssh\king-ai-studio.pem"

Write-Host "=== Dashboard Test ===" -ForegroundColor Cyan

# Check serve process
Write-Host "`n1. Checking serve process..." -ForegroundColor Yellow
ssh -i $KeyPath ubuntu@$IP "ps aux | grep 'serve --listen' | grep -v grep"

# Check port 3000
Write-Host "`n2. Checking port 3000..." -ForegroundColor Yellow  
ssh -i $KeyPath ubuntu@$IP "ss -tuln | grep 3000 || netstat -tuln | grep 3000 || echo 'Port not listening'"

# Test localhost:3000
Write-Host "`n3. Testing localhost:3000..." -ForegroundColor Yellow
ssh -i $KeyPath ubuntu@$IP "curl -s -o /dev/null -w 'HTTP Status: %{http_code}' http://localhost:3000/ ; echo ''"

# Check nginx
Write-Host "`n4. Checking nginx..." -ForegroundColor Yellow
ssh -i $KeyPath ubuntu@$IP "systemctl is-active nginx"

# Test external access
Write-Host "`n5. Testing external access..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://$IP/" -TimeoutSec 5 -UseBasicParsing
    Write-Host "HTTP Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Content Length: $($response.Content.Length) bytes"
} catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan
