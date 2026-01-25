# Check AWS Instance Status
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "AWS Instance Status Check" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

$IP = "54.224.134.220"

Write-Host "Current configured IP: $IP" -ForegroundColor White
Write-Host ""

Write-Host "Connectivity Status:" -ForegroundColor Yellow
$ping = Test-Connection -ComputerName $IP -Count 1 -Quiet
if ($ping) {
    Write-Host "✓ Ping: SUCCESS" -ForegroundColor Green
} else {
    Write-Host "✗ Ping: FAILED - Instance may be stopped" -ForegroundColor Red
}

$ssh = Test-NetConnection -ComputerName $IP -Port 22 -WarningAction SilentlyContinue
if ($ssh.TcpTestSucceeded) {
    Write-Host "✓ SSH Port (22): OPEN" -ForegroundColor Green
} else {
    Write-Host "✗ SSH Port (22): CLOSED" -ForegroundColor Red
}

Write-Host ""
Write-Host "Service Status:" -ForegroundColor Yellow

# Test Orchestrator
try {
    $response = Invoke-WebRequest -Uri "http://$IP`:8000/health" -TimeoutSec 5
    $health = $response.Content | ConvertFrom-Json
    Write-Host "✓ Orchestrator (8000): $($health.status.ToUpper())" -ForegroundColor $(if ($health.status -eq "healthy") { "Green" } else { "Yellow" })
} catch {
    Write-Host "✗ Orchestrator (8000): DOWN" -ForegroundColor Red
}

# Test MCP Gateway
try {
    $response = Invoke-WebRequest -Uri "http://$IP`:8080/health" -TimeoutSec 5
    $health = $response.Content | ConvertFrom-Json
    Write-Host "✓ MCP Gateway (8080): $($health.status.ToUpper())" -ForegroundColor Green
} catch {
    Write-Host "✗ MCP Gateway (8080): DOWN" -ForegroundColor Red
}

# Test Memory Service
$memTest = Test-NetConnection -ComputerName $IP -Port 8002 -WarningAction SilentlyContinue
if ($memTest.TcpTestSucceeded) {
    Write-Host "✓ Memory Service (8002): ACCESSIBLE" -ForegroundColor Green
} else {
    Write-Host "✗ Memory Service (8002): DOWN" -ForegroundColor Red
}

# Test Subagent Manager
$subTest = Test-NetConnection -ComputerName $IP -Port 8081 -WarningAction SilentlyContinue
if ($subTest.TcpTestSucceeded) {
    Write-Host "✓ Subagent Manager (8081): ACCESSIBLE" -ForegroundColor Green
} else {
    Write-Host "✗ Subagent Manager (8081): DOWN" -ForegroundColor Red
}

Write-Host ""
Write-Host "Possible Issues & Solutions:" -ForegroundColor Cyan
Write-Host "1. If SSH fails but services respond: SSH key or permissions issue" -ForegroundColor White
Write-Host "2. If all services down: Instance may have restarted with new IP" -ForegroundColor White
Write-Host "3. If some services down: Need to restart individual services" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "- Check AWS Console for current instance IP" -ForegroundColor White
Write-Host "- Verify Security Group allows all required ports" -ForegroundColor White
Write-Host "- If IP changed, update deployment scripts" -ForegroundColor White