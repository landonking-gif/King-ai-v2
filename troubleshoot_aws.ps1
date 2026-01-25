# AWS Deployment Troubleshooting Guide
# Run this when deployment fails

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "AWS Deployment Troubleshooting" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

$IP = "54.224.134.220"

Write-Host "Step 1: Testing basic connectivity..." -ForegroundColor Yellow
$ping = Test-Connection -ComputerName $IP -Count 1 -Quiet
if ($ping) {
    Write-Host "✓ Instance is reachable (ping works)" -ForegroundColor Green
} else {
    Write-Host "✗ Instance is not reachable - check if it's running" -ForegroundColor Red
    Write-Host "  → Go to AWS Console → EC2 → Check instance state" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "Step 2: Testing SSH port..." -ForegroundColor Yellow
$ssh = Test-NetConnection -ComputerName $IP -Port 22 -WarningAction SilentlyContinue
if ($ssh.TcpTestSucceeded) {
    Write-Host "✓ SSH port (22) is open" -ForegroundColor Green
} else {
    Write-Host "✗ SSH port (22) is blocked - check Security Group" -ForegroundColor Red
    Write-Host "  → EC2 → Security Groups → Allow SSH (22) from your IP" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "Step 3: Testing HTTP port..." -ForegroundColor Yellow
$http = Test-NetConnection -ComputerName $IP -Port 8000 -WarningAction SilentlyContinue
if ($http.TcpTestSucceeded) {
    Write-Host "✓ HTTP port (8000) is open" -ForegroundColor Green
} else {
    Write-Host "✗ HTTP port (8000) is blocked - check Security Group" -ForegroundColor Red
    Write-Host "  → EC2 → Security Groups → Allow Custom TCP 8000 from 0.0.0.0/0" -ForegroundColor White
}

Write-Host ""
Write-Host "Step 4: Testing SSH authentication..." -ForegroundColor Yellow
Write-Host "If SSH port is open but connection times out, possible issues:" -ForegroundColor White
Write-Host "  → SSH key permissions (should be 600)" -ForegroundColor White
Write-Host "  → Wrong SSH key" -ForegroundColor White
Write-Host "  → Instance was rebuilt/replaced" -ForegroundColor White
Write-Host "  → IP address changed" -ForegroundColor White

Write-Host ""
Write-Host "Quick fix commands:" -ForegroundColor Cyan
Write-Host "1. Fix SSH key permissions:" -ForegroundColor White
Write-Host "   wsl bash -c 'chmod 600 /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/king-ai-studio.pem'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Check current instance IP in AWS Console" -ForegroundColor White
Write-Host "3. Update deployment scripts with new IP if changed" -ForegroundColor White