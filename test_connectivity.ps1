# Simple connectivity test
param(
    [string]$IP = "54.224.134.220"
)

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "AWS Connectivity Test" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Testing connection to $IP..." -ForegroundColor Yellow

# Test basic connectivity
$pingResult = Test-Connection -ComputerName $IP -Count 1 -Quiet
if ($pingResult) {
    Write-Host "✓ Ping successful" -ForegroundColor Green
} else {
    Write-Host "✗ Ping failed - instance may be stopped" -ForegroundColor Red
}

# Test SSH port
$sshTest = Test-NetConnection -ComputerName $IP -Port 22 -WarningAction SilentlyContinue
if ($sshTest.TcpTestSucceeded) {
    Write-Host "✓ SSH port (22) is open" -ForegroundColor Green
} else {
    Write-Host "✗ SSH port (22) is closed - check security group" -ForegroundColor Red
}

# Test HTTP port
$httpTest = Test-NetConnection -ComputerName $IP -Port 8000 -WarningAction SilentlyContinue
if ($httpTest.TcpTestSucceeded) {
    Write-Host "✓ HTTP port (8000) is open" -ForegroundColor Green
} else {
    Write-Host "✗ HTTP port (8000) is closed - check security group" -ForegroundColor Red
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Check AWS Console: https://console.aws.amazon.com/ec2/" -ForegroundColor White
Write-Host "2. Verify instance is running" -ForegroundColor White
Write-Host "3. Check Security Group rules" -ForegroundColor White
Write-Host "4. If IP changed, update deployment scripts" -ForegroundColor White