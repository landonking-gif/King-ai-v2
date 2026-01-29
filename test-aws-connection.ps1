# AWS Deployment Connection Tester
# Tests connectivity to new AWS instance before deployment

param(
    [string]$IP = "52.90.242.99"
)

Write-Host "================================" -ForegroundColor Cyan
Write-Host "AWS Connection Diagnostic Tool" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Check if SSH key exists in WSL
Write-Host "[1/4] Checking SSH key..." -ForegroundColor Yellow
$keyCheck = wsl bash -c "test -f ~/.ssh/king-ai-studio.pem && echo 'found' || echo 'missing'"
if ($keyCheck -match "found") {
    Write-Host "  ✓ SSH key exists at ~/.ssh/king-ai-studio.pem" -ForegroundColor Green
    
    # Check permissions
    $perms = wsl bash -c "stat -c '%a' ~/.ssh/king-ai-studio.pem"
    if ($perms -eq "600") {
        Write-Host "  ✓ Key permissions are correct (600)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Key permissions are $perms (should be 600)" -ForegroundColor Yellow
        Write-Host "  Fixing permissions..." -ForegroundColor Yellow
        wsl bash -c "chmod 600 ~/.ssh/king-ai-studio.pem"
        Write-Host "  ✓ Permissions fixed" -ForegroundColor Green
    }
} else {
    Write-Host "  ❌ SSH key NOT found" -ForegroundColor Red
    Write-Host ""
    Write-Host "  To fix:" -ForegroundColor Yellow
    Write-Host "  1. Find your .pem file in Downloads:" -ForegroundColor White
    Write-Host "     wsl bash -c 'ls /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/*.pem'" -ForegroundColor Gray
    Write-Host "  2. Copy it to WSL:" -ForegroundColor White
    Write-Host "     wsl bash -c 'cp /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/YOUR-KEY.pem ~/.ssh/king-ai-studio.pem'" -ForegroundColor Gray
    Write-Host "  3. Set permissions:" -ForegroundColor White
    Write-Host "     wsl bash -c 'chmod 600 ~/.ssh/king-ai-studio.pem'" -ForegroundColor Gray
    exit 1
}

Write-Host ""

# Test 2: Check if instance is reachable (ping)
Write-Host "[2/4] Testing network connectivity to $IP..." -ForegroundColor Yellow
$pingResult = Test-Connection -ComputerName $IP -Count 2 -Quiet -ErrorAction SilentlyContinue
if ($pingResult) {
    Write-Host "  ✓ Instance is reachable (ping successful)" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Instance not responding to ping (this is normal if ICMP is blocked)" -ForegroundColor Yellow
}

Write-Host ""

# Test 3: Check if SSH port is open
Write-Host "[3/4] Testing SSH port (22)..." -ForegroundColor Yellow
$tcpTest = Test-NetConnection -ComputerName $IP -Port 22 -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
if ($tcpTest.TcpTestSucceeded) {
    Write-Host "  ✓ SSH port 22 is open" -ForegroundColor Green
} else {
    Write-Host "  ❌ SSH port 22 is CLOSED or BLOCKED" -ForegroundColor Red
    Write-Host ""
    Write-Host "  This is the problem! Fix AWS Security Group:" -ForegroundColor Yellow
    Write-Host "  1. Go to AWS Console -> EC2 -> Instances" -ForegroundColor White
    Write-Host "  2. Select your instance (IP: $IP)" -ForegroundColor White
    Write-Host "  3. Click Security tab -> Click security group name" -ForegroundColor White
    Write-Host "  4. Click Edit inbound rules" -ForegroundColor White
    Write-Host "  5. Add rule: Type=SSH, Port=22, Source=My IP" -ForegroundColor White
    Write-Host "  6. Save rules and try again" -ForegroundColor White
    Write-Host ""
    Write-Host "  Your current public IP: " -NoNewline -ForegroundColor White
    try {
        $myIP = (Invoke-WebRequest -Uri "https://api.ipify.org" -UseBasicParsing -TimeoutSec 5).Content
        Write-Host "$myIP" -ForegroundColor Cyan
    } catch {
        Write-Host "Unable to detect" -ForegroundColor Gray
    }
    exit 1
}

Write-Host ""

# Test 4: Try SSH connection
Write-Host "[4/4] Testing SSH authentication..." -ForegroundColor Yellow
$sshTest = wsl bash -c "timeout 10 ssh -i ~/.ssh/king-ai-studio.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$IP echo SUCCESS 2>&1"
if ($sshTest -match "SUCCESS") {
    Write-Host "  ✓ SSH authentication successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "ALL CHECKS PASSED!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run the deployment command from USER_GUIDE.md" -ForegroundColor White
} else {
    Write-Host "  ❌ SSH authentication failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Error details:" -ForegroundColor Yellow
    Write-Host $sshTest -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Possible causes:" -ForegroundColor Yellow
    Write-Host "  1. Wrong SSH key (not matching the instance key pair)" -ForegroundColor White
    Write-Host "  2. Instance not fully booted yet (wait 2-3 minutes)" -ForegroundColor White
    Write-Host "  3. Username is not ubuntu (try ec2-user or admin)" -ForegroundColor White
    exit 1
}
