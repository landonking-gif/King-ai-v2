# Fix Deployment Connection Issues

## Problem
SSH connection to AWS instance `52.90.242.99` is timing out.

## Solutions (Try in order)

### 1. **Fix AWS Security Group (Most Common Issue)**
Your AWS instance needs to allow SSH from your IP:

1. Go to AWS Console → EC2 → Instances
2. Select your instance (`52.90.242.99`)
3. Click **Security** tab → Click the security group link
4. Click **Edit inbound rules**
5. Add/Update SSH rule:
   - Type: SSH
   - Protocol: TCP
   - Port: 22
   - Source: **My IP** (or `0.0.0.0/0` for testing)
6. Click **Save rules**
7. Try deployment again

### 2. **Add SSH Key to WSL**
The script needs your AWS `.pem` key file:

```bash
# In PowerShell, copy your key to WSL
wsl bash -c "mkdir -p ~/.ssh"
wsl bash -c "cp /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/YOUR-KEY-FILE.pem ~/.ssh/king-ai-studio.pem"
wsl bash -c "chmod 600 ~/.ssh/king-ai-studio.pem"
```

**Replace `YOUR-KEY-FILE.pem`** with your actual key file name!

### 3. **Verify Instance is Running**
1. Go to AWS Console → EC2 → Instances
2. Check instance state is **Running**
3. Verify the **Public IPv4 address** matches `52.90.242.99`
4. If different, use the correct IP

### 4. **Test SSH Connection Directly**
```bash
wsl bash -c "ssh -i ~/.ssh/king-ai-studio.pem ubuntu@52.90.242.99 'echo Connection successful'"
```

If this works, retry the deployment!

### 5. **Use Alternative Deployment Method**
If SSH continues to fail, deploy manually:

```bash
# SSH into your instance
wsl bash -c "ssh -i ~/.ssh/king-ai-studio.pem ubuntu@52.90.242.99"

# Then run these commands on the server:
cd ~
git clone YOUR-REPO-URL agentic-framework-main
cd agentic-framework-main
bash scripts/setup_aws.sh
```

## Quick Fix Script
Run this to automatically check common issues:

```powershell
# Save as check-deployment.ps1
$ip = "52.90.242.99"

Write-Host "Checking deployment readiness..." -ForegroundColor Cyan

# Check if key exists
$keyExists = wsl bash -c "test -f ~/.ssh/king-ai-studio.pem && echo 'yes' || echo 'no'"
if ($keyExists -eq "no") {
    Write-Host "❌ SSH key not found in WSL" -ForegroundColor Red
    Write-Host "   Run: wsl bash -c 'ls /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/*.pem'" -ForegroundColor Yellow
} else {
    Write-Host "✓ SSH key found" -ForegroundColor Green
}

# Test SSH connection
Write-Host "`nTesting SSH connection to $ip..." -ForegroundColor Cyan
wsl bash -c "timeout 5 ssh -i ~/.ssh/king-ai-studio.pem -o StrictHostKeyChecking=no ubuntu@$ip 'echo Connected' 2>&1"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ SSH connection successful!" -ForegroundColor Green
    Write-Host "`nYou can now run the deployment." -ForegroundColor Green
} else {
    Write-Host "❌ SSH connection failed" -ForegroundColor Red
    Write-Host "`nPossible issues:" -ForegroundColor Yellow
    Write-Host "1. Security group doesn't allow SSH from your IP"
    Write-Host "2. Instance is not running"
    Write-Host "3. Wrong IP address"
    Write-Host "4. Wrong SSH key"
}
```

Run: `.\check-deployment.ps1`
