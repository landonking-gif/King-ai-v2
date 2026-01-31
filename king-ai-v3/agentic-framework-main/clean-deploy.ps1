#!/usr/bin/env pwsh
# Clean deployment using SCP and SSH

$SERVER = "52.90.206.76"
$USER = "ubuntu"
$KEY = "king-ai-studio.pem"

Write-Host "=== Step 1: Clean server ===" -ForegroundColor Cyan
ssh -i $KEY "$USER@$SERVER" @"
sudo rm -rf /home/ubuntu/agentic-framework-main
mkdir -p /home/ubuntu/agentic-framework-main
"@

Write-Host "`n=== Step 2: Copy files with SCP ===" -ForegroundColor Cyan
scp -i $KEY -r -C "agentic-framework-main" "$USER@${SERVER}:/home/ubuntu/"
scp -i $KEY -r -C "..\dashboard" "$USER@${SERVER}:/home/ubuntu/"
scp -i $KEY "enhanced-deploy.sh" "$USER@${SERVER}:/home/ubuntu/"

Write-Host "`n=== Step 3: Fix permissions and run deployment ===" -ForegroundColor Cyan
ssh -i $KEY "$USER@$SERVER" @"
chmod +x /home/ubuntu/enhanced-deploy.sh
dos2unix /home/ubuntu/enhanced-deploy.sh 2>/dev/null || sed -i 's/\r$//' /home/ubuntu/enhanced-deploy.sh
cd /home/ubuntu
./enhanced-deploy.sh
"@

Write-Host "`n=== Deployment complete ===" -ForegroundColor Green
