#!/usr/bin/env pwsh
#Clean deployment with local tar creation

$SERVER = "52.90.206.76"
$USER = "ubuntu"
$KEY = "king-ai-studio.pem"

Write-Host "=== Creating tar archive ===" -ForegroundColor Cyan
cd ..
tar -czf king-ai-deploy.tar.gz "agentic-framework-main" "dashboard"

if (-not (Test-Path "king-ai-deploy.tar.gz")) {
    Write-Host "Failed to create tar archive" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Uploading tar archive ===" -ForegroundColor Cyan
scp -i "agentic-framework-main\$KEY" "king-ai-deploy.tar.gz" "$USER@${SERVER}:/home/ubuntu/"

Write-Host "`n=== Extracting and deploying on server ===" -ForegroundColor Cyan
ssh -i "agentic-framework-main\$KEY" "$USER@$SERVER" @"
cd /home/ubuntu
echo '=== Cleaning old files ==='
rm -rf agentic-framework-main dashboard
rm -f agentic-framework-main\\* dashboard\\*

echo '=== Extracting tar.gz ==='
tar -xzf king-ai-deploy.tar.gz
ls -la | head -20

echo '=== Uploading deployment script ==='
"@

scp -i "agentic-framework-main\$KEY" "agentic-framework-main\enhanced-deploy.sh" "$USER@${SERVER}:/home/ubuntu/"

ssh -i "agentic-framework-main\$KEY" "$USER@$SERVER" @"
chmod +x enhanced-deploy.sh
dos2unix enhanced-deploy.sh 2>/dev/null || sed -i 's/\r$//' enhanced-deploy.sh
./enhanced-deploy.sh
"@

Write-Host "`n=== Deployment complete ===" -ForegroundColor Green
