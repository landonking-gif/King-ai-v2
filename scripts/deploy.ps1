
$AWS_IP = Read-Host "Enter AWS IP Address (Default: ec2-13-222-9-32.compute-1.amazonaws.com)"
if ([string]::IsNullOrWhiteSpace($AWS_IP)) { $AWS_IP = "ec2-13-222-9-32.compute-1.amazonaws.com" }

$PEM_FILE = "king-ai-studio.pem"

Write-Host "`n🚀 Starting Automated Deployment to $AWS_IP..." -ForegroundColor Cyan

# 0. Check for Git
if (-not (Test-Path ".git")) {
    Write-Host "Warning: .git directory not found. Initializing git..." -ForegroundColor Yellow
    git init
    git add .
    git commit -m "Initial commit"
}

# 1. Sync to GitHub
Write-Host "`n[1/3] Syncing Local Changes to GitHub..." -ForegroundColor Yellow
git add .
$commitMsg = Read-Host "Enter Commit Message (Default: 'Auto-deploy update')"
if ([string]::IsNullOrWhiteSpace($commitMsg)) { $commitMsg = "Auto-deploy update" }
git commit -m "$commitMsg"
git push
if ($LASTEXITCODE -ne 0) {
    Write-Host "Git Push Failed! Aborting." -ForegroundColor Red
    exit
}

# 2. Trigger Remote Update
Write-Host "`n[2/3] Triggering Remote Update on AWS..." -ForegroundColor Yellow
$RemoteCommand = "cd king-ai-v2 && git pull && ./start.sh"

ssh -o StrictHostKeyChecking=no -i $PEM_FILE ubuntu@$AWS_IP $RemoteCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Deployment Successful!" -ForegroundColor Green
    Write-Host "   Backend API: http://$AWS_IP:8000" -ForegroundColor Gray
    Write-Host "   Dashboard:   http://$AWS_IP:5173" -ForegroundColor Gray
} else {
    Write-Host "`n❌ Remote Deployment Failed. Check SSH connection." -ForegroundColor Red
}
