
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

# 1. Sync to GitHub (Best Effort)
Write-Host "`n[1/3] Syncing Local Changes..." -ForegroundColor Yellow
git add .
$commitMsg = Read-Host "Enter Commit Message (Default: 'Auto-deploy update')"
if ([string]::IsNullOrWhiteSpace($commitMsg)) { $commitMsg = "Auto-deploy update" }
git commit -m "$commitMsg"

# Try pushing if remote exists
$hasRemote = git remote -v
if ($hasRemote) {
    git push
    if ($LASTEXITCODE -ne 0) { Write-Host "Warning: Git Push Failed." -ForegroundColor Red }
} else {
    Write-Host "Note: No Git remote configured. Skipping push." -ForegroundColor Gray
}

# 2. Deploy to AWS (SCP)
Write-Host "`n[2/3] Deploying to AWS..." -ForegroundColor Yellow

# Create Archive
tar -czf king-ai-v2.tar.gz --exclude="node_modules" --exclude="venv" --exclude=".git" --exclude=".pytest_cache" src dashboard scripts config tests alembic.ini pyproject.toml README.md docker-compose.yml Dockerfile .env

# Upload
scp -o StrictHostKeyChecking=no -i $PEM_FILE king-ai-v2.tar.gz ubuntu@${AWS_IP}:/home/ubuntu/king-ai-v2.tar.gz

# Extract & Restart
$RemoteCommand = "mkdir -p king-ai-v2 && tar -xzf king-ai-v2.tar.gz -C king-ai-v2 && cd king-ai-v2 && chmod +x start.sh && ./start.sh"
ssh -o StrictHostKeyChecking=no -i $PEM_FILE ubuntu@$AWS_IP $RemoteCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Deployment Successful!" -ForegroundColor Green
    Write-Host "   Backend API: http://$AWS_IP:8000" -ForegroundColor Gray
    Write-Host "   Dashboard:   http://$AWS_IP:5173" -ForegroundColor Gray
} else {
    Write-Host "`n❌ Remote Deployment Failed." -ForegroundColor Red
}

# Cleanup
Remove-Item king-ai-v2.tar.gz -ErrorAction SilentlyContinue
