# AWS Instance Restart and Redeploy Script
# Handles IP changes automatically

param(
    [string]$InstanceId = "i-1234567890abcdef0",  # Replace with your actual instance ID
    [string]$Region = "us-east-1"
)

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "AWS Instance Restart & Redeploy" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if AWS CLI is available
if (!(Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "AWS CLI not found. Please install it from: https://aws.amazon.com/cli/" -ForegroundColor Red
    Write-Host "Or follow manual steps below." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Manual Steps:" -ForegroundColor White
    Write-Host "1. Go to AWS Console → EC2 → Instances" -ForegroundColor White
    Write-Host "2. Select your instance" -ForegroundColor White
    Write-Host "3. Actions → Instance State → Stop" -ForegroundColor White
    Write-Host "4. Wait for 'stopped' status" -ForegroundColor White
    Write-Host "5. Actions → Instance State → Start" -ForegroundColor White
    Write-Host "6. Note the new public IP address" -ForegroundColor White
    Write-Host "7. Update USER_GUIDE.md with new IP" -ForegroundColor White
    Write-Host "8. Run: echo 'NEW_IP' | bash run_service.sh" -ForegroundColor White
    exit 1
}

Write-Host "Stopping instance $InstanceId..." -ForegroundColor Yellow
aws ec2 stop-instances --instance-ids $InstanceId --region $Region

Write-Host "Waiting for instance to stop..." -ForegroundColor Yellow
aws ec2 wait instance-stopped --instance-ids $InstanceId --region $Region
Write-Host "✓ Instance stopped" -ForegroundColor Green

Write-Host ""
Write-Host "Starting instance $InstanceId..." -ForegroundColor Yellow
aws ec2 start-instances --instance-ids $InstanceId --region $Region

Write-Host "Waiting for instance to start..." -ForegroundColor Yellow
aws ec2 wait instance-running --instance-ids $InstanceId --region $Region
Write-Host "✓ Instance started" -ForegroundColor Green

Start-Sleep -Seconds 10  # Wait for networking

Write-Host ""
Write-Host "Getting new public IP..." -ForegroundColor Yellow
$instanceInfo = aws ec2 describe-instances --instance-ids $InstanceId --region $Region --query 'Reservations[0].Instances[0].[PublicIpAddress, State.Name]' --output text

$ip, $state = $instanceInfo -split '\t'

Write-Host "New IP Address: $ip" -ForegroundColor Green
Write-Host "Instance State: $state" -ForegroundColor Green

Write-Host ""
Write-Host "Updating configuration files..." -ForegroundColor Yellow

# Update USER_GUIDE.md
$userGuidePath = "C:\Users\dmilner.AGV-040318-PC\Downloads\landon\king-ai-v2\USER_GUIDE.md"
$content = Get-Content $userGuidePath -Raw
$content = $content -replace '54\.224\.134\.220', $ip
Set-Content -Path $userGuidePath -Value $content
Write-Host "✓ Updated USER_GUIDE.md" -ForegroundColor Green

Write-Host ""
Write-Host "Redeploying services..." -ForegroundColor Yellow
Write-Host "Running: echo '$ip' | bash run_service.sh" -ForegroundColor White

# Run the deployment
$deploymentScript = "C:\Users\dmilner.AGV-040318-PC\Downloads\landon\king-ai-v2\king-ai-v3\agentic-framework-main\orchestrator\run_service.sh"
$deploymentDir = Split-Path $deploymentScript -Parent

# Use WSL to run the deployment
wsl bash -c "cd /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/orchestrator && sed -i 's/\r$//' run_service.sh && echo '$ip' | bash run_service.sh"

Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Restart & Redeploy Complete!" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "New URLs:" -ForegroundColor Green
Write-Host "  Dashboard: http://$ip" -ForegroundColor White
Write-Host "  API: http://$ip/api" -ForegroundColor White
Write-Host "  Health: http://$ip/api/health" -ForegroundColor White
Write-Host ""
Write-Host "Instance ID: $InstanceId" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White