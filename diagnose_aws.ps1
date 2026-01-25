# AWS Instance Diagnostic Script
# Run this to check your EC2 instance status

echo "=================================="
echo "AWS Instance Diagnostic"
echo "=================================="
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Please install it first:"
    echo "https://aws.amazon.com/cli/"
    echo ""
    echo "Manual steps:"
    echo "1. Go to AWS Console: https://console.aws.amazon.com/ec2/"
    echo "2. Check if instance with IP 54.224.134.220 is running"
    echo "3. Verify Security Group allows SSH (port 22) and HTTP (port 8000)"
    echo "4. Check if the instance was stopped/restarted (IP may have changed)"
    exit 1
fi

echo "Checking AWS EC2 instances..."
echo ""

# List instances (you may need to configure AWS CLI first)
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress,Tags[?Key==`Name`].Value|[0]]' --output table

echo ""
echo "If your instance is stopped, start it with:"
echo "aws ec2 start-instances --instance-ids YOUR_INSTANCE_ID"
echo ""
echo "If the IP changed, update your deployment scripts."