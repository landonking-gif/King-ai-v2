# King AI v2 - Infrastructure as Code

This directory contains Terraform configurations for deploying King AI v2 to AWS.

## Architecture Overview

The infrastructure includes:
- **Multi-AZ VPC** with public and private subnets across 3 availability zones
- **Auto-Scaling GPU Cluster** (p5.48xlarge instances) for LLM inference with vLLM
- **RDS PostgreSQL** (db.r6g.xlarge) with Multi-AZ for high availability
- **ElastiCache Redis** (cache.r6g.large) for caching and task queuing
- **Application Load Balancer** for distributing traffic
- **SQS Queue** for inference request queuing
- **Auto-scaling** based on SQS queue depth (scales 2-8 instances when queue > 100)

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **Terraform** >= 1.0 installed
3. **AWS CLI** configured with credentials
4. **S3 Bucket** for Terraform state storage (update `main.tf` backend config)

## Configuration

### 1. Create Terraform State Bucket

```bash
aws s3 mb s3://king-ai-terraform-state --region us-east-1
aws s3api put-bucket-versioning \
  --bucket king-ai-terraform-state \
  --versioning-configuration Status=Enabled
```

### 2. Configure Variables

Create a `terraform.tfvars` file:

```hcl
aws_region = "us-east-1"
environment = "prod"
vpc_cidr = "10.0.0.0/16"
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

# GPU Configuration
gpu_instance_type = "p5.48xlarge"
gpu_min_instances = 2
gpu_max_instances = 8

# Database Configuration
db_instance_class = "db.r6g.xlarge"
redis_node_type = "cache.r6g.large"
```

## Deployment

### Initialize Terraform

```bash
cd infrastructure/terraform
terraform init
```

### Plan Deployment

```bash
terraform plan -out=tfplan
```

Review the plan to ensure all resources are correct.

### Apply Configuration

```bash
terraform apply tfplan
```

This will create all infrastructure resources. The process takes approximately 15-20 minutes.

### Retrieve Outputs

```bash
terraform output
```

Important outputs:
- `alb_dns_name` - Load balancer endpoint for inference requests
- `rds_endpoint` - Database connection endpoint
- `redis_endpoint` - Redis cache endpoint
- `inference_queue_url` - SQS queue URL for inference requests

## Post-Deployment Configuration

### 1. Update Application Configuration

Update your `.env` file with the Terraform outputs:

```bash
# Get outputs
ALB_DNS=$(terraform output -raw alb_dns_name)
RDS_ENDPOINT=$(terraform output -raw rds_endpoint)
REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)
QUEUE_URL=$(terraform output -raw inference_queue_url)

# Update .env
echo "VLLM_URL=http://${ALB_DNS}:8080" >> .env
echo "DATABASE_URL=postgresql+asyncpg://kingadmin:PASSWORD@${RDS_ENDPOINT}/kingai" >> .env
echo "REDIS_URL=redis://${REDIS_ENDPOINT}:6379" >> .env
echo "SQS_INFERENCE_QUEUE=${QUEUE_URL}" >> .env
```

### 2. Retrieve Database Password

The database password is stored in AWS Secrets Manager:

```bash
aws secretsmanager get-secret-value \
  --secret-id king-ai/prod/db-password \
  --query SecretString \
  --output text | jq -r .password
```

### 3. Deploy Application

Deploy your application to the API servers created by the EC2 configuration.

## Scaling

The infrastructure automatically scales based on SQS queue depth:
- **Scale Up**: When queue depth > 100 for 2 consecutive minutes
- **Scale Down**: When queue depth < 10 for 5 consecutive minutes
- **Min Instances**: 2 (configurable)
- **Max Instances**: 8 (configurable)

## Cost Estimates

Approximate monthly costs (us-east-1):
- **GPU Instances (2x p5.48xlarge)**: ~$20,000/month
- **RDS (db.r6g.xlarge Multi-AZ)**: ~$800/month
- **ElastiCache (cache.r6g.large)**: ~$250/month
- **ALB**: ~$20/month + data transfer
- **Data Transfer**: Variable based on usage
- **Total**: ~$21,000+/month

## Maintenance

### Update Infrastructure

1. Modify Terraform files
2. Run `terraform plan` to review changes
3. Run `terraform apply` to apply changes

### Destroy Infrastructure

⚠️ **Warning**: This will delete all resources and data.

```bash
terraform destroy
```

## Security Considerations

- All data at rest is encrypted (EBS, RDS, S3)
- Database credentials stored in AWS Secrets Manager
- Private subnets for compute resources
- Security groups restrict access between components
- NAT Gateways for outbound internet access from private subnets

## Monitoring

Configure CloudWatch alarms for:
- GPU instance health
- Database performance metrics
- Queue depth alerts
- Auto-scaling events

## Troubleshooting

### GPU Instances Not Starting

Check CloudWatch logs and user data execution:
```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=king-ai-gpu-inference" \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name]'
```

### Database Connection Issues

Verify security group rules and network connectivity:
```bash
aws rds describe-db-instances \
  --db-instance-identifier king-ai-prod
```

### Scaling Issues

Check CloudWatch alarms and SQS metrics:
```bash
aws cloudwatch describe-alarms \
  --alarm-names king-ai-queue-depth-high king-ai-queue-depth-low
```

## Support

For issues or questions, please refer to the main project documentation.
