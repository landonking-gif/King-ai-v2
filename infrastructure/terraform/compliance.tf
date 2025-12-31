# Snyk and Vanta Integration Terraform Module
# Adds compliance scanning and security monitoring to King AI infrastructure

# Snyk Organization Integration
resource "aws_secretsmanager_secret" "snyk" {
  name        = "king-ai/${var.environment}/snyk"
  description = "Snyk API credentials for vulnerability scanning"
}

resource "aws_secretsmanager_secret_version" "snyk" {
  secret_id = aws_secretsmanager_secret.snyk.id
  secret_string = jsonencode({
    api_token = var.snyk_api_token
    org_id    = var.snyk_org_id
  })
}

# Vanta Agent IAM Role
resource "aws_iam_role" "vanta_agent" {
  name = "king-ai-${var.environment}-vanta-agent"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${var.vanta_aws_account_id}:root"
        }
        Condition = {
          StringEquals = {
            "sts:ExternalId" = var.vanta_external_id
          }
        }
      }
    ]
  })
  
  tags = local.common_tags
}

# Vanta Agent Policy - Read-only access for compliance monitoring
resource "aws_iam_role_policy" "vanta_agent" {
  name = "vanta-compliance-monitoring"
  role = aws_iam_role.vanta_agent.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecurityAuditAccess"
        Effect = "Allow"
        Action = [
          "access-analyzer:List*",
          "access-analyzer:Get*",
          "acm:Describe*",
          "acm:List*",
          "cloudtrail:Describe*",
          "cloudtrail:Get*",
          "cloudtrail:List*",
          "cloudtrail:LookupEvents",
          "cloudwatch:Describe*",
          "cloudwatch:Get*",
          "cloudwatch:List*",
          "config:Describe*",
          "config:Get*",
          "config:List*",
          "ec2:Describe*",
          "ecr:Describe*",
          "ecr:Get*",
          "ecr:List*",
          "ecs:Describe*",
          "ecs:List*",
          "elasticache:Describe*",
          "elasticache:List*",
          "guardduty:Get*",
          "guardduty:List*",
          "iam:Get*",
          "iam:List*",
          "iam:Generate*",
          "kms:Describe*",
          "kms:Get*",
          "kms:List*",
          "lambda:Get*",
          "lambda:List*",
          "logs:Describe*",
          "logs:Get*",
          "rds:Describe*",
          "rds:List*",
          "s3:GetBucket*",
          "s3:GetEncryption*",
          "s3:GetLifecycle*",
          "s3:ListAllMyBuckets",
          "s3:ListBucket",
          "secretsmanager:Describe*",
          "secretsmanager:List*",
          "sns:Get*",
          "sns:List*",
          "sqs:Get*",
          "sqs:List*",
          "ssm:Describe*",
          "ssm:Get*",
          "ssm:List*",
          "tag:Get*",
          "vpc:Describe*"
        ]
        Resource = "*"
      }
    ]
  })
}

# AWS Config Rules for Compliance
resource "aws_config_config_rule" "encryption_at_rest" {
  name = "king-ai-encryption-at-rest"
  
  source {
    owner             = "AWS"
    source_identifier = "RDS_STORAGE_ENCRYPTED"
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

resource "aws_config_config_rule" "vpc_flow_logs" {
  name = "king-ai-vpc-flow-logs"
  
  source {
    owner             = "AWS"
    source_identifier = "VPC_FLOW_LOGS_ENABLED"
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

resource "aws_config_config_rule" "cloudtrail_enabled" {
  name = "king-ai-cloudtrail-enabled"
  
  source {
    owner             = "AWS"
    source_identifier = "CLOUD_TRAIL_ENABLED"
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

resource "aws_config_config_rule" "iam_mfa" {
  name = "king-ai-iam-mfa"
  
  source {
    owner             = "AWS"
    source_identifier = "IAM_USER_MFA_ENABLED"
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

resource "aws_config_config_rule" "s3_bucket_ssl" {
  name = "king-ai-s3-ssl"
  
  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_SSL_REQUESTS_ONLY"
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

# AWS Config Recorder
resource "aws_config_configuration_recorder" "main" {
  name     = "king-ai-${var.environment}-recorder"
  role_arn = aws_iam_role.config.arn
  
  recording_group {
    all_supported                 = true
    include_global_resource_types = true
  }
}

resource "aws_config_configuration_recorder_status" "main" {
  name       = aws_config_configuration_recorder.main.name
  is_enabled = true
  depends_on = [aws_config_delivery_channel.main]
}

resource "aws_config_delivery_channel" "main" {
  name           = "king-ai-${var.environment}-delivery"
  s3_bucket_name = aws_s3_bucket.config.id
  
  snapshot_delivery_properties {
    delivery_frequency = "TwentyFour_Hours"
  }
  
  depends_on = [aws_config_configuration_recorder.main]
}

# S3 Bucket for Config
resource "aws_s3_bucket" "config" {
  bucket = "king-ai-${var.environment}-config-${data.aws_caller_identity.current.account_id}"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "config" {
  bucket = aws_s3_bucket.config.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "config" {
  bucket = aws_s3_bucket.config.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# IAM Role for Config
resource "aws_iam_role" "config" {
  name = "king-ai-${var.environment}-config-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "config" {
  role       = aws_iam_role.config.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWS_ConfigRole"
}

resource "aws_iam_role_policy" "config_s3" {
  name = "config-s3-access"
  role = aws_iam_role.config.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetBucketAcl"
        ]
        Resource = [
          aws_s3_bucket.config.arn,
          "${aws_s3_bucket.config.arn}/*"
        ]
      }
    ]
  })
}

# CloudTrail for Audit Logging
resource "aws_cloudtrail" "main" {
  name                          = "king-ai-${var.environment}-trail"
  s3_bucket_name                = aws_s3_bucket.cloudtrail.id
  include_global_service_events = true
  is_multi_region_trail         = true
  enable_log_file_validation    = true
  
  event_selector {
    read_write_type           = "All"
    include_management_events = true
  }
  
  tags = local.common_tags
}

# S3 Bucket for CloudTrail
resource "aws_s3_bucket" "cloudtrail" {
  bucket = "king-ai-${var.environment}-cloudtrail-${data.aws_caller_identity.current.account_id}"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_policy" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSCloudTrailAclCheck"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.cloudtrail.arn
      },
      {
        Sid    = "AWSCloudTrailWrite"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.cloudtrail.arn}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      }
    ]
  })
}

resource "aws_s3_bucket_versioning" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# GuardDuty for Threat Detection
resource "aws_guardduty_detector" "main" {
  enable = true
  
  finding_publishing_frequency = "FIFTEEN_MINUTES"
  
  datasources {
    s3_logs {
      enable = true
    }
    kubernetes {
      audit_logs {
        enable = true
      }
    }
    malware_protection {
      scan_ec2_instance_with_findings {
        ebs_volumes {
          enable = true
        }
      }
    }
  }
  
  tags = local.common_tags
}

# Security Hub for Centralized Findings
resource "aws_securityhub_account" "main" {}

resource "aws_securityhub_standards_subscription" "cis" {
  standards_arn = "arn:aws:securityhub:::ruleset/cis-aws-foundations-benchmark/v/1.2.0"
  depends_on    = [aws_securityhub_account.main]
}

resource "aws_securityhub_standards_subscription" "aws_foundational" {
  standards_arn = "arn:aws:securityhub:${var.aws_region}::standards/aws-foundational-security-best-practices/v/1.0.0"
  depends_on    = [aws_securityhub_account.main]
}

# SNS Topic for Security Alerts
resource "aws_sns_topic" "security_alerts" {
  name = "king-ai-${var.environment}-security-alerts"
  
  tags = local.common_tags
}

resource "aws_sns_topic_subscription" "security_email" {
  topic_arn = aws_sns_topic.security_alerts.arn
  protocol  = "email"
  endpoint  = var.security_alert_email
}

# EventBridge Rule for GuardDuty Findings
resource "aws_cloudwatch_event_rule" "guardduty_findings" {
  name        = "king-ai-${var.environment}-guardduty-findings"
  description = "Capture GuardDuty findings"
  
  event_pattern = jsonencode({
    source      = ["aws.guardduty"]
    detail-type = ["GuardDuty Finding"]
    detail = {
      severity = [{ numeric = [">=", 4] }]
    }
  })
  
  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "guardduty_sns" {
  rule      = aws_cloudwatch_event_rule.guardduty_findings.name
  target_id = "send-to-sns"
  arn       = aws_sns_topic.security_alerts.arn
}

# CodeBuild Project for Snyk Scanning
resource "aws_codebuild_project" "snyk_scan" {
  name         = "king-ai-${var.environment}-snyk-scan"
  description  = "Snyk vulnerability scanning for King AI"
  service_role = aws_iam_role.codebuild.arn
  
  artifacts {
    type = "NO_ARTIFACTS"
  }
  
  environment {
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/standard:7.0"
    type                        = "LINUX_CONTAINER"
    privileged_mode             = true
    image_pull_credentials_type = "CODEBUILD"
    
    environment_variable {
      name  = "SNYK_TOKEN"
      value = aws_secretsmanager_secret.snyk.arn
      type  = "SECRETS_MANAGER"
    }
  }
  
  source {
    type      = "GITHUB"
    location  = var.github_repo_url
    buildspec = <<-EOF
      version: 0.2
      phases:
        install:
          commands:
            - npm install -g snyk
        pre_build:
          commands:
            - snyk auth $SNYK_TOKEN
        build:
          commands:
            - snyk test --severity-threshold=high
            - snyk container test --severity-threshold=high
            - snyk iac test --severity-threshold=medium
      reports:
        snyk:
          files:
            - '**/*'
          base-directory: .snyk
    EOF
  }
  
  tags = local.common_tags
}

# IAM Role for CodeBuild
resource "aws_iam_role" "codebuild" {
  name = "king-ai-${var.environment}-codebuild-snyk"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "codebuild.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy" "codebuild" {
  name = "codebuild-permissions"
  role = aws_iam_role.codebuild.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.snyk.arn
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# Variables for compliance module
variable "snyk_api_token" {
  description = "Snyk API token for vulnerability scanning"
  type        = string
  sensitive   = true
  default     = ""
}

variable "snyk_org_id" {
  description = "Snyk organization ID"
  type        = string
  default     = ""
}

variable "vanta_aws_account_id" {
  description = "Vanta AWS account ID for cross-account access"
  type        = string
  default     = "956882708938"  # Vanta's AWS account
}

variable "vanta_external_id" {
  description = "Vanta external ID for cross-account access"
  type        = string
  sensitive   = true
  default     = ""
}

variable "github_repo_url" {
  description = "GitHub repository URL for Snyk scanning"
  type        = string
  default     = "https://github.com/org/king-ai-v2"
}

variable "security_alert_email" {
  description = "Email address for security alerts"
  type        = string
  default     = "security@company.com"
}

# Data sources
data "aws_caller_identity" "current" {}

# Outputs
output "vanta_role_arn" {
  description = "IAM role ARN for Vanta agent"
  value       = aws_iam_role.vanta_agent.arn
}

output "cloudtrail_bucket" {
  description = "CloudTrail S3 bucket name"
  value       = aws_s3_bucket.cloudtrail.id
}

output "guardduty_detector_id" {
  description = "GuardDuty detector ID"
  value       = aws_guardduty_detector.main.id
}

output "security_alerts_topic" {
  description = "SNS topic ARN for security alerts"
  value       = aws_sns_topic.security_alerts.arn
}
