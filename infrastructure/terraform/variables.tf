variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU inference"
  type        = string
  default     = "p5.48xlarge"
}

variable "gpu_min_instances" {
  description = "Minimum number of GPU instances"
  type        = number
  default     = 2
}

variable "gpu_max_instances" {
  description = "Maximum number of GPU instances for auto-scaling"
  type        = number
  default     = 8
}

variable "gpu_instance_count" {
  description = "Number of GPU instances to deploy"
  type        = number
  default     = 2
}

variable "ollama_model" {
  description = "Ollama model to deploy"
  type        = string
  default     = "llama3.1:70b"
}

variable "vllm_model" {
  description = "vLLM model to deploy"
  type        = string
  default     = "meta-llama/Meta-Llama-3.1-70B-Instruct"
}

variable "vpn_cidr" {
  description = "CIDR block for VPN access (SSH, admin)"
  type        = string
  default     = "10.100.0.0/16"
}

variable "acm_certificate_arn" {
  description = "ARN of ACM certificate for HTTPS"
  type        = string
  default     = ""
}

variable "datadog_api_key" {
  description = "Datadog API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "datadog_app_key" {
  description = "Datadog Application key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r6g.large"
}
