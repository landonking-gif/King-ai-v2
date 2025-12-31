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

variable "gpu_instance_count" {
  description = "Number of GPU instances to deploy"
  type        = number
  default     = 2
}

variable "ollama_model" {
  description = "Ollama model to deploy"
  type        = string
  default     = "llama2:13b"
}

variable "vllm_model" {
  description = "vLLM model to deploy"
  type        = string
  default     = "meta-llama/Llama-2-13b-hf"
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
