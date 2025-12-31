terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    datadog = {
      source  = "DataDog/datadog"
      version = "~> 3.0"
    }
  }

  # Configure backend for state storage
  backend "s3" {
    bucket = "king-ai-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
}

# Datadog provider for monitoring
provider "datadog" {
  api_key = var.datadog_api_key
  app_key = var.datadog_app_key
}

# Pinecone secrets stored in AWS Secrets Manager
resource "aws_secretsmanager_secret" "pinecone" {
  name        = "king-ai/${var.environment}/pinecone"
  description = "Pinecone API credentials for vector embeddings"
}

resource "aws_secretsmanager_secret_version" "pinecone" {
  secret_id = aws_secretsmanager_secret.pinecone.id
  secret_string = jsonencode({
    api_key     = var.pinecone_api_key
    environment = var.pinecone_environment
    index_name  = var.pinecone_index_name
  })
}
