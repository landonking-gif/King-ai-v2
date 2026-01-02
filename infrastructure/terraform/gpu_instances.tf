# =============================================================================
# GPU Instance Configuration for Ollama/vLLM Inference
# NOTE: Primary GPU ASG is defined in autoscaling.tf - this file contains
# supporting resources like IAM roles and security groups only.
# =============================================================================

# IAM instance profile for Ollama
resource "aws_iam_instance_profile" "ollama" {
  name = "king-ai-ollama-instance-profile"
  role = aws_iam_role.ollama.name
}

resource "aws_iam_role" "ollama" {
  name = "king-ai-ollama-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ollama_ssm" {
  role       = aws_iam_role.ollama.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "ollama_cloudwatch" {
  role       = aws_iam_role.ollama.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_role_policy" "ollama_sqs" {
  name = "king-ai-ollama-sqs-policy"
  role = aws_iam_role.ollama.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:GetQueueUrl"
        ]
        Resource = aws_sqs_queue.inference_queue.arn
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = "arn:aws:secretsmanager:${var.aws_region}:*:secret:king-ai/*"
      }
    ]
  })
}

# Security group for Ollama
resource "aws_security_group" "ollama" {
  name        = "king-ai-ollama-sg"
  description = "Security group for Ollama inference servers"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 11434
    to_port         = 11434
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
    description     = "Ollama API"
  }

  # vLLM API - using consistent port 8080
  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
    description     = "vLLM API"
  }

  # Prometheus metrics endpoint
  ingress {
    from_port       = 9090
    to_port         = 9090
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
    description     = "Metrics endpoint"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "king-ai-ollama-sg"
  }
}