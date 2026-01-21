# Data source for Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# IAM Role for GPU Instances
resource "aws_iam_role" "gpu_inference" {
  name = "king-ai-gpu-inference-role"

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

resource "aws_iam_role_policy" "gpu_inference_policy" {
  name = "king-ai-gpu-inference-policy"
  role = aws_iam_role.gpu_inference.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.inference_queue.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "arn:aws:s3:::king-ai-artifacts/*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "gpu_inference" {
  name = "king-ai-gpu-inference-profile"
  role = aws_iam_role.gpu_inference.name
}

# Security Group for GPU Instances
resource "aws_security_group" "gpu_inference" {
  name        = "king-ai-gpu-inference-sg"
  description = "Security group for GPU inference instances"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "vLLM API"
  }

  ingress {
    from_port   = 11434
    to_port     = 11434
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Ollama API"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name = "king-ai-gpu-inference-sg"
  }
}

# Launch Template for GPU Instances
resource "aws_launch_template" "gpu_inference" {
  name_prefix   = "king-ai-gpu-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.gpu_instance_type

  vpc_security_group_ids = [aws_security_group.gpu_inference.id]

  iam_instance_profile {
    name = aws_iam_instance_profile.gpu_inference.name
  }

  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -e
    
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Install vLLM
    pip install vllm
    
    # Pull the model
    ollama pull qwen3:32b
    
    # Start Ollama service
    systemctl enable ollama
    systemctl start ollama
    
    # Start vLLM server for high-throughput (port 8080 matches ALB target group)
    nohup python -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Llama-3.1-70B-Instruct \
      --tensor-parallel-size 8 \
      --host 0.0.0.0 \
      --port 8080 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.9 &
    
    # Signal ready to ALB
    curl -X PUT "http://169.254.169.254/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"
  EOF
  )

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 500
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  monitoring {
    enabled = true
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "king-ai-gpu-inference"
      Environment = var.environment
      Role        = "llm-inference"
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "gpu_inference" {
  name                = "king-ai-gpu-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.inference.arn]
  health_check_type   = "ELB"
  
  min_size         = var.gpu_min_instances
  max_size         = var.gpu_max_instances
  desired_capacity = var.gpu_min_instances

  launch_template {
    id      = aws_launch_template.gpu_inference.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "king-ai-gpu-inference"
    propagate_at_launch = true
  }
}

# Scaling Policy - Scale based on SQS queue depth
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "king-ai-scale-up"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.gpu_inference.name
}

resource "aws_autoscaling_policy" "scale_down" {
  name                   = "king-ai-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 600
  autoscaling_group_name = aws_autoscaling_group.gpu_inference.name
}

# CloudWatch Alarm for Queue Depth > 100
resource "aws_cloudwatch_metric_alarm" "queue_depth_high" {
  alarm_name          = "king-ai-queue-depth-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 60
  statistic           = "Average"
  threshold           = 100
  alarm_description   = "Scale up when inference queue depth exceeds 100"
  alarm_actions       = [aws_autoscaling_policy.scale_up.arn]

  dimensions = {
    QueueName = aws_sqs_queue.inference_queue.name
  }
}

resource "aws_cloudwatch_metric_alarm" "queue_depth_low" {
  alarm_name          = "king-ai-queue-depth-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 5
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 60
  statistic           = "Average"
  threshold           = 10
  alarm_description   = "Scale down when inference queue depth is low"
  alarm_actions       = [aws_autoscaling_policy.scale_down.arn]

  dimensions = {
    QueueName = aws_sqs_queue.inference_queue.name
  }
}

# SQS Dead Letter Queue for failed inference messages
resource "aws_sqs_queue" "inference_dlq" {
  name                       = "king-ai-inference-dlq"
  message_retention_seconds  = 1209600  # 14 days retention for debugging
  
  tags = {
    Environment = var.environment
    Purpose     = "dead-letter-queue"
  }
}

# SQS Queue for inference requests
resource "aws_sqs_queue" "inference_queue" {
  name                       = "king-ai-inference-queue"
  delay_seconds              = 0
  max_message_size           = 262144
  message_retention_seconds  = 86400
  receive_wait_time_seconds  = 10
  visibility_timeout_seconds = 300

  # Send failed messages to DLQ after 3 attempts
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.inference_dlq.arn
    maxReceiveCount     = 3
  })

  tags = {
    Environment = var.environment
  }
}

# CloudWatch alarm for DLQ messages (indicates processing failures)
resource "aws_cloudwatch_metric_alarm" "dlq_messages" {
  alarm_name          = "king-ai-dlq-has-messages"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Alert when messages appear in DLQ - indicates inference failures"

  dimensions = {
    QueueName = aws_sqs_queue.inference_dlq.name
  }
}
