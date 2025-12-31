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

# GPU Instance Configuration for Ollama Inference
resource "aws_launch_template" "ollama_gpu" {
  name_prefix   = "king-ai-ollama-gpu-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = "p5.48xlarge"

  vpc_security_group_ids = [aws_security_group.ollama.id]
  iam_instance_profile {
    name = aws_iam_instance_profile.ollama.name
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 500
      volume_type = "gp3"
      iops        = 16000
      throughput  = 1000
    }
  }

  user_data = base64encode(templatefile("${path.module}/scripts/ollama_setup.sh", {
    ollama_model = var.ollama_model
    vllm_model   = var.vllm_model
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "king-ai-ollama-gpu"
      Component   = "inference"
      Environment = var.environment
    }
  }
}

resource "aws_instance" "ollama_gpu" {
  count         = var.gpu_instance_count
  ami           = data.aws_ami.deep_learning.id
  instance_type = "p5.48xlarge"  # 8x H100 GPUs

  subnet_id                   = aws_subnet.private[count.index % length(aws_subnet.private)].id
  vpc_security_group_ids      = [aws_security_group.ollama.id]
  iam_instance_profile        = aws_iam_instance_profile.ollama.name

  root_block_device {
    volume_size = 500
    volume_type = "gp3"
    iops        = 16000
    throughput  = 1000
  }

  user_data = base64encode(templatefile("${path.module}/scripts/ollama_setup.sh", {
    ollama_model = var.ollama_model
    vllm_model   = var.vllm_model
  }))

  tags = {
    Name        = "king-ai-ollama-gpu-${count.index}"
    Component   = "inference"
    Environment = var.environment
  }
}

# Deep Learning AMI data source
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*"]
  }
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

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
    description     = "vLLM API"
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

# Auto-scaling for GPU instances based on queue depth
resource "aws_autoscaling_group" "ollama_gpu" {
  name                = "king-ai-ollama-gpu-asg"
  min_size            = 2
  max_size            = 8
  desired_capacity    = 2
  vpc_zone_identifier = aws_subnet.private[*].id

  launch_template {
    id      = aws_launch_template.ollama_gpu.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "king-ai-ollama-gpu"
    propagate_at_launch = true
  }
}

# Scaling policy based on inference queue depth
resource "aws_autoscaling_policy" "ollama_scale_up" {
  name                   = "ollama-scale-up"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.ollama_gpu.name
}

resource "aws_cloudwatch_metric_alarm" "ollama_queue_high" {
  alarm_name          = "ollama-queue-depth-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "InferenceQueueDepth"
  namespace           = "KingAI/Inference"
  period              = 60
  statistic           = "Average"
  threshold           = 100
  alarm_actions       = [aws_autoscaling_policy.ollama_scale_up.arn]
}