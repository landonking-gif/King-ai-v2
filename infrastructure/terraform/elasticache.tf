# Security Group for ElastiCache
resource "aws_security_group" "redis" {
  name        = "king-ai-redis-sg"
  description = "Security group for ElastiCache Redis"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Redis access from VPC"
  }

  tags = {
    Name = "king-ai-redis-sg"
  }
}

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "main" {
  name       = "king-ai-redis-subnet-group"
  subnet_ids = aws_subnet.private[*].id
}

# ElastiCache Parameter Group
resource "aws_elasticache_parameter_group" "redis" {
  name   = "king-ai-redis-params"
  family = "redis7"

  parameter {
    name  = "maxmemory-policy"
    value = "volatile-lru"
  }
}

# ElastiCache Redis Replication Group (HA)
resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "king-ai-${var.environment}-redis"
  description                = "King AI Redis cluster with HA"
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = aws_elasticache_parameter_group.redis.name
  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.redis.id]

  # HA Configuration
  automatic_failover_enabled = true
  multi_az_enabled           = true
  num_cache_clusters         = 3

  # Maintenance
  maintenance_window         = "mon:05:00-mon:07:00"
  snapshot_retention_limit   = 7
  snapshot_window            = "03:00-05:00"

  # Encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  engine_version             = "7.0"

  tags = {
    Name        = "king-ai-redis"
    Environment = var.environment
  }
}
