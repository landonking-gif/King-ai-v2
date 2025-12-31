output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = aws_subnet.private[*].id
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "inference_queue_url" {
  description = "URL of the SQS inference queue"
  value       = aws_sqs_queue.inference_queue.url
}

output "rds_endpoint" {
  description = "Endpoint of the RDS PostgreSQL instance"
  value       = aws_db_instance.main.endpoint
}

output "redis_endpoint" {
  description = "Endpoint of the ElastiCache Redis replication group"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}
