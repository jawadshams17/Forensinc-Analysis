# ================================================================
# AWS Spot Instance Infrastructure — Terraform Configuration
# ================================================================
# Provisions:
#   - VPC with isolated subnet for malware execution
#   - Spot Fleet for cost-effective ephemeral VMs
#   - S3 bucket for memory dump storage
#   - CloudWatch alarms for billing alerts
#   - IAM roles with least-privilege access
# ================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ─────────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "amf-research"
}

variable "spot_max_price" {
  description = "Maximum hourly price for Spot Instances"
  type        = string
  default     = "0.10"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "m5.xlarge"
}

variable "budget_alert_threshold" {
  description = "Monthly budget alert threshold in USD"
  type        = number
  default     = 200
}

# ── Networking ────────────────────────────────────────────────────

resource "aws_vpc" "malware_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name    = "${var.project_name}-vpc"
    Project = var.project_name
  }
}

resource "aws_subnet" "isolated" {
  vpc_id                  = aws_vpc.malware_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = false

  tags = {
    Name    = "${var.project_name}-isolated-subnet"
    Project = var.project_name
  }
}

resource "aws_security_group" "malware_sg" {
  name_prefix = "${var.project_name}-sg-"
  vpc_id      = aws_vpc.malware_vpc.id

  # No inbound from internet
  # Outbound only to S3 (VPC endpoint) for dump upload
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS for S3 upload"
  }

  tags = {
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }
}

# ── S3 Storage ────────────────────────────────────────────────────

resource "aws_s3_bucket" "dumps" {
  bucket_prefix = "${var.project_name}-dumps-"

  tags = {
    Project = var.project_name
  }
}

resource "aws_s3_bucket_versioning" "dumps" {
  bucket = aws_s3_bucket.dumps.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dumps" {
  bucket = aws_s3_bucket.dumps.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "dumps" {
  bucket = aws_s3_bucket.dumps.id

  rule {
    id     = "transition-to-ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# ── IAM ───────────────────────────────────────────────────────────

resource "aws_iam_role" "spot_instance_role" {
  name_prefix = "${var.project_name}-spot-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Project = var.project_name
  }
}

resource "aws_iam_role_policy" "s3_upload" {
  name_prefix = "${var.project_name}-s3-"
  role        = aws_iam_role.spot_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetObject",
      ]
      Resource = "${aws_s3_bucket.dumps.arn}/*"
    }]
  })
}

resource "aws_iam_instance_profile" "spot_profile" {
  name_prefix = "${var.project_name}-"
  role        = aws_iam_role.spot_instance_role.name
}

# ── Spot Fleet ────────────────────────────────────────────────────

resource "aws_iam_role" "spot_fleet_role" {
  name_prefix = "${var.project_name}-fleet-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "spotfleet.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "spot_fleet_policy" {
  role       = aws_iam_role.spot_fleet_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

# ── CloudWatch Billing Alert ─────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "budget_alert" {
  alarm_name          = "${var.project_name}-budget-alert"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 86400
  statistic           = "Maximum"
  threshold           = var.budget_alert_threshold
  alarm_description   = "Alert when monthly spending exceeds ${var.budget_alert_threshold} USD"

  dimensions = {
    Currency = "USD"
  }

  tags = {
    Project = var.project_name
  }
}

# ── Outputs ───────────────────────────────────────────────────────

output "vpc_id" {
  value = aws_vpc.malware_vpc.id
}

output "s3_bucket" {
  value = aws_s3_bucket.dumps.bucket
}

output "instance_profile_arn" {
  value = aws_iam_instance_profile.spot_profile.arn
}

output "security_group_id" {
  value = aws_security_group.malware_sg.id
}
