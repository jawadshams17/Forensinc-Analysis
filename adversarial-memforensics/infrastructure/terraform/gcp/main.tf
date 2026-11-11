# ================================================================
# GCP Preemptible VM Infrastructure — Terraform Configuration
# ================================================================
# Provisions:
#   - VPC with firewall rules for isolated malware execution
#   - Preemptible instance template with 30s termination notice
#   - GCS bucket for memory dump storage
#   - Budget alerts via Cloud Billing
# ================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

# ── Variables ─────────────────────────────────────────────────────

variable "gcp_project" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "project_name" {
  description = "Project name for resource labeling"
  type        = string
  default     = "amf-research"
}

variable "machine_type" {
  description = "VM machine type"
  type        = string
  default     = "n1-standard-4"
}

variable "budget_amount" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 200
}

# ── Networking ────────────────────────────────────────────────────

resource "google_compute_network" "malware_vpc" {
  name                    = "${var.project_name}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "isolated" {
  name          = "${var.project_name}-isolated"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.gcp_region
  network       = google_compute_network.malware_vpc.id

  private_ip_google_access = true
}

resource "google_compute_firewall" "deny_all_ingress" {
  name    = "${var.project_name}-deny-ingress"
  network = google_compute_network.malware_vpc.name

  deny {
    protocol = "all"
  }

  direction     = "INGRESS"
  source_ranges = ["0.0.0.0/0"]
  priority      = 1000
}

resource "google_compute_firewall" "allow_gcs_egress" {
  name    = "${var.project_name}-allow-gcs"
  network = google_compute_network.malware_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  direction          = "EGRESS"
  destination_ranges = ["0.0.0.0/0"]
  priority           = 1000
}

# ── GCS Storage ───────────────────────────────────────────────────

resource "google_storage_bucket" "dumps" {
  name     = "${var.project_name}-dumps-${var.gcp_project}"
  location = var.gcp_region

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = ""
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  labels = {
    project = var.project_name
  }
}

# ── Service Account ──────────────────────────────────────────────

resource "google_service_account" "vm_sa" {
  account_id   = "${var.project_name}-vm"
  display_name = "AMF Research VM Service Account"
}

resource "google_storage_bucket_iam_member" "vm_writer" {
  bucket = google_storage_bucket.dumps.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.vm_sa.email}"
}

# ── Instance Template ────────────────────────────────────────────

resource "google_compute_instance_template" "preemptible" {
  name_prefix  = "${var.project_name}-"
  machine_type = var.machine_type

  scheduling {
    preemptible       = true
    automatic_restart = false
  }

  disk {
    source_image = "ubuntu-os-cloud/ubuntu-2204-lts"
    auto_delete  = true
    boot         = true
    disk_size_gb = 50
  }

  network_interface {
    subnetwork = google_compute_subnetwork.isolated.id
    # No external IP — use Private Google Access for GCS
  }

  service_account {
    email  = google_service_account.vm_sa.email
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    # Install dependencies
    apt-get update && apt-get install -y linux-headers-$(uname -r) git

    # Clone and build LiME
    cd /opt
    git clone https://github.com/504ensicsLabs/LiME.git
    cd LiME/src && make

    # Start preemption watcher (1-second polling for 30s notice window)
    while true; do
      STATUS=$(curl -s -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/preempted)
      if [ "$STATUS" = "TRUE" ]; then
        echo "[$(date)] PREEMPTION DETECTED — uploading partial dump"
        gsutil cp /tmp/acquisition/*.lime gs://${google_storage_bucket.dumps.name}/
        gsutil cp /tmp/acquisition/*.json gs://${google_storage_bucket.dumps.name}/
        break
      fi
      sleep 1
    done &

    echo "AMF acquisition VM ready"
  EOF

  labels = {
    project = var.project_name
  }
}

# ── Budget Alert ──────────────────────────────────────────────────

resource "google_billing_budget" "project_budget" {
  billing_account = data.google_billing_account.account.id
  display_name    = "${var.project_name}-budget"

  budget_filter {
    projects = ["projects/${var.gcp_project}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_amount)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.8
  }
  threshold_rules {
    threshold_percent = 1.0
  }
}

data "google_billing_account" "account" {
  display_name = "My Billing Account"
  open         = true
}

# ── Outputs ───────────────────────────────────────────────────────

output "vpc_id" {
  value = google_compute_network.malware_vpc.id
}

output "gcs_bucket" {
  value = google_storage_bucket.dumps.name
}

output "service_account_email" {
  value = google_service_account.vm_sa.email
}

output "instance_template" {
  value = google_compute_instance_template.preemptible.self_link
}
