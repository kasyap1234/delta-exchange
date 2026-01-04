# Terraform configuration for Delta Exchange Trading Bot on GCP Free Tier

terraform {
  required_version = ">= 1.0.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Network
resource "google_compute_network" "vpc_network" {
  name                    = "delta-bot-vpc-v2"
  auto_create_subnetworks = true
}

# Firewall rule to allow SSH
resource "google_compute_firewall" "allow_ssh" {
  name    = "delta-bot-allow-ssh"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["delta-bot"]
}

# Static External IP
resource "google_compute_address" "static_ip" {
  name   = "delta-bot-static-ip"
  region = var.region
}

# Compute Engine Instance (GCP Free Tier: e2-micro)
resource "google_compute_instance" "trading_bot" {
  name         = "delta-trading-bot"
  machine_type = "e2-micro" # GCP Free Tier eligible
  zone         = var.zone
  tags         = ["delta-bot"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = 10 # 30GB is the free tier limit
      type  = "pd-standard"
    }
  }

  network_interface {
    network = google_compute_network.vpc_network.name
    access_config {
      nat_ip = google_compute_address.static_ip.address
    }
  }

  metadata = {
    ssh-keys = "${var.ssh_user}:${file(var.ssh_pub_key_path)}"
  }

  # SSH Connection details for provisioners
  connection {
    type        = "ssh"
    user        = var.ssh_user
    private_key = file(var.ssh_private_key_path)
    host        = google_compute_address.static_ip.address
  }

  # 1. Create app directory
  provisioner "remote-exec" {
    inline = [
      "mkdir -p /home/${var.ssh_user}/delta-exchange"
    ]
  }

  # 2. Push local code to VM using rsync
  provisioner "local-exec" {
    command = "rsync -avz -e 'ssh -o StrictHostKeyChecking=no -i ${var.ssh_private_key_path}' --exclude 'venv' --exclude '.terraform' --exclude '.git' --exclude '__pycache__' ${path.module}/../ ${var.ssh_user}@${google_compute_address.static_ip.address}:/home/${var.ssh_user}/delta-exchange"
  }

  # 3. Run the automated setup script
  provisioner "remote-exec" {
    inline = [
      "cd /home/${var.ssh_user}/delta-exchange",
      "chmod +x deploy/setup.sh",
      "bash deploy/setup.sh"
    ]
  }

  service_account {
    scopes = ["cloud-platform"]
  }

  lifecycle {
    ignore_changes = [metadata["ssh-keys"]]
  }
}
