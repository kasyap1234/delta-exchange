output "external_ip" {
  description = "The external static IP of the trading bot VM"
  value       = google_compute_address.static_ip.address
}

output "ssh_command" {
  description = "Command to connect to the VM"
  value       = "ssh ${var.ssh_user}@${google_compute_address.static_ip.address}"
}
