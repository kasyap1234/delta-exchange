#!/bin/bash
# Fully Automated Deployment Script for Delta Exchange Bot
# Uses 'uv' for fast, reliable Python management.

set -e # Exit on error

echo "🚀 Starting Full Automation Setup (UV Edition)..."

# 0. Get current user and dir
USER=$(whoami)
APP_DIR=$(pwd)
SERVICE_NAME="delta-bot"

echo "👤 Running as user: $USER"
echo "📂 App directory: $APP_DIR"

# 1. Update system dependencies (with lock handling)
echo "📦 Updating system dependencies..."
export DEBIAN_FRONTEND=noninteractive

# Wait for any existing apt process (e.g., unattended-upgrades)
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "Waiting for apt lock..."
    sleep 5
done

sudo -E apt-get update
sudo -E apt-get install -y curl build-essential

# 2. Install uv (The Python Package Manager)
echo "⚡ Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Setup Python Environment
echo "🐍 Setting up Python with uv..."
# Install Python 3.12 (Stable, modern)
uv python install 3.12
# Create virtual environment and install deps
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 4. Create data/log directories
mkdir -p logs data

# 5. Configure Systemd Service
echo "⚙️ Configuring systemd service..."

# Update the service file template with real paths
# Point ExecStart to the venv python executable
VENV_PYTHON="$APP_DIR/.venv/bin/python3"

sed -i "s|YOUR_VM_USER|$USER|g" deploy/delta-bot.service
sed -i "s|YOUR_WORKING_DIR|$APP_DIR|g" deploy/delta-bot.service
sed -i "s|YOUR_PYTHON_EXEC|$VENV_PYTHON|g" deploy/delta-bot.service

# Copy and enable service
sudo cp deploy/delta-bot.service /etc/systemd/system/$SERVICE_NAME.service
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# 6. Start the bot
echo "🏁 Starting the trading bot..."
sudo systemctl restart $SERVICE_NAME

echo "--------------------------------------------------"
echo "✅ AUTOMATION COMPLETE!"
echo "--------------------------------------------------"
echo "Status: $(sudo systemctl is-active $SERVICE_NAME)"
echo "Logs:   tail -f logs/trading.log"
echo "--------------------------------------------------"
