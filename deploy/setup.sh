#!/bin/bash
# Fully Automated Deployment Script for Delta Exchange Bot
# This script is intended to be run on an Ubuntu/Debian VM.

set -e # Exit on error

echo "🚀 Starting Full Automation Setup..."

# 0. Get current user and dir
USER=$(whoami)
APP_DIR=$(pwd)
SERVICE_NAME="delta-bot"

echo "👤 Running as user: $USER"
echo "📂 App directory: $APP_DIR"

# 1. Update system and install Python
echo "📦 Updating system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential

# Optional: Try to install libta-lib if available, but don't fail if not
# We have built-in Python fallbacks for all indicators in technical_analysis.py
sudo apt-get install -y libta-lib0 || echo "⚠️ libta-lib0 not found, using Python math fallbacks."

# 2. Install Python requirements
echo "🐍 Installing Python requirements..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# 3. Create data/log directories
mkdir -p logs data

# 4. Configure Systemd Service
echo "⚙️ Configuring systemd service..."

# Update the service file template with real paths
sed -i "s|YOUR_VM_USER|$USER|g" deploy/delta-bot.service
sed -i "s|/home/$USER/delta-exchange|$APP_DIR|g" deploy/delta-bot.service

# Copy and enable service
sudo cp deploy/delta-bot.service /etc/systemd/system/$SERVICE_NAME.service
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# 5. Start the bot
echo "🏁 Starting the trading bot..."
sudo systemctl restart $SERVICE_NAME

echo "--------------------------------------------------"
echo "✅ AUTOMATION COMPLETE!"
echo "--------------------------------------------------"
echo "Status: $(sudo systemctl is-active $SERVICE_NAME)"
echo "Logs:   tail -f logs/trading.log"
echo "--------------------------------------------------"
