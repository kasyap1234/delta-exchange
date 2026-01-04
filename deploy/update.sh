#!/bin/bash
# Update script for Delta Trading Bot
# Run this on your GCP VM to pull the latest changes and restart the bot.

set -e

echo "🔄 Updating Delta Trading Bot..."

# 1. Check if it's a git repo
if [ ! -d .git ]; then
    echo "⚠️ Not a git repository. Suggestion: Clone the repo properly for easier updates."
    echo "For now, please manually upload the changed files or use: git clone <repo_url>"
    exit 1
fi

# 2. Pull changes (User may need to enter credentials)
echo "📥 Pulling latest code from GitHub..."
git pull origin main

# 3. Restart the service
echo "🔃 Restarting delta-bot service..."
sudo systemctl restart delta-bot

# 4. Check status
echo "📊 Current Status:"
sudo systemctl status delta-bot --no-pager | grep "Active:"

echo "✅ Update complete! Viewing logs (Ctrl+C to stop)..."
tail -f logs/trading.log
