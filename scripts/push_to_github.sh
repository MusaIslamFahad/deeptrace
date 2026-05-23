#!/bin/bash
# DeepTrace — GitHub Push Script
# Run this from inside your deeptrace/ folder
# Usage: bash scripts/push_to_github.sh YOUR_GITHUB_USERNAME

set -e

USERNAME=$1

if [ -z "$USERNAME" ]; then
  echo "Usage: bash scripts/push_to_github.sh YOUR_GITHUB_USERNAME"
  exit 1
fi

REPO_URL="https://github.com/$USERNAME/deeptrace.git"

echo ""
echo "========================================"
echo "  DeepTrace — GitHub Push"
echo "========================================"
echo ""
echo "GitHub username : $USERNAME"
echo "Repository URL  : $REPO_URL"
echo ""
echo "Make sure you have already:"
echo "  1. Created the repo at github.com/$USERNAME/deeptrace"
echo "  2. Copied all deploy config files into this project"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."

# Init git if not already done
if [ ! -d ".git" ]; then
  echo ""
  echo "[1/5] Initialising git..."
  git init
else
  echo "[1/5] Git already initialised — skipping"
fi

# Stage all files
echo "[2/5] Staging all files..."
git add .

# Commit
echo "[3/5] Creating commit..."
git commit -m "feat: initial DeepTrace project — AI image provenance platform" \
  --allow-empty 2>/dev/null || echo "  Nothing new to commit"

# Set branch to main
echo "[4/5] Setting branch to main..."
git branch -M main

# Add remote (replace if already exists)
echo "[5/5] Pushing to GitHub..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git push -u origin main

echo ""
echo "========================================"
echo "  Done! Your code is now on GitHub."
echo "  Visit: https://github.com/$USERNAME/deeptrace"
echo "========================================"
echo ""
echo "Next step: Deploy to Render"
echo "  -> Follow DEPLOYMENT_GUIDE.md Part 2"
echo ""
