#!/bin/bash
# Submodule Setup Script
# Sets up the original FramePack repository as a submodule
# Note: Unofficial support - no warranty

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to FramePack-eichi root directory (parent of script directory)
cd "$(dirname "$SCRIPT_DIR")"

# Prepare submodule directory
mkdir -p webui/submodules

# Remove existing symbolic link or directory if exists
if [ -e webui/submodules/FramePack ]; then
  echo "Removing existing FramePack directory..."
  rm -rf webui/submodules/FramePack
fi

echo "Initializing submodules..."

# Add original repository as a submodule
# (Not needed if submodule is already configured)
if ! grep -q "\[submodule \"webui/submodules/FramePack\"\]" .gitmodules 2>/dev/null; then
  echo "Adding original FramePack repository as a submodule..."
  git submodule add https://github.com/lllyasviel/FramePack.git webui/submodules/FramePack
else
  echo "Updating submodules..."
  git submodule update --init --recursive
fi

# Run update script to complete initial setup
echo "Applying FramePack-eichi files..."
"$SCRIPT_DIR/update.sh"

echo "Submodule setup completed!"