#!/bin/bash
# FramePack-eichi Update Script
# Updates the original repository and applies FramePack-eichi features
# Note: Unofficial support - no warranty

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to FramePack-eichi root directory (parent of script directory)
cd "$(dirname "$SCRIPT_DIR")"

echo "=== FramePack-eichi Update ==="
echo "1. Updating submodules..."

# Update submodules
if [ -d "webui/submodules/FramePack/.git" ]; then
  cd webui/submodules/FramePack
  git pull origin main
  # Return to root directory
  cd "$(dirname "$SCRIPT_DIR")"
else
  # Initialize submodules if they don't exist
  echo "Initializing submodules..."
  git submodule update --init --recursive
fi

echo "2. Overwriting FramePack-eichi files..."

# Copy eichi-specific files
mkdir -p webui/submodules/FramePack/webui/eichi_utils
cp -r webui/eichi_utils/* webui/submodules/FramePack/webui/eichi_utils/

mkdir -p webui/submodules/FramePack/webui/lora_utils
cp -r webui/lora_utils/* webui/submodules/FramePack/webui/lora_utils/

mkdir -p webui/submodules/FramePack/webui/diffusers_helper
cp -r webui/diffusers_helper/* webui/submodules/FramePack/webui/diffusers_helper/

mkdir -p webui/submodules/FramePack/webui/locales
cp -r webui/locales/* webui/submodules/FramePack/webui/locales/

# Copy main script files
cp webui/endframe_ichi.py webui/submodules/FramePack/webui/
cp webui/endframe_ichi_f1.py webui/submodules/FramePack/webui/
cp webui/oneframe_ichi.py webui/submodules/FramePack/webui/

# Create settings directory and copy settings files
mkdir -p webui/submodules/FramePack/webui/settings
if [ -d "webui/settings" ]; then
  cp -n webui/settings/* webui/submodules/FramePack/webui/settings/ 2>/dev/null || true
fi

echo "3. Setting permissions for execution scripts..."
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

echo "=== Update Complete! ==="
echo "FramePack-eichi files have been successfully applied."