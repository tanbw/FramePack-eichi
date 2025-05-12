#!/bin/bash
# FramePack-eichi Linux Installation Script
# Note: Unofficial support - no warranty

set -e

INSTALL_DIR="${HOME}/FramePack-eichi"
REPO_URL="https://github.com/git-ai-code/FramePack-eichi.git"
PYTHON_VERSION="3.10"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dir=*)
      INSTALL_DIR="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --dir=PATH  Specify installation directory (default: $INSTALL_DIR)"
      echo "  --help      Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=== FramePack-eichi Installer ==="
echo "Installation destination: $INSTALL_DIR"

# Check and install required packages
echo "1. Checking dependencies..."

# Check Python installation
if ! command -v python$PYTHON_VERSION &> /dev/null && ! command -v python3 &> /dev/null; then
  echo "Installing Python..."
  if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
  elif command -v yum &> /dev/null; then
    sudo yum install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-devel
  elif command -v dnf &> /dev/null; then
    sudo dnf install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-devel
  elif command -v pacman &> /dev/null; then
    sudo pacman -S python
  elif command -v brew &> /dev/null; then
    brew install python@${PYTHON_VERSION}
  else
    echo "Warning: System not supported for automatic installation. Please install Python manually."
  fi
fi

# Check Git installation
if ! command -v git &> /dev/null; then
  echo "Installing Git..."
  if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y git
  elif command -v yum &> /dev/null; then
    sudo yum install -y git
  elif command -v dnf &> /dev/null; then
    sudo dnf install -y git
  elif command -v pacman &> /dev/null; then
    sudo pacman -S git
  elif command -v brew &> /dev/null; then
    brew install git
  else
    echo "Warning: System not supported for automatic installation. Please install Git manually."
  fi
fi

# Create and navigate to directory
echo "2. Preparing repository..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone or update repository
if [ -d ".git" ]; then
  echo "Updating existing repository..."
  git pull
else
  echo "Cloning repository..."
  git clone "$REPO_URL" .
fi

# Initialize and update submodules
echo "3. Initializing submodules..."
git submodule init
git submodule update

# Create execution scripts
echo "4. Creating execution scripts..."

# Copy Linux scripts
echo "Copying Linux scripts..."
mkdir -p "$INSTALL_DIR/linux"
cp -r "$INSTALL_DIR/linux"/*.sh "$INSTALL_DIR/linux/"
chmod +x "$INSTALL_DIR/linux"/*.sh

# Apply eichi-specific files
echo "5. Applying FramePack-eichi files..."
chmod +x "$INSTALL_DIR/linux/update.sh"
"$INSTALL_DIR/linux/update.sh"

echo "=== Installation Complete! ==="
echo "You can run FramePack-eichi with the following commands:"
echo "cd $INSTALL_DIR && ./linux/run_endframe_ichi.sh"
echo ""
echo "For F1 model version:"
echo "cd $INSTALL_DIR && ./linux/run_endframe_ichi_f1.sh"
echo ""
echo "For English UI:"
echo "cd $INSTALL_DIR && ./linux/run_endframe_ichi_en.sh"
echo ""
echo "Note: These scripts are unofficial and you use them at your own risk."