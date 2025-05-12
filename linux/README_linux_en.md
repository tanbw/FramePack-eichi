# FramePack-eichi Linux Support (Unofficial)

This directory contains unofficial support scripts for using FramePack-eichi in a Linux environment. These scripts are provided for convenience and are **not officially supported**. Use at your own risk.

## System Requirements

- **OS**: Ubuntu 22.04 LTS recommended (other distributions supporting Python 3.10 should also work)
- **CPU**: 8+ cores of a modern multi-core CPU recommended
- **RAM**: Minimum 16GB, 32GB+ recommended (64GB recommended for complex processing and high resolutions)
- **GPU**: NVIDIA RTX 30XX/40XX/50XX series (8GB+ VRAM)
- **VRAM**: Minimum 8GB (12GB+ recommended)
- **Storage**: 150GB+ of free space (SSD recommended)
- **Required Software**:
  - CUDA Toolkit 12.6
  - Python 3.10.x
  - PyTorch 2.6 with CUDA support

## Included Scripts

- `update.sh` - Script to update the main repository and apply FramePack-eichi files
- `setup_submodule.sh` - Script for initial setup
- `install_linux.sh` - Simple installer for Linux
- `run_endframe_ichi.sh` - Standard version/Japanese execution script
- `run_endframe_ichi_f1.sh` - F1 version/Japanese execution script
- `run_oneframe_ichi.sh` - One-frame inference version/Japanese execution script
- Other language version execution scripts

## Linux Setup Guide (Submodule Method)

### 1. Install Prerequisites

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install basic development tools and libraries
sudo apt install -y git wget ffmpeg libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libopenblas-dev

# Install CUDA Toolkit 12.6
# Note: Follow NVIDIA's official instructions to install CUDA Toolkit
# https://developer.nvidia.com/cuda-12-6-0-download-archive

# Install Python 3.10
sudo apt install -y python3.10 python3.10-venv python3-pip
```

### 2. Clone and Set Up FramePack-eichi

```bash
# Clone FramePack-eichi repository
git clone https://github.com/git-ai-code/FramePack-eichi.git
cd FramePack-eichi

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Set up submodules (automatically downloads original FramePack)
./linux/setup_submodule.sh

# Install PyTorch with CUDA support and dependencies
cd webui/submodules/FramePack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 3. Launch FramePack-eichi

```bash
# Return to FramePack-eichi root directory
cd ~/FramePack-eichi  # Adjust path according to your installation location

# Launch using execution scripts
./linux/run_endframe_ichi_en.sh    # Standard version/English UI
./linux/run_endframe_ichi_en_f1.sh # F1 model version/English UI
./linux/run_oneframe_ichi_en.sh    # One-frame inference version/English UI

# Other language versions
./linux/run_endframe_ichi.sh       # Japanese UI
./linux/run_endframe_ichi_zh-tw.sh # Traditional Chinese UI
```

## Usage

### Setup for Existing Repository

```bash
cd /path/to/FramePack-eichi
./linux/setup_submodule.sh
```

### Update from Original Repository

```bash
cd /path/to/FramePack-eichi
./linux/update.sh
```

### Running the Application

```bash
cd /path/to/FramePack-eichi
./linux/run_endframe_ichi.sh  # Standard version/Japanese
./linux/run_endframe_ichi_f1.sh  # F1 version/Japanese
./linux/run_oneframe_ichi.sh  # One-frame inference version/Japanese
```

## Installing Acceleration Libraries

If you see the following messages when running FramePack, the acceleration libraries are not installed:

```
Xformers is not installed!
Flash Attn is not installed!
Sage Attn is not installed!
```

Installing these libraries can improve processing speed (approximately 30% speedup can be expected).

### Installation Method

Depending on your Python environment, run the following commands:

```bash
# 1. Navigate to the FramePack directory
cd /path/to/FramePack-eichi/webui/submodules/FramePack

# 2. Install required libraries
pip install xformers triton
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install sage-attn==1.0.6

# 3. Restart to verify installation
```

### Installing Acceleration Libraries for Standalone Setup

For standalone setup, install as follows:

```bash
# Ensure your virtual environment is activated
source venv/bin/activate

# Navigate to FramePack directory
cd FramePack

# Install acceleration libraries
pip install xformers triton
pip install packaging ninja
pip install flash-attn --no-build-isolation 
pip install sageattention==1.0.6
```

### Installation Notes

- Only supported with CUDA 12.x (for CUDA 11.x, some libraries need to be built)
- Installing `flash-attn` may be difficult in some environments. In that case, using only Xformers can still improve performance
- Make sure your PyTorch version is 2.0.0 or higher
- The sage-attn package may be renamed to sageattention (specify version 1.0.6)

## Troubleshooting

### "CUDA out of memory" Error

If you encounter memory issues, try the following:

1. Close other applications using the GPU
2. Reduce image size (512x512 range recommended)
3. Reduce batch size
4. Increase the `gpu_memory_preservation` value (higher settings reduce memory usage but also reduce processing speed)

### CUDA Installation and Compatibility Issues

If you see "CUDA is not available" errors or warnings about "switching to CPU execution":

1. Check if CUDA is correctly installed:
   ```bash
   nvidia-smi
   ```

2. Check if PyTorch recognizes CUDA:
   ```python
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```

3. Verify that you're using a supported GPU (RTX 30XX, 40XX, or 50XX series recommended)

4. Check CUDA driver and PyTorch compatibility:
   - Driver compatible with CUDA 12.6
   - PyTorch 2.6 with CUDA support

### Model Loading Failures

If you encounter errors when loading model shards:

1. Verify that models are properly downloaded
2. For first launch, wait for the necessary models (about 30GB) to download automatically
3. Ensure you have sufficient disk space (minimum 150GB recommended)

## Notes

- These scripts are not officially supported
- If you encounter errors related to execution paths, please modify the scripts accordingly
- Complex processing and high-resolution settings increase memory usage (sufficient RAM and high VRAM GPUs recommended)
- If memory leaks occur after extended use, restart the application
- While you can register questions or bug reports as Issues, we cannot guarantee they will be addressed

## References

- Official FramePack: https://github.com/lllyasviel/FramePack
- FramePack-eichi: https://github.com/git-ai-code/FramePack-eichi
- Acceleration Library Installation: https://github.com/lllyasviel/FramePack/issues/138
- CUDA Toolkit: https://developer.nvidia.com/cuda-12-6-0-download-archive