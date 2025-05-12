# FramePack-eichi Linux 支援 (非官方)

此目錄包含在 Linux 環境中使用 FramePack-eichi 的非官方支援腳本。這些腳本僅為便利提供，**不提供官方支援**。使用時風險自負。

## 系統需求

- **作業系統**: 推薦 Ubuntu 22.04 LTS（其他支援 Python 3.10 的發行版也應該可以運作）
- **CPU**: 推薦 8 核心以上的現代多核心 CPU
- **RAM**: 最低 16GB，推薦 32GB 以上（複雜處理和高解析度推薦 64GB）
- **GPU**: NVIDIA RTX 30XX/40XX/50XX 系列（8GB 以上 VRAM）
- **VRAM**: 最低 8GB（推薦 12GB 以上）
- **儲存空間**: 150GB 以上的可用空間（推薦 SSD）
- **必要軟體**:
  - CUDA Toolkit 12.6
  - Python 3.10.x
  - 支援 CUDA 的 PyTorch 2.6

## 包含的腳本

- `update.sh` - 更新主要儲存庫並應用 FramePack-eichi 檔案的腳本
- `setup_submodule.sh` - 初始設置用腳本
- `install_linux.sh` - Linux 簡易安裝程式
- `run_endframe_ichi.sh` - 標準版/日文執行腳本
- `run_endframe_ichi_f1.sh` - F1版/日文執行腳本
- `run_oneframe_ichi.sh` - 單幀推論版/日文執行腳本
- 其他語言版本執行腳本

## Linux 設置指南（子模組方法）

### 1. 安裝必要條件

```bash
# 更新系統套件
sudo apt update && sudo apt upgrade -y

# 安裝基本開發工具和庫
sudo apt install -y git wget ffmpeg libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libopenblas-dev

# 安裝 CUDA Toolkit 12.6
# 注意：請按照 NVIDIA 的官方說明安裝 CUDA Toolkit
# https://developer.nvidia.com/cuda-12-6-0-download-archive

# 安裝 Python 3.10
sudo apt install -y python3.10 python3.10-venv python3-pip
```

### 2. 複製並設置 FramePack-eichi

```bash
# 複製 FramePack-eichi 儲存庫
git clone https://github.com/git-ai-code/FramePack-eichi.git
cd FramePack-eichi

# 創建並啟用虛擬環境
python3.10 -m venv venv
source venv/bin/activate

# 設置子模組（自動下載原始 FramePack）
./linux/setup_submodule.sh

# 安裝帶 CUDA 支援的 PyTorch 和依賴項
cd webui/submodules/FramePack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 3. 啟動 FramePack-eichi

```bash
# 返回 FramePack-eichi 根目錄
cd ~/FramePack-eichi  # 根據您的安裝位置調整路徑

# 使用執行腳本啟動
./linux/run_endframe_ichi_zh-tw.sh       # 標準版/繁體中文介面
./linux/run_endframe_ichi_zh-tw_f1.sh    # F1 模型版本/繁體中文介面
./linux/run_oneframe_ichi_zh-tw.sh       # 單幀推論版本/繁體中文介面

# 其他語言版本
./linux/run_endframe_ichi.sh       # 日文介面
./linux/run_endframe_ichi_en.sh    # 英文介面
```

## 使用方式

### 現有存儲庫的設置

```bash
cd /path/to/FramePack-eichi
./linux/setup_submodule.sh
```

### 更新原始儲存庫

```bash
cd /path/to/FramePack-eichi
./linux/update.sh
```

### 執行應用程式

```bash
cd /path/to/FramePack-eichi
./linux/run_endframe_ichi.sh  # 標準版/日文
./linux/run_endframe_ichi_f1.sh  # F1版/日文
./linux/run_oneframe_ichi.sh  # 單幀推論版/日文
```

## 安裝加速庫

如果執行 FramePack 時看到以下訊息，表示尚未安裝加速庫：

```
Xformers is not installed!
Flash Attn is not installed!
Sage Attn is not installed!
```

安裝這些庫可以提高處理速度（預期可達約 30% 的加速）。

### 安裝方法

根據您的 Python 環境，執行以下命令：

```bash
# 1. 導航到 FramePack 目錄
cd /path/to/FramePack-eichi/webui/submodules/FramePack

# 2. 安裝必要庫
pip install xformers triton
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install sage-attn==1.0.6

# 3. 重新啟動以驗證安裝
```

### 為獨立設置安裝加速庫

對於獨立設置，請按照以下方式安裝：

```bash
# 確保您的虛擬環境已啟用
source venv/bin/activate

# 導航到 FramePack 目錄
cd FramePack

# 安裝加速庫
pip install xformers triton
pip install packaging ninja
pip install flash-attn --no-build-isolation 
pip install sageattention==1.0.6
```

### 安裝注意事項

- 僅支援 CUDA 12.x（對於 CUDA 11.x，需要編譯某些庫）
- 在某些環境中安裝 `flash-attn` 可能有困難。在這種情況下，僅使用 Xformers 仍然可以提高性能
- 確保您的 PyTorch 版本為 2.0.0 或更高
- sage-attn 包可能被重命名為 sageattention（指定版本 1.0.6）

## 故障排除

### "CUDA out of memory" 錯誤

如果遇到記憶體問題，請嘗試以下方法：

1. 關閉使用 GPU 的其他應用程式
2. 減小圖像大小（建議 512x512 左右）
3. 減少批次大小
4. 增加 `gpu_memory_preservation` 值（較高的設定會減少記憶體使用量，但也會降低處理速度）

### CUDA 安裝和相容性問題

如果看到 "CUDA 不可用" 錯誤或關於 "切換到 CPU 執行" 的警告：

1. 檢查 CUDA 是否正確安裝：
   ```bash
   nvidia-smi
   ```

2. 檢查 PyTorch 是否識別 CUDA：
   ```python
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```

3. 確認您使用的是支援的 GPU（建議 RTX 30XX、40XX 或 50XX 系列）

4. 檢查 CUDA 驅動程式和 PyTorch 相容性：
   - 與 CUDA 12.6 相容的驅動程式
   - 支援 CUDA 的 PyTorch 2.6

### 模型載入失敗

如果遇到載入模型分片時的錯誤：

1. 確認模型已正確下載
2. 對於首次啟動，等待必要的模型（約 30GB）自動下載
3. 確保有足夠的磁碟空間（建議最少 150GB）

## 注意事項

- 這些腳本不受官方支援
- 如果遇到與執行路徑相關的錯誤，請相應修改腳本
- 複雜處理和高解析度設定會增加記憶體使用量（建議足夠的 RAM 和高 VRAM GPU）
- 如果長時間使用後出現記憶體洩漏，請重新啟動應用程式
- 雖然您可以將問題或錯誤報告註冊為 Issues，但我們不保證會解決這些問題

## 參考資訊

- 官方 FramePack：https://github.com/lllyasviel/FramePack
- FramePack-eichi：https://github.com/git-ai-code/FramePack-eichi
- 加速庫安裝：https://github.com/lllyasviel/FramePack/issues/138
- CUDA Toolkit：https://developer.nvidia.com/cuda-12-6-0-download-archive