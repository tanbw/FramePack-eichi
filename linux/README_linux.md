# FramePack-eichi Linuxサポート (非公式)

このディレクトリには、FramePack-eichiをLinux環境で使用するための非公式サポートスクリプトが含まれています。これらのスクリプトは便宜上提供されるものであり、**公式サポート対象外**です。ご利用は自己責任でお願いします。

## システム要件

- **OS**: Ubuntu 22.04 LTS以降が推奨（他のPython 3.10対応ディストリビューションも動作可能）
- **CPU**: 8コア以上の最新のマルチコアCPU推奨
- **RAM**: 最小16GB、推奨32GB以上（複雑な処理や高解像度には64GB推奨）
- **GPU**: NVIDIA RTX 30XX/40XX/50XX シリーズ（8GB VRAM以上）
- **VRAM**: 最小8GB（推奨12GB以上）
- **ストレージ**: 150GB以上の空き容量（SSD推奨）
- **必須ソフトウェア**:
  - CUDA Toolkit 12.6
  - Python 3.10.x
  - CUDAサポート付きPyTorch 2.6

## 含まれるスクリプト

- `update.sh` - 本家リポジトリの更新とFramePack-eichiファイルの上書き適用を行うスクリプト
- `setup_submodule.sh` - 初回セットアップ用スクリプト
- `install_linux.sh` - Linux向け簡易インストーラー
- `run_endframe_ichi.sh` - 無印版/日本語実行スクリプト
- `run_endframe_ichi_f1.sh` - F1版/日本語実行スクリプト
- `run_oneframe_ichi.sh` - 1フレーム推論版/日本語実行スクリプト
- その他言語版実行スクリプト

## Linuxセットアップ手順（サブモジュール）

### 1. 前提条件のインストール

```bash
# システムパッケージの更新
sudo apt update && sudo apt upgrade -y

# 基本的な開発ツールとライブラリのインストール
sudo apt install -y git wget ffmpeg libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libopenblas-dev

# CUDA Toolkit 12.6のインストール
# 注: NVIDIAの公式手順に従ってCUDA Toolkitをインストールしてください
# https://developer.nvidia.com/cuda-12-6-0-download-archive

# Python 3.10のインストール
sudo apt install -y python3.10 python3.10-venv python3-pip
```

### 2. FramePack-eichiのクローンとセットアップ

```bash
# FramePack-eichiリポジトリのクローン
git clone https://github.com/git-ai-code/FramePack-eichi.git
cd FramePack-eichi

# 仮想環境の作成と有効化
python3.10 -m venv venv
source venv/bin/activate

# サブモジュールのセットアップ (本家FramePackを自動的にダウンロード)
./linux/setup_submodule.sh

# CUDAサポート付きPyTorchと依存関係のインストール
cd webui/submodules/FramePack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 3. FramePack-eichiの起動

```bash
# FramePack-eichiのルートディレクトリに戻る
cd ~/FramePack-eichi  # インストール先に合わせてパスを調整してください

# 実行スクリプトを使用して起動
./linux/run_endframe_ichi.sh       # 標準版/日本語UI
./linux/run_endframe_ichi_f1.sh    # F1モデル版/日本語UI
./linux/run_oneframe_ichi.sh       # 1フレーム推論版/日本語UI

# 他の言語バージョン
./linux/run_endframe_ichi_en.sh    # 英語UI
./linux/run_endframe_ichi_zh-tw.sh # 繁体字中国語UI
```

## 使い方

### 既存リポジトリのセットアップ

```bash
cd /path/to/FramePack-eichi
./linux/setup_submodule.sh
```

### 本家の更新反映

```bash
cd /path/to/FramePack-eichi
./linux/update.sh
```

### アプリケーション実行

```bash
cd /path/to/FramePack-eichi
./linux/run_endframe_ichi.sh  # 無印版/日本語
./linux/run_endframe_ichi_f1.sh  # F1版/日本語
./linux/run_oneframe_ichi.sh  # 1フレーム推論版/日本語
```

## 高速化ライブラリのインストール

FramePackの実行時に以下のメッセージが表示される場合、高速化ライブラリがインストールされていません：

```
Xformers is not installed!
Flash Attn is not installed!
Sage Attn is not installed!
```

これらのライブラリをインストールすると処理速度が向上します（約30%程度の高速化が期待できます）。

### インストール方法

お使いのPython環境に応じて以下のコマンドを実行してください：

```bash
# 1. FramePackのディレクトリに移動
cd /path/to/FramePack-eichi/webui/submodules/FramePack

# 2. 必要なライブラリをインストール
pip install xformers triton
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install sage-attn==1.0.6

# 3. 再起動してインストールを確認
```

### スタンドアロン環境での高速化ライブラリインストール

スタンドアロンセットアップの場合は、以下のようにインストールしてください：

```bash
# 仮想環境が有効化されていることを確認
source venv/bin/activate

# FramePackディレクトリに移動
cd FramePack

# 高速化ライブラリをインストール
pip install xformers triton
pip install packaging ninja
pip install flash-attn --no-build-isolation 
pip install sageattention==1.0.6
```

### インストール時の注意点

- CUDA 12.xを使用している場合のみ対応（CUDA 11.xでは一部ライブラリをビルドする必要があります）
- 環境によっては`flash-attn`のインストールが難しい場合があります。その場合、Xformersだけでも性能向上が見込めます
- PyTorchのバージョンが2.0.0以上であることを確認してください
- sage-attnパッケージはsageattentionという名前に変更されている場合があります（バージョン1.0.6を指定）

## トラブルシューティング

### エラー「CUDA out of memory」が発生する場合

メモリ不足の場合は、以下の対策を試してください：

1. 他のGPUを使用するアプリケーションを終了する
2. 画像サイズを小さくする（512x512付近を推奨）
3. バッチサイズを減らす
4. `gpu_memory_preservation`値を増やす（設定値を高くすると使用メモリは減りますが処理速度も低下）

### CUDAインストールと互換性の問題

「CUDAは利用できません」エラーや「CPU実行に切り替えます」という警告が表示される場合：

1. CUDAを正しくインストールしているか確認：
   ```bash
   nvidia-smi
   ```

2. PyTorchがCUDAを認識しているか確認：
   ```python
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```

3. サポートされているGPU（RTX 30XX、40XX、または50XXシリーズ推奨）を使用していることを確認

4. CUDAドライバーとPyTorchの互換性を確認：
   - CUDA 12.6対応のドライバー
   - CUDAサポート付きPyTorch 2.6

### モデルロードの失敗

モデルシャードのロード時にエラーが発生する場合：

1. モデルが適切にダウンロードされていることを確認
2. 初回起動時は、必要なモデル（約30GB）が自動的にダウンロードされるのを待つ
3. ディスク容量が十分にあることを確認（最低150GB推奨）

## 注意事項

- これらのスクリプトは公式サポート対象外です
- 実行パスの関係でエラーが発生する場合は、スクリプトを適宜修正してください
- 複雑な処理や高解像度設定ではメモリ使用量が増加します（十分なRAMと高VRAMのGPUを推奨）
- 長時間使用後にメモリリークが発生した場合は、アプリケーションを再起動してください
- ご質問やバグ報告はIssueとして登録していただけますが、対応を約束するものではありません

## 参考情報

- 公式FramePack: https://github.com/lllyasviel/FramePack
- FramePack-eichi: https://github.com/git-ai-code/FramePack-eichi
- 高速化ライブラリのインストール: https://github.com/lllyasviel/FramePack/issues/138
- CUDAツールキット: https://developer.nvidia.com/cuda-12-6-0-download-archive