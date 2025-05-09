import os
import sys
sys.path.append(os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './submodules/FramePack'))))

from diffusers_helper.hf_login import login

import os
import random  # ランダムシード生成用
import time
import traceback  # デバッグログ出力用
import yaml
import argparse
import json
import glob
from PIL import Image

# PNGメタデータ処理モジュールのインポート
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from eichi_utils.png_metadata import (
    embed_metadata_to_png, extract_metadata_from_png, extract_metadata_from_numpy_array,
    PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY, SECTION_NUMBER_KEY
)

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--lang", type=str, default='ja', help="Language: ja, zh-tw, en")
args = parser.parse_args()

# Load translations from JSON files
from locales.i18n_extended import (set_lang, translate)
set_lang(args.lang)

# サーバーがすでに実行中かチェック
import socket
def is_port_in_use(port):
    """指定ポートが使用中かどうかを確認"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except:
        # エラーが発生した場合はポートが使用されていないと判断
        return False

# キャッシュ制御設定
# ローカルファイルの優先的利用を無効化し安定性を向上
use_cache_files = True  # ファイルキャッシュは使用
first_run = True  # 通常は初回起動として扱う

# ポートが使用中の場合は警告を表示
if is_port_in_use(args.port):
    print(translate("警告: ポート {0} はすでに使用されています。他のインスタンスが実行中かもしれません。").format(args.port))
    print(translate("5秒後に処理を続行します..."))
    first_run = False  # 初回実行ではない
    time.sleep(5)  # 5秒待機して続行

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
    print(translate("HF_HOMEを設定: {0}").format(os.environ['HF_HOME']))
else:
    print(translate("既存のHF_HOMEを使用: {0}").format(os.environ['HF_HOME']))

# LoRAサポートの確認
has_lora_support = False
has_fp8_support = False
try:
    import lora_utils
    from lora_utils.fp8_optimization_utils import check_fp8_support, apply_fp8_monkey_patch
    
    has_lora_support = True
    # FP8サポート確認
    has_e4m3, has_e5m2, has_scaled_mm = check_fp8_support()
    has_fp8_support = has_e4m3 and has_e5m2
    
    if has_fp8_support:
        print(translate("LoRAサポートとFP8最適化が有効です"))
        print(translate("FP8最適化は実際のモデルロード時に適用されます"))
        # FP8最適化のモンキーパッチはLoRA適用時に使用される
        # apply_fp8_monkey_patch()は引数が必要なため、ここでは呼び出さない
    else:
        print(translate("LoRAサポートが有効です (FP8最適化はサポートされていません)"))
except ImportError:
    print(translate("LoRAサポートが無効です（lora_utilsモジュールがインストールされていません）"))

# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_settings_file_path,
    get_output_folder_path,
    initialize_settings,
    load_settings,
    save_settings,
    open_output_folder
)

import gradio as gr
from eichi_utils.ui_styles import get_app_css
import torch
import einops
import safetensors.torch as sf
import numpy as np
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from eichi_utils.ui_styles import get_app_css
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

from eichi_utils.transformer_manager import TransformerManager
from eichi_utils.text_encoder_manager import TextEncoderManager

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 100

print(translate('Free VRAM {0} GB').format(free_mem_gb))
print(translate('High-VRAM Mode: {0}').format(high_vram))

# グローバルなモデル状態管理インスタンスを作成（モデルは実際に使用するまでロードしない）
transformer_manager = TransformerManager(device=gpu, high_vram_mode=high_vram, use_f1_model=False)
text_encoder_manager = TextEncoderManager(device=gpu, high_vram_mode=high_vram)

# LoRAの状態を確認
def reload_transformer_if_needed():
    """transformerモデルが必要に応じてリロードする"""
    try:
        # 既存のensure_transformer_stateメソッドを使用する
        if hasattr(transformer_manager, 'ensure_transformer_state'):
            return transformer_manager.ensure_transformer_state()
        # 互換性のために古い方法も維持
        elif hasattr(transformer_manager, '_needs_reload') and hasattr(transformer_manager, '_reload_transformer'):
            if transformer_manager._needs_reload():
                return transformer_manager._reload_transformer()
            return True
        return False
    except Exception as e:
        print(f"transformerリロードエラー: {e}")
        traceback.print_exc()
        return False

# 遅延ロード方式に変更 - 起動時にはtokenizerのみロードする
try:
    # tokenizerのロードは起動時から行う
    try:
        print(translate("tokenizer, tokenizer_2のロードを開始します..."))
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        print(translate("tokenizer, tokenizer_2のロードが完了しました"))
    except Exception as e:
        print(translate("tokenizer, tokenizer_2のロードに失敗しました: {0}").format(e))
        traceback.print_exc()
        print(translate("5秒間待機後に再試行します..."))
        time.sleep(5)
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    
    # feature_extractorは軽量なのでここでロード
    try:
        print(translate("feature_extractorのロードを開始します..."))
        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        print(translate("feature_extractorのロードが完了しました"))
    except Exception as e:
        print(translate("feature_extractorのロードに失敗しました: {0}").format(e))
        print(translate("再試行します..."))
        time.sleep(2)
        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    
    # 他の重いモデルは遅延ロード方式に変更
    # 変数の初期化だけ行い、実際のロードはworker関数内で行う
    vae = None
    text_encoder = None
    text_encoder_2 = None
    transformer = None
    image_encoder = None
    
    # transformerダウンロードの確保だけは起動時に
    try:
        # モデルのダウンロードを確保（実際のロードはまだ行わない）
        print(translate("モデルのダウンロードを確保します..."))
        transformer_manager.ensure_download_models()
        print(translate("モデルのダウンロードを確保しました"))
    except Exception as e:
        print(translate("モデルダウンロード確保エラー: {0}").format(e))
        traceback.print_exc()
    
except Exception as e:
    print(translate("初期化エラー: {0}").format(e))
    print(translate("プログラムを終了します..."))
    import sys
    sys.exit(1)

# モデル設定のデフォルト値を定義（実際のモデルロードはworker関数内で行う）
def setup_vae_if_loaded():
    global vae
    if vae is not None:
        vae.eval()
        if not high_vram:
            vae.enable_slicing()
            vae.enable_tiling()
        vae.to(dtype=torch.float16)
        vae.requires_grad_(False)
        if high_vram:
            vae.to(gpu)
            
def setup_image_encoder_if_loaded():
    global image_encoder
    if image_encoder is not None:
        image_encoder.eval()
        image_encoder.to(dtype=torch.float16)
        image_encoder.requires_grad_(False)
        if high_vram:
            image_encoder.to(gpu)

stream = AsyncStream()

# フォルダ構造を先に定義
webui_folder = os.path.dirname(os.path.abspath(__file__))
base_path = webui_folder  # endframe_ichiとの互換性のため

# 設定保存用フォルダの設定
settings_folder = os.path.join(webui_folder, 'settings')
os.makedirs(settings_folder, exist_ok=True)

# 設定から出力フォルダを取得
app_settings = load_settings()
output_folder_name = app_settings.get('output_folder', 'outputs')
print(translate("設定から出力フォルダを読み込み: {0}").format(output_folder_name))

# 出力フォルダのフルパスを生成
outputs_folder = get_output_folder_path(output_folder_name)
os.makedirs(outputs_folder, exist_ok=True)

# グローバル変数
g_frame_size_setting = "1フレーム"
batch_stopped = False  # バッチ処理中断フラグ

# ワーカー関数
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, 
           gpu_memory_preservation, use_teacache, lora_files=None, lora_files2=None, lora_scales_text="0.8,0.8", 
           output_dir=None, use_lora=False, fp8_optimization=False, resolution=640,
           latent_window_size=9, latent_index=0, use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True):
    
    # モデル変数をグローバルとして宣言（遅延ロード用）
    global vae, text_encoder, text_encoder_2, transformer, image_encoder
    
    job_id = generate_timestamp()
    
    # 1フレームモード固有の設定
    total_latent_sections = 1  # 1セクションに固定
    frame_count = 1  # 1フレームモード
    
    # 詳細設定のログ出力
    print(translate("\n[詳細設定]"))
    print(translate("RoPE値 (latent_window_size): {0}").format(latent_window_size))
    print(translate("レイテントインデックス: {0}").format(latent_index))
    print(translate("clean_latents_2xを使用: {0}").format(use_clean_latents_2x))
    print(translate("clean_latents_4xを使用: {0}").format(use_clean_latents_4x))
    print(translate("clean_latents_postを使用: {0}").format(use_clean_latents_post))
    
    # 出力フォルダの設定
    if output_dir:
        outputs_folder = output_dir
    else:
        # 出力フォルダはwebui内のoutputsに固定
        outputs_folder = os.path.join(webui_folder, 'outputs')
    
    os.makedirs(outputs_folder, exist_ok=True)
    
    # プログレスバーの初期化
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    
    # モデルや中間ファイルなどのキャッシュ利用フラグ
    use_cached_files = use_cache_files
    
    try:
        # LoRA 設定 - endframe_ichiと同じ処理
        current_lora_paths = []
        current_lora_scales = []
        
        if use_lora and has_lora_support:
            print(translate("\u25c6 LoRA情報: use_lora = {0}, has_lora_support = {1}").format(use_lora, has_lora_support))
            
            # 全LoRAファイルを収集
            all_lora_files = []
            
            # 1つ目のLoRAファイルを処理
            if lora_files is not None:
                if isinstance(lora_files, list):
                    all_lora_files.extend(lora_files)
                else:
                    all_lora_files.append(lora_files)
            
            # 2つ目のLoRAファイルを処理
            if lora_files2 is not None:
                if isinstance(lora_files2, list):
                    all_lora_files.extend(lora_files2)
                else:
                    all_lora_files.append(lora_files2)
            
            # 有効なLoRAファイルパスを抽出
            for lora_file in all_lora_files:
                try:
                    if hasattr(lora_file, 'name'):
                        current_lora_paths.append(lora_file.name)
                        print(translate("\u25c6 LoRAファイル: {0}").format(os.path.basename(lora_file.name)))
                    elif isinstance(lora_file, dict) and 'name' in lora_file:
                        current_lora_paths.append(lora_file['name'])
                        print(translate("\u25c6 LoRAファイル: {0}").format(os.path.basename(lora_file['name'])))
                    elif isinstance(lora_file, str):
                        current_lora_paths.append(lora_file)
                        print(translate("\u25c6 LoRAファイル: {0}").format(os.path.basename(lora_file)))
                except Exception as e:
                    print(translate("[WARN] LoRAファイル処理中のエラー: {0}").format(e))
                    
            # スケール値を処理
            if current_lora_paths:  # LoRAパスがある場合のみ解析
                try:
                    scales_text = lora_scales_text.strip()
                    scales = [float(scale.strip()) for scale in scales_text.split(',') if scale.strip()]
                    current_lora_scales = scales
                    
                    if len(current_lora_scales) < len(current_lora_paths):
                        print(translate("[INFO] LoRAスケールの数が不足しているため、デフォルト値で補完します"))
                        current_lora_scales.extend([0.8] * (len(current_lora_paths) - len(current_lora_scales)))
                    elif len(current_lora_scales) > len(current_lora_paths):
                        print(translate("[INFO] LoRAスケールの数が多すぎるため、不要なものを切り捨てます"))
                        current_lora_scales = current_lora_scales[:len(current_lora_paths)]
                        
                    for i, (path, scale) in enumerate(zip(current_lora_paths, current_lora_scales)):
                        print(translate("\u25c6 LoRA {0}: {1} (スケール: {2})").format(i+1, os.path.basename(path), scale))
                except Exception as e:
                    print(translate("[ERROR] LoRAスケール解析エラー: {0}").format(e))
                    # デフォルト値で埋める
                    current_lora_scales = [0.8] * len(current_lora_paths)
                    for i, (path, scale) in enumerate(zip(current_lora_paths, current_lora_scales)):
                        print(translate("\u25c6 LoRA {0}: {1} (デフォルトスケール: {2})").format(i+1, os.path.basename(path), scale))
        
        # LoRA設定を更新（リロードは行わない）
        print(translate("\n[INFO] LoRA設定を更新します："))
        print(translate("  - LoRAパス: {0}").format(current_lora_paths))
        print(translate("  - LoRAスケール: {0}").format(current_lora_scales))
        print(translate("  - FP8最適化: {0}").format(fp8_optimization))
        print(translate("  - 高VRAM: {0}").format(high_vram))
        
        # このポイントで適切な設定を渡す
        transformer_manager.set_next_settings(
            lora_paths=current_lora_paths,
            lora_scales=current_lora_scales,
            fp8_enabled=fp8_optimization,
            high_vram_mode=high_vram
        )
        
        # セクション処理開始前にtransformerの状態を確認
        print(translate("\nLoRA適用前のtransformer状態チェック..."))
        try:
            # transformerの状態を確認し、必要に応じてリロード
            if not transformer_manager.ensure_transformer_state():
                raise Exception(translate("transformer状態の確認に失敗しました"))
                
            # 最新のtransformerインスタンスを取得
            transformer = transformer_manager.get_transformer()
            print(translate("transformer状態チェック完了"))
        except Exception as e:
            print(translate("transformerのリロードに失敗しました: {0}").format(e))
            traceback.print_exc()
            raise Exception(translate("transformerのリロードに失敗しました"))
        
        # 入力画像の処理
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
        
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        
        # 入力画像は必要な場合のみ保存（デバッグ用）
        # Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}_input.png'))
        
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # VAE エンコーディング
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        
        try:
            # エンコード前のメモリ状態を記録
            free_mem_before_encode = get_cuda_free_memory_gb(gpu)
            print(translate("\nVAEエンコード前の空きVRAM: {0} GB").format(free_mem_before_encode))
            
            # VAEモデルのロード（未ロードの場合）
            if vae is None:
                print(translate("\nVAEモデルを初めてロードします..."))
                try:
                    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                    setup_vae_if_loaded()  # VAEの設定を適用
                    print(translate("VAEモデルのロードが完了しました"))
                except Exception as e:
                    print(translate("VAEモデルのロードに失敗しました: {0}").format(e))
                    traceback.print_exc()
                    print(translate("5秒間待機後に再試行します..."))
                    time.sleep(5)
                    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                    setup_vae_if_loaded()  # VAEの設定を適用
            
            # ハイVRAM以外では明示的にモデルをGPUにロード
            if not high_vram:
                print(translate("\nVAEモデルをGPUにロード..."))
                load_model_as_complete(vae, target_device=gpu)
            
            # VAEエンコード実行
            with torch.no_grad():  # 明示的にno_gradコンテキストを使用
                # 効率的な処理のために入力をGPUで処理
                input_image_gpu = input_image_pt.to(gpu)
                start_latent = vae_encode(input_image_gpu, vae)
                
                # 入力をCPUに戻す
                del input_image_gpu
                torch.cuda.empty_cache()
            
            # ローVRAMモードでは使用後すぐにCPUに戻す
            if not high_vram:
                vae.to('cpu')
                
                # メモリ状態をログ
                free_mem_after_encode = get_cuda_free_memory_gb(gpu)
                print(translate("\nVAEエンコード後の空きVRAM: {0} GB").format(free_mem_after_encode))
                print(translate("\nVAEエンコードで使用したVRAM: {0} GB").format(free_mem_before_encode - free_mem_after_encode))
                
                # メモリクリーンアップ
                torch.cuda.empty_cache()
        except Exception as e:
            print(translate("VAEエンコードエラー: {0}").format(e))
            
            # エラー発生時のメモリ解放
            if 'input_image_gpu' in locals():
                del input_image_gpu
            torch.cuda.empty_cache()
            
            raise e
        
        # CLIP Vision エンコーディング
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        
        try:
            # 画像エンコーダのロード（未ロードの場合）
            if image_encoder is None:
                print(translate("\n画像エンコーダを初めてロードします..."))
                try:
                    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
                    setup_image_encoder_if_loaded()  # 画像エンコーダの設定を適用
                    print(translate("画像エンコーダのロードが完了しました"))
                except Exception as e:
                    print(translate("画像エンコーダのロードに失敗しました: {0}").format(e))
                    print(translate("再試行します..."))
                    time.sleep(2)
                    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
                    setup_image_encoder_if_loaded()  # 画像エンコーダの設定を適用
            
            if not high_vram:
                print(translate("\n画像エンコーダをGPUにロード..."))
                load_model_as_complete(image_encoder, target_device=gpu)
            
            # CLIP Vision エンコード実行
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            
            # ローVRAMモードでは使用後すぐにCPUに戻す
            if not high_vram:
                image_encoder.to('cpu')
                
                # メモリ状態をログ
                free_mem_gb = get_cuda_free_memory_gb(gpu)
                print(translate("\nCLIP Vision エンコード後の空きVRAM {0} GB").format(free_mem_gb))
                
                # メモリクリーンアップ
                torch.cuda.empty_cache()
        except Exception as e:
            print(translate("CLIP Vision エンコードエラー: {0}").format(e))
            raise e
        
        # テキストエンコーディング
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        
        try:
            # 常にtext_encoder_managerから最新のテキストエンコーダーを取得する
            print(translate("\nテキストエンコーダを初期化します..."))
            try:
                # text_encoder_managerを使用して初期化
                if not text_encoder_manager.ensure_text_encoder_state():
                    print(translate("テキストエンコーダの初期化に失敗しました。再試行します..."))
                    time.sleep(3)
                    if not text_encoder_manager.ensure_text_encoder_state():
                        raise Exception(translate("テキストエンコーダとtext_encoder_2の初期化に複数回失敗しました"))
                text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()
                print(translate("テキストエンコーダの初期化が完了しました"))
            except Exception as e:
                print(translate("テキストエンコーダのロードに失敗しました: {0}").format(e))
                traceback.print_exc()
                raise e
            
            if not high_vram:
                print(translate("\nテキストエンコーダをGPUにロード..."))
                fake_diffusers_current_device(text_encoder, gpu)
                load_model_as_complete(text_encoder_2, target_device=gpu)
            
            # テキストエンコーディング実行
            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            
            # ローVRAMモードでは使用後すぐにCPUに戻す
            if not high_vram:
                if text_encoder is not None and hasattr(text_encoder, 'to'):
                    text_encoder.to('cpu')
                if text_encoder_2 is not None and hasattr(text_encoder_2, 'to'):
                    text_encoder_2.to('cpu')
                
                # メモリ状態をログ
                free_mem_gb = get_cuda_free_memory_gb(gpu)
                print(translate("\nテキストエンコード後の空きVRAM {0} GB").format(free_mem_gb))
                
                # メモリクリーンアップ
                torch.cuda.empty_cache()
        except Exception as e:
            print(translate("テキストエンコードエラー: {0}").format(e))
            raise e
        
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        # データ型変換
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        
        # 1フレームモード用の設定
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = 1  # 1フレームモード
        
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        
        # 1フレームモード用に特別に設定
        latent_paddings = [0] * total_latent_sections
        print(translate("[INFO] 1フレームモード: latent_paddings = {0}に設定しました").format(latent_paddings))
        
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0  # 常にTrue
            latent_padding_size = latent_padding * latent_window_size  # 常に0
            
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return
            
            # 1フレームモード用のindices設定
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            
            # 詳細設定のlatent_indexに基づいたインデックス処理
            all_indices = torch.arange(0, latent_window_size).unsqueeze(0)
            if latent_index > 0 and latent_index < latent_window_size:
                print(translate("\n[INFO] カスタムレイテントインデックス {0} を使用します").format(latent_index))
                # ユーザー指定のインデックスを使用
                latent_indices = all_indices[:, latent_index:latent_index+1]
            else:
                print(translate("\n[INFO] デフォルトの最後のインデックスを使用します"))
                # デフォルトは最後のインデックス
                latent_indices = all_indices[:, -1:]
            
            # clean_latents設定
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            
            # 形状の確認と修正
            print(translate("[DEBUG] start_latent形状: {0}").format(start_latent.shape))
            print(translate("[DEBUG] history_latents形状: {0}").format(history_latents.shape))
            
            # start_latentの形状を確認
            if len(start_latent.shape) < 5:  # バッチとフレーム次元がない場合
                # [B, C, H, W] → [B, C, 1, H, W] の形に変換
                clean_latents_pre = start_latent.unsqueeze(2).to(history_latents.dtype).to(history_latents.device)
            else:
                clean_latents_pre = start_latent.to(history_latents.dtype).to(history_latents.device)
            
            print(translate("[DEBUG] clean_latents_pre形状(変換後): {0}").format(clean_latents_pre.shape))
            
            # history_latentsからデータを適切に分割
            try:
                # 分割前に形状確認
                frames_to_split = history_latents.shape[2]
                
                if frames_to_split >= 19:  # 1+2+16フレームを想定
                    # 正常分割
                    clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                else:
                    # フレーム数が不足している場合は適切なサイズで初期化
                    print(translate("[WARN] フレーム数が不足しているため、ゼロで初期化します"))
                    clean_latents_post = torch.zeros(1, 16, 1, height // 8, width // 8, dtype=history_latents.dtype, device=history_latents.device)
                    clean_latents_2x = torch.zeros(1, 16, 2, height // 8, width // 8, dtype=history_latents.dtype, device=history_latents.device)
                    clean_latents_4x = torch.zeros(1, 16, 16, height // 8, width // 8, dtype=history_latents.dtype, device=history_latents.device)
            except Exception as e:
                print(translate("[ERROR] history_latentsの分割中にエラー: {0}").format(e))
                # エラー発生時はゼロで初期化
                clean_latents_post = torch.zeros(1, 16, 1, height // 8, width // 8, dtype=torch.float32, device='cpu')
                clean_latents_2x = torch.zeros(1, 16, 2, height // 8, width // 8, dtype=torch.float32, device='cpu')
                clean_latents_4x = torch.zeros(1, 16, 16, height // 8, width // 8, dtype=torch.float32, device='cpu')
            
            # 詳細設定のオプションに基づいて処理
            if use_clean_latents_post:
                try:
                    # 次元確認のためのログ出力
                    print(translate("\n[DEBUG] clean_latents_pre形状: {0}").format(clean_latents_pre.shape))
                    print(translate("[DEBUG] clean_latents_post形状: {0}").format(clean_latents_post.shape))
                    
                    # 正しい形状に変換して結合
                    if len(clean_latents_pre.shape) != len(clean_latents_post.shape):
                        print(translate("[DEBUG] 形状が異なるため次元調整を行います"))
                        # 形状を合わせる
                        if len(clean_latents_pre.shape) < len(clean_latents_post.shape):
                            clean_latents_pre = clean_latents_pre.unsqueeze(2)
                        else:
                            clean_latents_post = clean_latents_post.unsqueeze(1)
                    
                    # 次元調整後の形状確認
                    print(translate("[DEBUG] 調整後 clean_latents_pre形状: {0}").format(clean_latents_pre.shape))
                    print(translate("[DEBUG] 調整後 clean_latents_post形状: {0}").format(clean_latents_post.shape))
                    
                    # 結合
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                    print(translate("[DEBUG] 結合後 clean_latents形状: {0}").format(clean_latents.shape))
                except Exception as e:
                    print(translate("\n[ERROR] clean_latentsの結合中にエラーが発生しました: {0}").format(e))
                    print(translate("[FALLBACK] 前処理のみを使用します"))
                    clean_latents = clean_latents_pre
                    if len(clean_latents.shape) == 4:  # [B, C, H, W]
                        clean_latents = clean_latents.unsqueeze(2)  # [B, C, 1, H, W]
            else:
                print(translate("\n[OPTIMIZE] clean_latents_postは無効化されています。生成が高速化されますが、ノイズが増える可能性があります"))
                # clean_latents_postを使用しない場合、前処理+空白レイテント（ゼロテンソル）を結合
                # これはオリジナルの実装をできるだけ維持しつつ、エラーを回避するためのアプローチ
                clean_latents_pre_shaped = clean_latents_pre
                if len(clean_latents_pre.shape) == 4:  # [B, C, H, W]
                    clean_latents_pre_shaped = clean_latents_pre.unsqueeze(2)  # [B, C, 1, H, W]
                
                # 空のレイテントを作成（形状を合わせる）
                shape = list(clean_latents_pre_shaped.shape)
                # [B, C, 1, H, W]の形状に対して、[B, C, 1, H, W]の空テンソルを作成
                empty_latent = torch.zeros_like(clean_latents_pre_shaped)
                
                # 結合して形状を維持
                clean_latents = torch.cat([clean_latents_pre_shaped, empty_latent], dim=2)
                print(translate("[DEBUG] 代替手法による結合後 clean_latents形状: {0}").format(clean_latents.shape))
            
            # transformerの初期化とロード（未ロードの場合）
            if transformer is None:
                print(translate("\ntransformerモデルを初めてロードします..."))
                try:
                    # transformerの状態を確認
                    if not transformer_manager.ensure_transformer_state():
                        raise Exception(translate("transformer状態の確認に失敗しました"))
                        
                    # transformerインスタンスを取得
                    transformer = transformer_manager.get_transformer()
                    print(translate("transformerの初期化が完了しました"))
                except Exception as e:
                    print(translate("transformerのロードに失敗しました: {0}").format(e))
                    traceback.print_exc()
                    
                    # 再試行
                    print(translate("transformerのロードを再試行します..."))
                    time.sleep(5)
                    
                    if not transformer_manager.ensure_transformer_state():
                        raise Exception(translate("transformer状態の再確認に失敗しました"))
                    
                    transformer = transformer_manager.get_transformer()
            
            # eichiと同様にtransformerをGPUに移動（ensure_transformer_stateでリロードしたtransformerは
            # 仮想デバイス上のものなので、明示的にGPUに移動する必要がある）
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
            
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)
            
            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)
                
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                
                if stream.input_queue.top() == 'end':
                    print(translate("\n[INFO] ユーザーがタスクを中断しました"))
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')
                
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = translate('1フレームモード: サンプリング中...')
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
            
            # 詳細設定に基づいてパラメータを準備
            # 形状チェックのデバッグ
            print(translate("\n[DEBUG] clean_latents_2x形状: {0}").format(clean_latents_2x.shape))
            print(translate("[DEBUG] clean_latents_4x形状: {0}").format(clean_latents_4x.shape))

            # 異常な次元数を持つテンソルを処理
            try:
                if len(clean_latents_2x.shape) > 5:
                    print(translate("[DEBUG] clean_latents_2xの形状を修正します"))
                    # エラーメッセージは[1, 16, 1, 1, 96, 64]のような6次元テンソルを示しています
                    # 必要なのは5次元テンソル[B, C, T, H, W]です
                    if clean_latents_2x.shape[2] == 1 and clean_latents_2x.shape[3] == 1:
                        # 余分な次元を削除
                        clean_latents_2x = clean_latents_2x.squeeze(2)  # [1, 16, 1, 96, 64]
                        print(translate("[DEBUG] 調整後 clean_latents_2x形状: {0}").format(clean_latents_2x.shape))
            except Exception as e:
                print(translate("[ERROR] clean_latents_2xの形状調整中にエラー: {0}").format(e))
            
            try:
                if len(clean_latents_4x.shape) > 5:
                    print(translate("[DEBUG] clean_latents_4xの形状を修正します"))
                    if clean_latents_4x.shape[2] == 1 and clean_latents_4x.shape[3] == 1:
                        # 余分な次元を削除
                        clean_latents_4x = clean_latents_4x.squeeze(2)  # [1, 16, 1, 96, 64]
                        print(translate("[DEBUG] 調整後 clean_latents_4x形状: {0}").format(clean_latents_4x.shape))
            except Exception as e:
                print(translate("[ERROR] clean_latents_4xの形状調整中にエラー: {0}").format(e))
            
            # clean_latents_2xとclean_latents_4xの設定に応じて変数を調整
            clean_latents_2x_param = clean_latents_2x if use_clean_latents_2x else None
            clean_latents_4x_param = clean_latents_4x if use_clean_latents_4x else None
            
            # 最適化オプションのログ
            if not use_clean_latents_2x:
                print(translate("\n[INFO] clean_latents_2xは無効化されています。出力画像に変化が発生します"))
            if not use_clean_latents_4x:
                print(translate("\n[INFO] clean_latents_4xは無効化されています。出力画像に変化が発生します"))
                
            # RoPE値を設定 - transformer内部のmax_positionを設定してみる
            print(translate("\n[INFO] 設定されたRoPE値(latent_window_size): {0}").format(latent_window_size))
            
            try:
                # transformerモデルの内部パラメータを調整
                # HunyuanVideoTransformerモデル内部のmax_positionに相当する値を変更する
                if hasattr(transformer, 'max_pos_embed_window_size'):
                    original_value = transformer.max_pos_embed_window_size
                    print(translate("[INFO] 元のmax_pos_embed_window_size: {0}").format(original_value))
                    transformer.max_pos_embed_window_size = latent_window_size
                    print(translate("[INFO] max_pos_embed_window_sizeを{0}に設定しました").format(latent_window_size))
                
                # RoFormerなどのRoPE実装を探して調整
                if hasattr(transformer, 'attn_processors'):
                    print(translate("[INFO] attn_processorsが見つかりました、RoPE関連設定を探します"))
                    # 詳細は出力しない
                
                # HunyuanVideo特有の実装を探す
                if hasattr(transformer, 'create_image_rotary_emb'):
                    print(translate("[INFO] create_image_rotary_embを調整中..."))
            except Exception as e:
                print(translate("[WARN] RoPE値の設定中にエラーが発生しました: {0}").format(e))
                print(translate("[INFO] デフォルト値を使用します"))
            
            # この値は保存されますが、実際のモデル内部には適用されません（テンソルサイズエラー回避のため）
            print(translate("[INFO] sample_hunyuan関数を呼び出します"))
            
            # sample_hunyuan関数呼び出し部分
            try:
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x_param,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x_param,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )
                
                print(translate("[INFO] 生成は正常に完了しました"))
                
            except RuntimeError as e:
                error_msg = str(e)
                if "size of tensor" in error_msg:
                    print(translate("\n[ERROR] テンソルサイズの不一致エラーが発生しました: {0}").format(error_msg))
                    print(translate("[ERROR] 現在の実装ではこのエラーは発生しないはずです。開発者に報告してください。"))
                    
                    # デバッグ情報を提供
                    print(translate("[DEBUG] clean_latents形状: {0}").format(clean_latents.shape if 'clean_latents' in locals() else "未定義"))
                    print(translate("[DEBUG] latent_indices形状: {0}").format(latent_indices.shape if 'latent_indices' in locals() else "未定義"))
                    raise e
                else:
                    # その他のランタイムエラーはそのまま投げる
                    raise e
            
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            
            # eichiと同様にメモリを明示的かつ効率的に解放
            if not high_vram:
                # transformerのメモリを解放
                print(translate("\n生成完了 - transformerをアンロード中..."))
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                
                # アンロード後のメモリ状態をログ
                free_mem_gb_after_unload = get_cuda_free_memory_gb(gpu)
                print(translate("transformerアンロード後の空きVRAM {0} GB").format(free_mem_gb_after_unload))
                
                # VAEをデコード用にロード（VAEが解放されていた場合は再ロード）
                if vae is None:
                    print(translate("VAEモデルを再ロードします..."))
                    try:
                        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                        setup_vae_if_loaded()  # VAEの設定を適用
                        print(translate("VAEモデルの再ロードが完了しました"))
                    except Exception as e:
                        print(translate("VAEモデルの再ロードに失敗しました: {0}").format(e))
                        traceback.print_exc()
                        print(translate("5秒間待機後に再試行します..."))
                        time.sleep(5)
                        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                        setup_vae_if_loaded()  # VAEの設定を適用
                
                print(translate("VAEをGPUにロード中..."))
                load_model_as_complete(vae, target_device=gpu)
                
                # ロード後のメモリ状態をログ
                free_mem_gb_after_vae = get_cuda_free_memory_gb(gpu)
                print(translate("VAEロード後の空きVRAM {0} GB").format(free_mem_gb_after_vae))
                
                # 追加のメモリクリーンアップ
                torch.cuda.empty_cache()
            
            # 実際に使用するラテントを抽出
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            
            # 使用していないテンソルを早めに解放
            del history_latents
            torch.cuda.empty_cache()
            
            # 1フレームモードではVAEデコードを行い、画像を直接保存
            try:
                print(translate("[INFO] 1フレームモード: 生成されたラテントから直接画像を保存します"))
                print(translate("[INFO] 現在のラテント形状: {0}").format(real_history_latents.shape))
                
                # VAEデコード処理前にメモリを確認
                free_mem_before_decode = get_cuda_free_memory_gb(gpu)
                print(translate("[INFO] VAEデコード前の空きVRAM: {0} GB").format(free_mem_before_decode))
                
                # VAEが解放されていた場合は再ロード
                if vae is None:
                    print(translate("VAEモデルを最終デコード用に再ロードします..."))
                    try:
                        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                        setup_vae_if_loaded()  # VAEの設定を適用
                        # VAEをGPUに移動
                        load_model_as_complete(vae, target_device=gpu)
                        print(translate("VAEモデルの再ロードが完了しました"))
                    except Exception as e:
                        print(translate("VAEモデルの再ロードに失敗しました: {0}").format(e))
                        traceback.print_exc()
                        raise e
                
                # VAEデコード処理
                with torch.no_grad():  # 明示的にno_gradコンテキストを使用
                    # ラテントをCPUではなくGPUに置いて効率的に処理
                    real_history_latents_gpu = real_history_latents.to(gpu)
                    decoded_image = vae_decode(real_history_latents_gpu, vae).cpu()
                    
                    # 不要なGPU上のラテントをすぐに解放
                    del real_history_latents_gpu
                    torch.cuda.empty_cache()
                
                # デコード後にVAEをCPUに移動してメモリを解放
                if not high_vram and vae is not None:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                
                # デコード後のメモリを確認
                free_mem_after_decode = get_cuda_free_memory_gb(gpu)
                print(translate("[INFO] VAEデコード後の空きVRAM: {0} GB").format(free_mem_after_decode))
                
                # 単一フレームを抽出
                frame = decoded_image[0, :, 0, :, :]
                frame = torch.clamp(frame, -1., 1.) * 127.5 + 127.5
                frame = frame.detach().cpu().to(torch.uint8)
                frame = einops.rearrange(frame, 'c h w -> h w c').numpy()
                
                # デコード結果を解放
                del decoded_image
                del real_history_latents
                torch.cuda.empty_cache()
                
                # メタデータを設定
                metadata = {
                    PROMPT_KEY: prompt,
                    SEED_KEY: str(seed)
                }
                
                # 画像として保存（メタデータ埋め込み）
                output_filename = os.path.join(outputs_folder, f'{job_id}_oneframe.png')
                pil_img = Image.fromarray(frame)
                pil_img.save(output_filename)  # 一度保存
                
                # メタデータを埋め込み
                try:
                    embed_metadata_to_png(output_filename, output_filename, metadata)
                    print(translate("[INFO] 画像メタデータを埋め込みました"))
                except Exception as e:
                    print(translate("[WARNING] メタデータ埋め込みエラー: {0}").format(e))
                
                print(translate("[INFO] 1フレーム画像を保存しました: {0}").format(output_filename))
                
                # MP4保存はスキップして、画像ファイルパスを返す
                stream.output_queue.push(('file', output_filename))
                
            except Exception as e:
                print(translate("[ERROR] 1フレームの画像保存中にエラーが発生しました: {0}").format(e))
                traceback.print_exc()
                
                # エラー発生時のメモリ解放を試みる
                if 'real_history_latents_gpu' in locals():
                    del real_history_latents_gpu
                if 'real_history_latents' in locals():
                    del real_history_latents
                if 'decoded_image' in locals():
                    del decoded_image
                torch.cuda.empty_cache()
            
            break  # 1フレーム生成は1回のみ
            
    except Exception as e:
        print(translate("\n処理中にエラーが発生しました: {0}").format(e))
        traceback.print_exc()
        
        # エラー時の詳細なメモリクリーンアップ
        try:
            if not high_vram:
                print(translate("\nエラー発生時のメモリクリーンアップを実行..."))
                
                # 効率的なクリーンアップのために、重いモデルから順にアンロード
                models_to_unload = [
                    ('transformer', transformer), 
                    ('vae', vae), 
                    ('image_encoder', image_encoder), 
                    ('text_encoder', text_encoder), 
                    ('text_encoder_2', text_encoder_2)
                ]
                
                # 各モデルを個別にアンロードして解放
                for model_name, model in models_to_unload:
                    if model is not None:
                        try:
                            print(translate("{0}をアンロード中...").format(model_name))
                            # モデルをCPUに移動
                            if hasattr(model, 'to'):
                                model.to('cpu')
                            # 参照を明示的に削除
                            if model_name == 'transformer':
                                transformer = None
                            elif model_name == 'vae':
                                vae = None
                            elif model_name == 'image_encoder':
                                image_encoder = None
                            elif model_name == 'text_encoder':
                                text_encoder = None
                            elif model_name == 'text_encoder_2':
                                text_encoder_2 = None
                            # 各モデル解放後にすぐメモリ解放
                            torch.cuda.empty_cache()
                        except Exception as unload_error:
                            print(translate("{0}のアンロード中にエラー: {1}").format(model_name, unload_error))
                
                # 一括アンロード（遅延ロード方式のため実際に存在するモデルのみアンロード）
                models_to_unload = []
                for model in [text_encoder, text_encoder_2, image_encoder, vae, transformer]:
                    if model is not None:
                        models_to_unload.append(model)
                
                if models_to_unload:
                    unload_complete_models(*models_to_unload)
                
                # 明示的なガベージコレクション（複数回）
                import gc
                print(translate("ガベージコレクション実行中..."))
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                
                # メモリ状態を報告
                free_mem_gb = get_cuda_free_memory_gb(gpu)
                print(translate("クリーンアップ後の空きVRAM {0} GB").format(free_mem_gb))
                
                # 追加の変数クリーンアップ
                for var_name in ['start_latent', 'decoded_image', 'history_latents', 'real_history_latents', 
                              'real_history_latents_gpu', 'generated_latents', 'input_image_pt', 'input_image_gpu']:
                    if var_name in locals():
                        try:
                            exec(f"del {var_name}")
                            print(translate("変数 {0} を解放しました").format(var_name))
                        except:
                            pass
        except Exception as cleanup_error:
            print(translate("メモリクリーンアップ中にエラー: {0}").format(cleanup_error))
    
    # 処理完了を通知（個別バッチの完了）
    print(translate("\n処理が完了しました"))
    
    # worker関数内では効果音を鳴らさない（バッチ処理全体の完了時のみ鳴らす）
    
    stream.output_queue.push(('end', None))
    return

def handle_open_folder_btn(folder_name):
    """フォルダ名を保存し、そのフォルダを開く - endframe_ichiと同じ実装"""
    if not folder_name or not folder_name.strip():
        folder_name = "outputs"

    # フォルダパスを取得
    folder_path = get_output_folder_path(folder_name)

    # 設定を更新して保存
    settings = load_settings()
    old_folder_name = settings.get('output_folder')

    if old_folder_name != folder_name:
        settings['output_folder'] = folder_name
        save_result = save_settings(settings)
        if save_result:
            # グローバル変数も更新
            global output_folder_name, outputs_folder
            output_folder_name = folder_name
            outputs_folder = folder_path
        print(translate("出力フォルダ設定を保存しました: {folder_name}").format(folder_name=folder_name))

    # フォルダを開く
    open_output_folder(folder_path)

    # 出力ディレクトリ入力欄とパス表示を更新
    return gr.update(value=folder_name), gr.update(value=folder_path)
    
def update_from_image_metadata(image, should_copy):
    """画像からメタデータを抽出してプロンプトとシードを更新する関数
    
    Args:
        image: 画像データ（numpy配列）
        should_copy: メタデータを複写するかどうかの指定
        
    Returns:
        tuple: (プロンプト更新データ, シード値更新データ)
    """
    if not should_copy or image is None:
        return gr.update(), gr.update()
    
    try:
        # NumPy配列からメタデータを抽出
        metadata = extract_metadata_from_numpy_array(image)
        
        if not metadata:
            print(translate("画像にメタデータが含まれていません"))
            return gr.update(), gr.update()
        
        # プロンプトとSEEDをUIに反映
        prompt_update = gr.update()
        seed_update = gr.update()
        
        if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
            prompt_update = gr.update(value=metadata[PROMPT_KEY])
            print(translate("プロンプトを画像から取得: {0}").format(metadata[PROMPT_KEY]))
        
        if SEED_KEY in metadata and metadata[SEED_KEY]:
            # SEED値を整数に変換
            try:
                seed_value = int(metadata[SEED_KEY])
                seed_update = gr.update(value=seed_value)
                print(translate("シード値を画像から取得: {0}").format(seed_value))
            except ValueError:
                print(translate("シード値の変換に失敗しました: {0}").format(metadata[SEED_KEY]))
        
        return prompt_update, seed_update
    
    except Exception as e:
        print(translate("メタデータ抽出エラー: {0}").format(e))
        return gr.update(), gr.update()

def check_metadata_on_checkbox_change(should_copy, image):
    """チェックボックスの状態が変更された時に画像からメタデータを抽出する関数"""
    return update_from_image_metadata(image, should_copy)

def process(input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, 
            lora_files, lora_files2, lora_scales_text, use_lora, fp8_optimization, resolution, output_directory=None, 
            batch_count=1, use_random_seed=False, latent_window_size=9, latent_index=0, 
            use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True):
    global stream
    
    # 出力フォルダの設定
    global outputs_folder
    global output_folder_name
    if output_directory and output_directory.strip():
        # 出力フォルダパスを取得
        outputs_folder = get_output_folder_path(output_directory)
        print(translate("出力フォルダを設定: {0}").format(outputs_folder))

        # フォルダ名が現在の設定と異なる場合は設定ファイルを更新
        if output_directory != output_folder_name:
            settings = load_settings()
            settings['output_folder'] = output_directory
            if save_settings(settings):
                output_folder_name = output_directory
                print(translate("出力フォルダ設定を保存しました: {0}").format(output_directory))
    else:
        # デフォルト設定を使用
        outputs_folder = get_output_folder_path(output_folder_name)
        print(translate("デフォルト出力フォルダを使用: {0}").format(outputs_folder))

    # フォルダが存在しない場合は作成
    os.makedirs(outputs_folder, exist_ok=True)
    
    # 出力ディレクトリを設定
    output_dir = outputs_folder
    
    # バッチ処理のパラメータチェック
    batch_count = max(1, min(int(batch_count), 100))  # 1〜100の間に制限
    print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))
    
    # 入力画像チェック
    assert input_image is not None, translate('No input image!')
    
    # LoRAの状態をログ出力
    if use_lora and has_lora_support:
        print(translate("[INFO] LoRAの使用設定: use_lora = {0}, has_lora_support = {1}").format(use_lora, has_lora_support))
        print(translate("[INFO] lora_files = {0}, 型: {1}").format(lora_files, type(lora_files)))
        print(translate("[INFO] lora_scales_text = {0}").format(lora_scales_text))
    
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    
    # バッチ処理用の変数
    batch_stopped = False
    original_seed = seed if seed else (random.randint(0, 2**32 - 1) if use_random_seed else 31337)
    
    # バッチ処理ループ
    for batch_index in range(batch_count):
        if batch_stopped:
            break
            
        # バッチ情報をログ出力
        if batch_count > 1:
            batch_info = translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)
            print(f"\n{batch_info}")
            # UIにもバッチ情報を表示
            yield None, gr.update(visible=False), batch_info, "", gr.update(interactive=False), gr.update(interactive=True)
                
        # バッチごとにシードを変更
        current_seed = original_seed + batch_index
        
        if batch_count > 1:
            print(translate("\u25c6 現在のシード値: {0}").format(current_seed))
        
        if batch_stopped:
            break
            
        stream = AsyncStream()
        
        # バッチインデックスをジョブIDに含める
        batch_suffix = f"{batch_index}" if batch_index > 0 else ""
        
        # ワーカー実行 - 詳細設定パラメータを含む
        async_run(worker, input_image, prompt, n_prompt, current_seed, steps, cfg, gs, rs, 
                 gpu_memory_preservation, use_teacache, lora_files, lora_files2, lora_scales_text, 
                 output_dir, use_lora, fp8_optimization, resolution,
                 latent_window_size, latent_index, use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post)
        
        output_filename = None
        
        # ジョブ完了まで監視
        while True:
            flag, data = stream.output_queue.next()
            
            if flag == 'file':
                output_filename = data
                yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
            
            if flag == 'progress':
                preview, desc, html = data
                yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
            
            if flag == 'end':
                # バッチ処理中は最後の画像のみを表示
                if batch_index == batch_count - 1:  # 最後のバッチ
                    yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
                break
                
            # ユーザーが中断した場合
            if stream.input_queue.top() == 'end' or batch_stopped:
                batch_stopped = True
                print(translate("バッチ処理が中断されました（{0}/{1}）").format(batch_index + 1, batch_count))
                yield output_filename, gr.update(visible=False), translate("バッチ処理が中断されました"), '', gr.update(interactive=True), gr.update(interactive=False)
                return
    
    # すべてのバッチ処理が正常に完了した場合のみ、効果音を鳴らす
    if not batch_stopped:
        print(translate("\n[INFO] 全てのバッチ処理が完了しました"))
        
        # 処理完了時の効果音（Windowsの場合）
        if HAS_WINSOUND:
            try:
                # Windows環境では完了音を鳴らす
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
                print(translate("[INFO] Windows完了通知音を再生しました"))
            except Exception as e:
                print(translate("[WARN] 完了通知音の再生に失敗しました: {0}").format(e))
        else:
            # Linux/Mac環境ではログにメッセージを出力
            print("\n" + "*" * 50)
            print(translate("【全バッチ処理完了】プロセスが完了しました - ") + time.strftime("%Y-%m-%d %H:%M:%S"))
            print("*" * 50 + "\n")
            
    return

def end_process():
    """生成終了ボタンが押された時の処理"""
    global stream
    global batch_stopped
    
    # 現在のバッチと次のバッチ処理を全て停止するフラグを設定
    batch_stopped = True
    print(translate("\n[INFO] 停止ボタンが押されました。バッチ処理を停止します..."))
    # 現在実行中のバッチを停止
    stream.input_queue.push('end')
    
    # ボタンの名前を一時的に変更することでユーザーに停止処理が進行中であることを表示
    return gr.update(value=translate("停止処理中..."))

css = get_app_css()  # eichi_utilsのスタイルを使用
block = gr.Blocks(css=css).queue()
with block:
    # eichiと同じ半透明度スタイルを使用
    gr.HTML('<h1>FramePack<span class="title-suffix">-oichi</span></h1>')
    
    # 初期化時にtransformerの状態確認は行わない（必要時に遅延ロード）
    # ここではロードをスキップして、ワーカー関数内で必要になったときにだけロードする
    
    # Use Random Seedの初期値（先に定義して後で使う）
    use_random_seed_default = True
    seed_default = random.randint(0, 2**32 - 1) if use_random_seed_default else 31337
    
    # eichiのプリセット管理関連のインポート
    from eichi_utils.preset_manager import get_default_startup_prompt, load_presets, save_preset, delete_preset
    
    with gr.Row():
        with gr.Column(scale=1):
            # 左カラム - eichiと同じ順番
            # モードについての説明を画像枠の上に表示
            gr.Markdown(translate("**「1フレーム推論」モードでは、1枚の新しい未来の画像を生成します。**"))
            
            input_image = gr.Image(sources='upload', type="numpy", label=translate("画像"), height=320)
            
            # 解像度設定（画像の直下に）
            resolution = gr.Dropdown(
                label=translate("解像度"),
                choices=[512, 640, 768, 960, 1080],
                value=640,
                info=translate("出力画像の基準解像度。640推奨。960/1080は高負荷・高メモリ消費")
            )
            
            # バッチ処理設定 - endframe_ichiと同じUIにする
            with gr.Column(scale=1):
                batch_count = gr.Slider(
                    label=translate("バッチ処理回数"),
                    minimum=1,
                    maximum=100,
                    value=1,
                    step=1,
                    info=translate("同じ設定で連続生成する回数。SEEDは各回で+1されます")
                )
            
            # 生成開始/中止ボタン - endframe_ichiと完全に同じ実装
            with gr.Row():
                start_button = gr.Button(value=translate("Start Generation"))
                end_button = gr.Button(value=translate("End Generation"), interactive=False)
            
            # 埋め込みプロンプト機能 - endframe_ichiと完全に同じ実装
            # グローバル変数として定義し、後で他の場所から参照できるようにする
            global copy_metadata
            copy_metadata = gr.Checkbox(
                label=translate("埋め込みプロンプトおよびシードを複写する"),
                value=False,
                info=translate("チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します")
            )
                
            # メタデータ抽出結果表示用（非表示）
            extracted_info = gr.Markdown(visible=False)
            extracted_prompt = gr.Textbox(visible=False)
            extracted_seed = gr.Textbox(visible=False)
            
            # 詳細設定アコーディオン - 埋め込みプロンプト機能の直後に配置
            with gr.Accordion(translate("詳細設定"), open=False, elem_classes="section-accordion"):
                gr.Markdown(f"### " + translate("レイテント処理設定"))
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # ツイートに基づくRoPE値の設定
                        latent_window_size = gr.Slider(
                            label=translate("RoPE値 (latent_window_size)"),
                            minimum=1,
                            maximum=64,
                            value=9,  # デフォルト値
                            step=1,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("動きの変化量に影響します。大きい値ほど大きな変化が発生します。モデルの内部調整用パラメータです。")
                        )
                    
                    with gr.Column(scale=1):
                        # レイテントインデックス
                        latent_index = gr.Slider(
                            label=translate("レイテントインデックス"),
                            minimum=0,
                            maximum=64,
                            value=0,  # デフォルト値
                            step=1,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("0は基本、大きい値で衣装変更などの効果が得られる場合があります。値が大きいとノイズが増えます。")
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # clean_latents_2xの有効/無効
                        use_clean_latents_2x = gr.Checkbox(
                            label=translate("clean_latents_2xを使用"),
                            value=True,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("オフにすると変化が発生します。画質や速度に影響があります")
                        )
                    
                    with gr.Column(scale=1):
                        # clean_latents_4xの有効/無効
                        use_clean_latents_4x = gr.Checkbox(
                            label=translate("clean_latents_4xを使用"),
                            value=True,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("オフにすると変化が発生します。画質や速度に影響があります")
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # clean_latents_postの有効/無効
                        use_clean_latents_post = gr.Checkbox(
                            label=translate("clean_latents_postを使用"),
                            value=True,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("オフにするとかなり速くなりますが、ノイズが増える可能性があります")
                        )
            
            # LoRA設定 - endframe_ichiと完全に同じ実装にする
            if has_lora_support:
                with gr.Group() as lora_settings_group:
                    gr.Markdown(f"### " + translate("LoRA設定"))
                    
                    # LoRA使用有無のチェックボックス
                    use_lora = gr.Checkbox(label=translate("LoRAを使用する"), value=False, info=translate("チェックをオンにするとLoRAを使用します（要16GB VRAM以上）"))

                    # LoRA設定コンポーネント（初期状態では非表示）
                    # メインのLoRAファイル
                    lora_files = gr.File(
                        label=translate("LoRAファイル (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"],
                        visible=False
                    )
                    # 追加のLoRAファイル
                    lora_files2 = gr.File(
                        label=translate("LoRAファイル2 (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"],
                        visible=False
                    )
                    # スケール値の入力フィールド
                    lora_scales_text = gr.Textbox(
                        label=translate("LoRA適用強度 (カンマ区切り)"),
                        value="0.8,0.8",
                        info=translate("各LoRAのスケール値をカンマ区切りで入力 (例: 0.8,0.5)"),
                        visible=False
                    )
                    # FP8最適化オプション（高速化のための実験的機能）
                    fp8_optimization = gr.Checkbox(
                        label=translate("FP8最適化（高速化）"), 
                        value=False, 
                        info=translate("GPUで高速化できますが若干精度が落ちます"), 
                        visible=False
                    )

                    # チェックボックスの状態によって他のLoRA設定の表示/非表示を切り替える関数
                    def toggle_lora_settings(use_lora):
                        return [
                            gr.update(visible=use_lora),  # lora_files
                            gr.update(visible=use_lora),  # lora_files2
                            gr.update(visible=use_lora),  # lora_scales_text
                            gr.update(visible=use_lora),  # fp8_optimization
                        ]

                    # チェックボックスの変更イベントに関数を紐づけ
                    use_lora.change(fn=toggle_lora_settings,
                               inputs=[use_lora],
                               outputs=[lora_files, lora_files2, lora_scales_text, fp8_optimization])

                    # LoRAサポートが無効の場合のメッセージ
                    if not has_lora_support:
                        gr.Markdown(translate("LoRAサポートは現在無効です。lora_utilsモジュールが必要です。"))
            else:
                # LoRAサポートが無効の場合はダミー変数を作成
                use_lora = gr.Checkbox(visible=False, value=False)
                lora_files = gr.File(visible=False)
                lora_files2 = gr.File(visible=False) 
                lora_scales_text = gr.Textbox(visible=False, value="0.8,0.8")
                fp8_optimization = gr.Checkbox(visible=False, value=False)
            
            # プロンプト入力
            prompt = gr.Textbox(label=translate("プロンプト"), value=get_default_startup_prompt(), lines=6)
            n_prompt = gr.Textbox(label=translate("ネガティブプロンプト"), value='')
            
            # プロンプト管理パネル
            with gr.Group(visible=True) as prompt_management:
                gr.Markdown(f"### " + translate("プロンプト管理"))
                
                # 編集画面を常時表示する
                with gr.Group(visible=True):
                    # 起動時デフォルトの初期表示用に取得
                    default_prompt = ""
                    default_name = ""
                    for preset in load_presets()["presets"]:
                        if preset.get("is_startup_default", False):
                            default_prompt = preset["prompt"]
                            default_name = preset["name"]
                            break
                    
                    with gr.Row():
                        edit_name = gr.Textbox(label=translate("プリセット名"), placeholder=translate("名前を入力..."), value=default_name)
                    
                    edit_prompt = gr.Textbox(label=translate("プロンプト"), lines=5, value=default_prompt)
                    
                    with gr.Row():
                        # 起動時デフォルトをデフォルト選択に設定
                        default_preset = translate("起動時デフォルト")
                        # プリセットデータから全プリセット名を取得
                        presets_data = load_presets()
                        choices = [preset["name"] for preset in presets_data["presets"]]
                        default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
                        user_presets = [name for name in choices if name not in default_presets]
                        sorted_choices = [(name, name) for name in sorted(default_presets) + sorted(user_presets)]
                        preset_dropdown = gr.Dropdown(label=translate("プリセット"), choices=sorted_choices, value=default_preset, type="value")
                    
                    with gr.Row():
                        save_btn = gr.Button(value=translate("保存"), variant="primary")
                        apply_preset_btn = gr.Button(value=translate("反映"), variant="primary")
                        clear_btn = gr.Button(value=translate("クリア"))
                        delete_preset_btn = gr.Button(value=translate("削除"))
                
                # メッセージ表示用
                result_message = gr.Markdown("")
                
        with gr.Column(scale=1):
            # 右カラム - 生成結果と設定
            result_image = gr.Image(label=translate("生成結果"), height=512)
            preview_image = gr.Image(label=translate("処理中のプレビュー"), height=200, visible=False)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            
            # endframe_ichiと同じ順序で設定項目を配置
            # TeaCacheとランダムシード設定
            use_teacache = gr.Checkbox(label=translate('Use TeaCache'), value=True, info=translate('Faster speed, but often makes hands and fingers slightly worse.'))
            
            # Use Random Seedの初期値
            use_random_seed = gr.Checkbox(label=translate("Use Random Seed"), value=use_random_seed_default)
            seed = gr.Number(label=translate("Seed"), value=seed_default, precision=0)
            
            # ステップ数などの設定を右カラムに配置
            steps = gr.Slider(label=translate("ステップ数"), minimum=1, maximum=100, value=25, step=1, info=translate('この値の変更は推奨されません'))
            gs = gr.Slider(label=translate("蒸留CFGスケール"), minimum=1.0, maximum=32.0, value=10.0, step=0.01, info=translate('この値の変更は推奨されません'))
            
            # 非表示設定
            cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
            rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
            
            # GPU設定
            gpu_memory_preservation = gr.Slider(
                label=translate("GPUメモリ保持 (GB)"), 
                minimum=6, maximum=128, value=6, step=0.1, 
                info=translate("OOMが発生する場合は値を大きくしてください。値が大きいほど速度が遅くなります。")
            )
            
            # GPUメモリ保持部分のみ残し、詳細設定アコーディオンは削除（すでに移動済み）

            # 出力設定はシード値の下に配置 - endframe_ichiと同じ実装にする
            with gr.Group():
                gr.Markdown(f"### " + translate("出力設定"))
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4):
                        # フォルダ名だけを入力欄に設定
                        output_dir = gr.Textbox(
                            label=translate("出力フォルダ名"),
                            value=output_folder_name,  # 設定から読み込んだ値を使用
                            info=translate("生成画像の保存先フォルダ名"),
                            placeholder="outputs"
                        )
                    with gr.Column(scale=1, min_width=100):
                        open_folder_btn = gr.Button(value=translate("📂 保存および出力フォルダを開く"), size="sm")

                # 実際の出力パスを表示
                with gr.Row(visible=False):
                    path_display = gr.Textbox(
                        label=translate("出力フォルダの完全パス"),
                        value=os.path.join(base_path, output_folder_name),
                        interactive=False
                    )
    
    # シードのランダム化機能
    def set_random_seed(is_checked):
        if is_checked:
            return random.randint(0, 2**32 - 1)
        return gr.update()
    
    # チェックボックス変更時にランダムシードを生成
    use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)
    
    # バッチ処理回数に応じてランダムシードを扱う処理
    def randomize_seed_if_needed(use_random, batch_num=1):
        """バッチ処理用のシード設定関数"""
        if use_random:
            # ランダムシードの場合はバッチごとに異なるシードを生成
            random_seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_num)]
            return random_seeds[0]  # 最初のシードを返す（表示用）
        return gr.update()  # ランダムシードでない場合は何もしない
    
    # プリセット管理に関するイベントハンドラ
    # プリセット保存ボタンのイベント
    def save_button_click_handler(name, prompt_text):
        """保存ボタンクリック時のハンドラ関数"""
        # 重複チェックと正規化
        if "A character" in prompt_text and prompt_text.count("A character") > 1:
            sentences = prompt_text.split(".")
            if len(sentences) > 0:
                prompt_text = sentences[0].strip() + "."
                # 重複を検出したため正規化

        # プリセット保存
        result_msg = save_preset(name, prompt_text)

        # プリセットデータを取得してドロップダウンを更新
        presets_data = load_presets()
        choices = [preset["name"] for preset in presets_data["presets"]]
        default_presets = [n for n in choices if any(p["name"] == n and p.get("is_default", False) for p in presets_data["presets"])]
        user_presets = [n for n in choices if n not in default_presets]
        sorted_choices = [(n, n) for n in sorted(default_presets) + sorted(user_presets)]

        # メインプロンプトは更新しない（保存のみを行う）
        return result_msg, gr.update(choices=sorted_choices), gr.update()

    # クリアボタン処理
    def clear_fields():
        return gr.update(value=""), gr.update(value="")

    # プリセット読込処理
    def load_preset_handler(preset_name):
        # プリセット選択時に編集欄のみを更新
        for preset in load_presets()["presets"]:
            if preset["name"] == preset_name:
                return gr.update(value=preset_name), gr.update(value=preset["prompt"])
        return gr.update(), gr.update()

    # プリセット選択時に編集欄に反映
    def load_preset_handler_wrapper(preset_name):
        # プリセット名がタプルの場合も処理する
        if isinstance(preset_name, tuple) and len(preset_name) == 2:
            preset_name = preset_name[1]  # 値部分を取得
        return load_preset_handler(preset_name)

    # 反映ボタン処理 - 編集画面の内容をメインプロンプトに反映
    def apply_to_prompt(edit_text):
        """編集画面の内容をメインプロンプトに反映する関数"""
        # 編集画面のプロンプトをメインに適用
        return gr.update(value=edit_text)

    # プリセット削除処理
    def delete_preset_handler(preset_name):
        # プリセット名がタプルの場合も処理する
        if isinstance(preset_name, tuple) and len(preset_name) == 2:
            preset_name = preset_name[1]  # 値部分を取得

        result = delete_preset(preset_name)

        # プリセットデータを取得してドロップダウンを更新
        presets_data = load_presets()
        choices = [preset["name"] for preset in presets_data["presets"]]
        default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
        user_presets = [name for name in choices if name not in default_presets]
        sorted_names = sorted(default_presets) + sorted(user_presets)
        updated_choices = [(name, name) for name in sorted_names]

        return result, gr.update(choices=updated_choices)
    
    # 保存ボタンのクリックイベントを接続
    save_btn.click(
        fn=save_button_click_handler,
        inputs=[edit_name, edit_prompt],
        outputs=[result_message, preset_dropdown, prompt]
    )
    
    # クリアボタンのクリックイベントを接続
    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[edit_name, edit_prompt]
    )
    
    # プリセット選択時のイベントを接続
    preset_dropdown.change(
        fn=load_preset_handler_wrapper,
        inputs=[preset_dropdown],
        outputs=[edit_name, edit_prompt]
    )
    
    # 反映ボタンのクリックイベントを接続
    apply_preset_btn.click(
        fn=apply_to_prompt,
        inputs=[edit_prompt],
        outputs=[prompt]
    )
    
    # 削除ボタンのクリックイベントを接続
    delete_preset_btn.click(
        fn=delete_preset_handler,
        inputs=[preset_dropdown],
        outputs=[result_message, preset_dropdown]
    )
    
    # 画像変更時にメタデータを抽出するイベント設定
    input_image.change(
        fn=update_from_image_metadata,
        inputs=[input_image, copy_metadata],
        outputs=[prompt, seed]
    )
    
    # チェックボックス変更時にメタデータを抽出するイベント設定
    copy_metadata.change(
        fn=check_metadata_on_checkbox_change,
        inputs=[copy_metadata, input_image],
        outputs=[prompt, seed]
    )
    
    # フォルダを開くボタンのイベント
    open_folder_btn.click(
        fn=handle_open_folder_btn,
        inputs=[output_dir],
        outputs=[output_dir, path_display]
    )
    
    # 生成開始・中止のイベント
    ips = [input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, 
           lora_files, lora_files2, lora_scales_text, use_lora, fp8_optimization, resolution, output_dir, 
           batch_count, use_random_seed, latent_window_size, latent_index, 
           use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post]  # 詳細設定パラメータを追加
    start_button.click(fn=process, inputs=ips, outputs=[result_image, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process, outputs=[end_button])
    
    gr.HTML(f'<div style="text-align:center; margin-top:20px;">{translate("FramePack 単一フレーム生成版")}</div>')

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)