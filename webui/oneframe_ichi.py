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

# グローバルなモデル状態管理インスタンスを作成
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

# 設定保存用フォルダの設定
settings_folder = os.path.join(webui_folder, 'settings')
os.makedirs(settings_folder, exist_ok=True)

# 出力フォルダの設定
outputs_folder = os.path.join(webui_folder, 'outputs')
os.makedirs(outputs_folder, exist_ok=True)

# グローバル変数
g_frame_size_setting = "1フレーム"

# ワーカー関数
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, 
           gpu_memory_preservation, use_teacache, lora_files=None, lora_scales_text="0.8", 
           output_dir=None, use_lora=False, fp8_optimization=False, resolution=640):
    
    # モデル変数をグローバルとして宣言（遅延ロード用）
    global vae, text_encoder, text_encoder_2, transformer, image_encoder
    
    job_id = generate_timestamp()
    
    # 1フレームモード固有の設定
    latent_window_size = 9  # RoPE値
    total_latent_sections = 1  # 1セクションに固定
    frame_count = 1  # 1フレームモード
    
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
        # LoRA 設定
        current_lora_paths = []
        current_lora_scales = []
        
        if use_lora and has_lora_support:
            print(translate("[DEBUG] LoRA情報: use_lora = {0}, has_lora_support = {1}").format(use_lora, has_lora_support))
            print(translate("[DEBUG] lora_files = {0}, 型: {1}").format(lora_files, type(lora_files)))
            
            # LoRAファイルを収集
            if lora_files is not None:
                if isinstance(lora_files, list):
                    # 複数のLoRAファイル
                    print(translate("[DEBUG] 複数のLoRAファイルが指定されています"))
                    current_lora_paths.extend([file.name for file in lora_files])
                else:
                    # 単一のLoRAファイル
                    print(translate("[DEBUG] 単一のLoRAファイルが指定されています: {0}").format(
                        lora_files.name if hasattr(lora_files, 'name') else str(lora_files)))
                    
                    if hasattr(lora_files, 'name'):
                        current_lora_paths.append(lora_files.name)
                    elif isinstance(lora_files, dict) and 'name' in lora_files:
                        current_lora_paths.append(lora_files['name'])
                    elif isinstance(lora_files, str):
                        current_lora_paths.append(lora_files)
            
            # パスのチェック
            print(translate("[DEBUG] 処理するLoRAパス: {0}").format(current_lora_paths))
            
            if current_lora_paths:  # LoRAパスがある場合のみ解析
                try:
                    scales_text = lora_scales_text.strip()
                    print(translate("[DEBUG] LoRAスケールテキスト: {0}").format(scales_text))
                    scales = [float(scale.strip()) for scale in scales_text.split(',') if scale.strip()]
                    current_lora_scales = scales
                    print(translate("[DEBUG] 解析されたLoRAスケール: {0}").format(current_lora_scales))
                except Exception as e:
                    print(translate("[ERROR] LoRAスケール解析エラー: {0}").format(e))
                    current_lora_scales = [0.8] * len(current_lora_paths)
                    print(translate("[INFO] デフォルトのLoRAスケールを使用: {0}").format(current_lora_scales))
                
                # スケール値の数がLoRAパスの数と一致しない場合は調整
                if len(current_lora_scales) < len(current_lora_paths):
                    print(translate("[INFO] LoRAスケールの数が不足しているため、デフォルト値で補完します"))
                    current_lora_scales.extend([0.8] * (len(current_lora_paths) - len(current_lora_scales)))
                elif len(current_lora_scales) > len(current_lora_paths):
                    print(translate("[INFO] LoRAスケールの数が多すぎるため、不要なものを切り捨てます"))
                    current_lora_scales = current_lora_scales[:len(current_lora_paths)]
                
                print(translate("[INFO] 最終的なLoRA設定:"))
                for i, (path, scale) in enumerate(zip(current_lora_paths, current_lora_scales)):
                    print(f"  - LoRA {i+1}: {os.path.basename(path)} (スケール: {scale})")
        
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
            # テキストエンコーダーの初期化（未ロードの場合）
            if text_encoder is None or text_encoder_2 is None:
                print(translate("\nテキストエンコーダを初めてロードします..."))
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
            
            # 1フレームモード用のlatent_indices特別処理
            all_indices = torch.arange(0, latent_window_size).unsqueeze(0)
            latent_indices = all_indices[:, -1:]  # 最後のインデックスのみ使用
            
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            
            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
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
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')
                
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = translate('1フレームモード: サンプリング中...')
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return
            
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
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            
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
                
                # 画像として保存
                output_filename = os.path.join(outputs_folder, f'{job_id}_oneframe.png')
                Image.fromarray(frame).save(output_filename)
                
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
    
    # 処理完了を通知
    print(translate("\n処理が完了しました"))
    stream.output_queue.push(('end', None))
    return

def process(input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lora_files, lora_scales, use_lora, fp8_optimization, resolution):
    global stream
    assert input_image is not None, translate('No input image!')
    
    # LoRAの状態をログ出力
    if use_lora and has_lora_support:
        print(translate("[INFO] LoRAの使用設定: use_lora = {0}, has_lora_support = {1}").format(use_lora, has_lora_support))
        print(translate("[INFO] lora_files = {0}, 型: {1}").format(lora_files, type(lora_files)))
        print(translate("[INFO] lora_scales = {0}").format(lora_scales))
    
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    
    stream = AsyncStream()
    
    async_run(worker, input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, 
             gpu_memory_preservation, use_teacache, lora_files, lora_scales, 
             None, use_lora, fp8_optimization, resolution)
    
    output_filename = None
    
    while True:
        flag, data = stream.output_queue.next()
        
        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
        
        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
        
        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break

def end_process():
    stream.input_queue.push('end')

css = get_app_css()  # eichi_utilsのスタイルを使用
block = gr.Blocks(css=css).queue()
with block:
    # eichiと同じ半透明度スタイルを使用
    gr.HTML('<h1>FramePack<span class="title-suffix">-oichi</span></h1>')
    
    # まず初期化時にtransformerの状態確認を行う
    if not reload_transformer_if_needed():
        print(translate("警告: transformerの初期ロードに問題がありました。LoRAが正しく適用されない可能性があります。"))
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label=translate("画像"), height=320)
            prompt = gr.Textbox(label=translate("プロンプト"), value='')
            n_prompt = gr.Textbox(label=translate("ネガティブプロンプト"), value='')
            
            with gr.Row():
                start_button = gr.Button(value=translate("生成開始"))
                end_button = gr.Button(value=translate("生成中止"), interactive=False)
            
            with gr.Group():
                # Use Random Seedの初期値
                use_random_seed_default = True
                seed_default = random.randint(0, 2**32 - 1) if use_random_seed_default else 31337
                
                # ランダムシードチェックボックスとシード値入力欄
                use_random_seed = gr.Checkbox(label=translate("ランダムシードを使用"), value=use_random_seed_default)
                seed = gr.Number(label=translate("シード値"), value=seed_default, precision=0)
                steps = gr.Slider(label=translate("ステップ数"), minimum=1, maximum=100, value=25, step=1, info=translate('この値の変更は推奨されません'))
                
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                gs = gr.Slider(label=translate("蒸留CFGスケール"), minimum=1.0, maximum=32.0, value=10.0, step=0.01, info=translate('この値の変更は推奨されません'))
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                
                resolution = gr.Dropdown(
                    label=translate("解像度"),
                    choices=[512, 640, 768, 960, 1080],
                    value=640,
                    info=translate("出力画像の基準解像度。640推奨。960/1080は高負荷・高メモリ消費")
                )
                gpu_memory_preservation = gr.Slider(label=translate("GPUメモリ保持 (GB) (大きいほど遅くなります)"), minimum=6, maximum=128, value=6, step=0.1, info=translate("OOMが発生する場合は値を大きくしてください。値が大きいほど速度が遅くなります。"))
                use_teacache = gr.Checkbox(label=translate("TeaCacheを使用"), value=True, info=translate('より高速ですが、手や指が少し苦手です'))
                
                # LoRA関連のUI
                with gr.Group(visible=has_lora_support) as lora_group:
                    use_lora = gr.Checkbox(label=translate("LoRAを使用"), value=False)
                    
                    # LoRAファイルと設定の行
                    with gr.Row(visible=False) as lora_settings_row:
                        lora_files = gr.File(
                            label=translate("LoRAファイル (.safetensors, .pt, .bin)"),
                            file_types=[".safetensors", ".pt", ".bin"]
                        )
                        lora_scales = gr.Textbox(
                            label=translate("LoRAスケール"),
                            value="0.8",
                            info=translate("カンマ区切りで複数指定可能")
                        )
                    
                    # FP8最適化チェックボックス - LoRa設定行と同様に表示制御
                    with gr.Row(visible=False) as fp8_optimization_row:
                        fp8_optimization = gr.Checkbox(
                            label=translate("FP8最適化"), 
                            value=False,  # デフォルトでオフに設定
                            info=translate("LoRA使用時に有効。モデルを圧縮し高速化しますが、精度が若干低下します")
                        )
                    
                    # LoRAチェックボックスの変更に応じて設定の表示/非表示を切り替え
                    def toggle_lora_settings(use_lora_checked):
                        # LoRA設定行とFP8最適化行の両方を制御
                        return [
                            gr.update(visible=use_lora_checked),  # LoRA設定行
                            gr.update(visible=use_lora_checked)   # FP8最適化行
                        ]
                        
                    use_lora.change(
                        fn=toggle_lora_settings,
                        inputs=[use_lora],
                        outputs=[lora_settings_row, fp8_optimization_row]
                    )
                
        with gr.Column():
            result_image = gr.Image(label=translate("生成結果"), height=512)
            preview_image = gr.Image(label=translate("処理中のプレビュー"), height=200, visible=False)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            
            gr.Markdown(translate("**「1フレーム」モードでは、最初の画像から1フレーム分進んだ中間フレームを生成します。**"))
    
    # シードのランダム化機能
    def set_random_seed(is_checked):
        if is_checked:
            return random.randint(0, 2**32 - 1)
        return gr.update()
    
    # チェックボックス変更時にランダムシードを生成
    use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)
    
    ips = [input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lora_files, lora_scales, use_lora, fp8_optimization, resolution]
    start_button.click(fn=process, inputs=ips, outputs=[result_image, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)
    
    gr.HTML(f'<div style="text-align:center; margin-top:20px;">{translate("FramePack 単一フレーム生成版")}</div>')

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)