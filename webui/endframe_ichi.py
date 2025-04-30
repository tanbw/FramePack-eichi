import os
import sys
sys.path.append(os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './submodules/FramePack'))))

from diffusers_helper.hf_login import login

import os
import random
import time
import subprocess
import copy  # transformer_loraのディープコピー用
# クロスプラットフォーム対応のための条件付きインポート

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--lang", type=str, default='ja', help="Language: ja, zh-tw, en")
args = parser.parse_args()

# Load translations from JSON files
from locales.i18n_extended import (set_lang, translate)
set_lang(args.lang)

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False
import json
import traceback
from datetime import datetime, timedelta

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# LoRAサポートの確認
has_lora_support = False
try:
    import lora_utils
    has_lora_support = True
    print(translate("LoRAサポートが有効です"))
except ImportError:
    print(translate("LoRAサポートが無効です（lora_utilsモジュールがインストールされていません）"))

# 設定モジュールをインポート（ローカルモジュール）
import os.path
from eichi_utils.video_mode_settings import (
    VIDEO_MODE_SETTINGS, get_video_modes, get_video_seconds, get_important_keyframes,
    get_copy_targets, get_max_keyframes_count, get_total_sections, generate_keyframe_guide_html,
    handle_mode_length_change, process_keyframe_change, MODE_TYPE_NORMAL, MODE_TYPE_LOOP
)

# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_settings_file_path,
    get_output_folder_path,
    initialize_settings,
    load_settings,
    save_settings,
    open_output_folder
)

# プリセット管理モジュールをインポート
from eichi_utils.preset_manager import (
    initialize_presets,
    load_presets,
    get_default_startup_prompt,
    save_preset,
    delete_preset
)

# キーフレーム処理モジュールをインポート
from eichi_utils.keyframe_handler import (
    ui_to_code_index,
    code_to_ui_index,
    unified_keyframe_change_handler,
    unified_input_image_change_handler,
    print_keyframe_debug_info
)

# 拡張キーフレーム処理モジュールをインポート
from eichi_utils.keyframe_handler_extended import extended_mode_length_change_handler
import gradio as gr
# UI関連モジュールのインポート
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
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 100

print(translate('Free VRAM {0} GB').format(free_mem_gb))
print(translate('High-VRAM Mode: {0}').format(high_vram))


# 元のモデル読み込みコード
try:
    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
except Exception as e:
    print(translate("モデル読み込みエラー: {0}").format(e))
    print(translate("プログラムを終了します..."))
    import sys
    sys.exit(1)

# 他のモデルも同様に例外処理
try:
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
except Exception as e:
    print(translate("モデル読み込みエラー (追加モデル): {0}").format(e))
    print(translate("プログラムを終了します..."))
    import sys
    sys.exit(1)

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_settings_file_path,
    get_output_folder_path,
    initialize_settings,
    load_settings,
    save_settings,
    open_output_folder
)

# フォルダ構造を先に定義
webui_folder = os.path.dirname(os.path.abspath(__file__))

# 設定保存用フォルダの設定
settings_folder = os.path.join(webui_folder, 'settings')
os.makedirs(settings_folder, exist_ok=True)

# 設定ファイル初期化
initialize_settings()

# ベースパスを定義
base_path = os.path.dirname(os.path.abspath(__file__))

# 設定から出力フォルダを取得
app_settings = load_settings()
output_folder_name = app_settings.get('output_folder', 'outputs')
print(translate("設定から出力フォルダを読み込み: {0}").format(output_folder_name))

# 出力フォルダのフルパスを生成
outputs_folder = get_output_folder_path(output_folder_name)
os.makedirs(outputs_folder, exist_ok=True)


# transformerのLoRA状態管理用のグローバル変数を初期化
class TransformerLoraState:
    def __init__(self):
        self.current_lora_path = None  # 現在適用されているLoRAファイル名
    
    def set_loaded_lora_path(self, lora_file_path):
        self.current_lora_path = lora_file_path
        print(translate("loaded transformer LoRA path: {0}").format(self.current_lora_path))

    def is_lora_loaded(self, lora_file_path):
        # Noneの場合の明示的な処理
        if self.current_lora_path is None or lora_file_path is None:
            return self.current_lora_path is lora_file_path
        # 文字列の比較
        return self.current_lora_path == lora_file_path

transformer_lora_state = TransformerLoraState()


# transformerリロード用の関数を定義
def reload_transformer_model():
    # グローバル変数で状態管理している変数を宣言する
    global transformer
    try:
        # モデルをリロード
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
        transformer.eval()
        transformer.high_quality_fp32_output_for_inference = True
        transformer.to(dtype=torch.bfloat16)
        transformer.requires_grad_(False)
        
        if not high_vram:
            DynamicSwapInstaller.install_model(transformer, device=gpu)
        else:
            transformer.to(gpu)
        print(translate("transformerモデルをリロードしました"))
    except Exception as e:
        print(translate("transformerモデルのリロードエラー: {0}").format(e))


# キーフレーム処理関数は keyframe_handler.py に移動済み

# v1.9.1テスト実装

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf=16, all_padding_value=1.0, end_frame=None, end_frame_strength=1.0, keep_section_videos=False, lora_file=None, lora_scale=0.8, output_dir=None, save_section_frames=False, section_settings=None, use_all_padding=False, use_lora=False, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640):

    # グローバル変数で状態管理している変数を宣言する
    global transformer
    global transformer_lora_state

    # 入力画像または表示されている最後のキーフレーム画像のいずれかが存在するか確認
    has_any_image = (input_image is not None)
    last_visible_section_image = None
    last_visible_section_num = -1

    if not has_any_image and section_settings is not None:
        # 現在の動画長設定から表示されるセクション数を計算
        total_display_sections = None
        try:
            # 動画長を秒数で取得
            seconds = get_video_seconds(total_second_length)

            # フレームサイズ設定からlatent_window_sizeを計算
            current_latent_window_size = 4.5 if frame_size_setting == "0.5秒 (17フレーム)" else 9
            frame_count = current_latent_window_size * 4 - 3

            # セクション数を計算
            total_frames = int(seconds * 30)
            total_display_sections = int(max(round(total_frames / frame_count), 1))
            print(f"[DEBUG] worker内の現在の設定によるセクション数: {total_display_sections}")
        except Exception as e:
            print(f"[ERROR] worker内のセクション数計算エラー: {e}")

        # 有効なセクション番号を収集
        valid_sections = []
        for section in section_settings:
            if section and len(section) > 1 and section[0] is not None and section[1] is not None:
                try:
                    section_num = int(section[0])
                    # 表示セクション数が計算されていれば、それ以下のセクションのみ追加
                    if total_display_sections is None or section_num < total_display_sections:
                        valid_sections.append((section_num, section[1]))
                except (ValueError, TypeError):
                    continue

        # 有効なセクションがあれば、最大の番号（最後のセクション）を探す
        if valid_sections:
            # 番号でソート
            valid_sections.sort(key=lambda x: x[0])
            # 最後のセクションを取得
            last_visible_section_num, last_visible_section_image = valid_sections[-1]
            print(f"[DEBUG] worker内の最後のキーフレーム確認: セクション{last_visible_section_num} (画像あり)")

    has_any_image = has_any_image or (last_visible_section_image is not None)
    if not has_any_image:
        raise ValueError("入力画像または表示されている最後のキーフレーム画像のいずれかが必要です")

    # 入力画像がない場合はキーフレーム画像を使用
    if input_image is None and last_visible_section_image is not None:
        print(f"[INFO] 入力画像が指定されていないため、セクション{last_visible_section_num}のキーフレーム画像を使用します")
        input_image = last_visible_section_image

    # 出力フォルダの設定
    global outputs_folder
    global output_folder_name
    if output_dir and output_dir.strip():
        # 出力フォルダパスを取得
        outputs_folder = get_output_folder_path(output_dir)
        print(translate("出力フォルダを設定: {0}").format(outputs_folder))

        # フォルダ名が現在の設定と異なる場合は設定ファイルを更新
        if output_dir != output_folder_name:
            settings = load_settings()
            settings['output_folder'] = output_dir
            if save_settings(settings):
                output_folder_name = output_dir
                print(translate("出力フォルダ設定を保存しました: {0}").format(output_dir))
    else:
        # デフォルト設定を使用
        outputs_folder = get_output_folder_path(output_folder_name)
        print(translate("デフォルト出力フォルダを使用: {0}").format(outputs_folder))

    # フォルダが存在しない場合は作成
    os.makedirs(outputs_folder, exist_ok=True)
    # 処理時間計測の開始
    process_start_time = time.time()

    # 既存の計算方法を保持しつつ、設定からセクション数も取得する
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # 現在のモードを取得（UIから渡された情報から）
    # セクション数を全セクション数として保存
    total_sections = total_latent_sections

    job_id = generate_timestamp()

    # セクション处理の詳細ログを出力
    if use_all_padding:
        # オールパディングが有効な場合、すべてのセクションで同じ値を使用
        padding_value = round(all_padding_value, 1)  # 小数点1桁に固定（小数点対応）
        latent_paddings = [padding_value] * total_latent_sections
        print(translate("オールパディングを有効化: すべてのセクションにパディング値 {0} を適用").format(padding_value))
    else:
        # 通常のパディング値計算
        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

    # 全セクション数を事前に計算して保存（イテレータの消費を防ぐため）
    latent_paddings_list = list(latent_paddings)
    total_sections = len(latent_paddings_list)
    latent_paddings = latent_paddings_list  # リストに変換したものを使用

    print(translate("\u25a0 セクション生成詳細:"))
    print(translate("  - 生成予定セクション: {0}").format(latent_paddings))
    frame_count = latent_window_size * 4 - 3
    print(translate("  - 各セクションのフレーム数: 約{0}フレーム (latent_window_size: {1})").format(frame_count, latent_window_size))
    print(translate("  - 合計セクション数: {0}").format(total_sections))

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # セクション設定の前処理
        def get_section_settings_map(section_settings):
            """
            section_settings: DataFrame形式のリスト [[番号, 画像, プロンプト], ...]
            → {セクション番号: (画像, プロンプト)} のdict
            """
            result = {}
            if section_settings is not None:
                for row in section_settings:
                    if row and row[0] is not None:
                        sec_num = int(row[0])
                        img = row[1]
                        prm = row[2] if len(row) > 2 else ""
                        result[sec_num] = (img, prm)
            return result

        section_map = get_section_settings_map(section_settings)
        section_numbers_sorted = sorted(section_map.keys()) if section_map else []

        def get_section_info(i_section):
            """
            i_section: int
            section_map: {セクション番号: (画像, プロンプト)}
            指定がなければ次のセクション、なければNone
            """
            if not section_map:
                return None, None, None
            # i_section以降で最初に見つかる設定
            for sec in range(i_section, max(section_numbers_sorted)+1):
                if sec in section_map:
                    img, prm = section_map[sec]
                    return sec, img, prm
            return None, None, None

        # セクション固有のプロンプト処理を行う関数
        def process_section_prompt(i_section, section_map, llama_vec, clip_l_pooler, llama_attention_mask):
            """セクションに固有のプロンプトがあればエンコードして返す
            なければメインプロンプトのエンコード結果を返す
            返り値: (llama_vec, clip_l_pooler, llama_attention_mask)
            """
            if not isinstance(llama_vec, torch.Tensor) or not isinstance(llama_attention_mask, torch.Tensor):
                print(translate("[ERROR] メインプロンプトのエンコード結果またはマスクが不正です"))
                return llama_vec, clip_l_pooler, llama_attention_mask

            # セクション固有のプロンプトがあるか確認
            section_info = None
            if section_map:
                valid_section_nums = [k for k in section_map.keys() if k >= i_section]
                if valid_section_nums:
                    section_num = min(valid_section_nums)
                    section_info = section_map[section_num]

            # セクション固有のプロンプトがあれば使用
            if section_info and len(section_info) > 1:
                _, section_prompt = section_info
                if section_prompt and section_prompt.strip():
                    print(translate("[section_prompt] セクション{0}の専用プロンプトを処理: {1}...").format(i_section, section_prompt[:30]))

                    try:
                        # プロンプト処理
                        section_llama_vec, section_clip_l_pooler = encode_prompt_conds(
                            section_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
                        )

                        # マスクの作成
                        section_llama_vec, section_llama_attention_mask = crop_or_pad_yield_mask(
                            section_llama_vec, length=512
                        )

                        # データ型を明示的にメインプロンプトと合わせる
                        section_llama_vec = section_llama_vec.to(
                            dtype=llama_vec.dtype, device=llama_vec.device
                        )
                        section_clip_l_pooler = section_clip_l_pooler.to(
                            dtype=clip_l_pooler.dtype, device=clip_l_pooler.device
                        )
                        section_llama_attention_mask = section_llama_attention_mask.to(
                            device=llama_attention_mask.device
                        )

                        return section_llama_vec, section_clip_l_pooler, section_llama_attention_mask
                    except Exception as e:
                        print(translate("[ERROR] セクションプロンプト処理エラー: {0}").format(e))

            # 共通プロンプトを使用
            print(translate("[section_prompt] セクション{0}は共通プロンプトを使用します").format(i_section))
            return llama_vec, clip_l_pooler, llama_attention_mask

        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Text encoding ...")))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # テンソルデータのアップロードがあれば読み込み
        uploaded_tensor = None
        if tensor_data_input is not None:
            try:
                tensor_path = tensor_data_input.name
                print(translate("テンソルデータを読み込み: {0}").format(os.path.basename(tensor_path)))
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate('Loading tensor data ...')))))

                # safetensorsからテンソルを読み込み
                tensor_dict = sf.load_file(tensor_path)

                # テンソルに含まれているキーとシェイプを確認
                print(translate("テンソルデータの内容:"))
                for key, tensor in tensor_dict.items():
                    print(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")

                # history_latentsと呼ばれるキーが存在するか確認
                if "history_latents" in tensor_dict:
                    uploaded_tensor = tensor_dict["history_latents"]
                    print(translate("テンソルデータ読み込み成功: shape={0}, dtype={1}").format(uploaded_tensor.shape, uploaded_tensor.dtype))
                    stream.output_queue.push(('progress', (None, translate('Tensor data loaded successfully!'), make_progress_bar_html(10, translate('Tensor data loaded successfully!')))))
                else:
                    print(translate("警告: テンソルデータに 'history_latents' キーが見つかりません"))
            except Exception as e:
                print(translate("テンソルデータ読み込みエラー: {0}").format(e))
                traceback.print_exc()

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Image processing ...")))))

        def preprocess_image(img, resolution=640):
            """入力画像を処理して適切なサイズに変換する"""
            if img is None:
                # 画像がない場合は指定解像度の黒い画像を生成
                img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                height = width = resolution
                return img, img, height, width
                
            H, W, C = img.shape
            # 解像度パラメータを使用してサイズを決定
            height, width = find_nearest_bucket(H, W, resolution=resolution)
            img_np = resize_and_center_crop(img, target_width=width, target_height=height)
            img_pt = torch.from_numpy(img_np).float() / 127.5 - 1
            img_pt = img_pt.permute(2, 0, 1)[None, :, None]
            return img_np, img_pt, height, width

        input_image_np, input_image_pt, height, width = preprocess_image(input_image, resolution=resolution)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("VAE encoding ...")))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        # アップロードされたテンソルがあっても、常に入力画像から通常のエンコーディングを行う
        # テンソルデータは後で後付けとして使用するために保持しておく
        if uploaded_tensor is not None:
            print(translate("アップロードされたテンソルデータを検出: 動画生成後に後方に結合します"))
            # 入力画像がNoneの場合、テンソルからデコードして表示画像を生成
            if input_image is None:
                try:
                    # テンソルの最初のフレームから画像をデコードして表示用に使用
                    preview_latent = uploaded_tensor[:, :, 0:1, :, :].clone()
                    if preview_latent.device != torch.device('cpu'):
                        preview_latent = preview_latent.cpu()
                    if preview_latent.dtype != torch.float16:
                        preview_latent = preview_latent.to(dtype=torch.float16)

                    decoded_image = vae_decode(preview_latent, vae)
                    decoded_image = (decoded_image[0, :, 0] * 127.5 + 127.5).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
                    # デコードした画像を保存
                    Image.fromarray(decoded_image).save(os.path.join(outputs_folder, f'{job_id}_tensor_preview.png'))
                    # デコードした画像を入力画像として設定
                    input_image = decoded_image
                    # 前処理用のデータも生成
                    input_image_np, input_image_pt, height, width = preprocess_image(input_image)
                    print(translate("テンソルからデコードした画像を生成しました: {0}x{1}").format(height, width))
                except Exception as e:
                    print(translate("テンソルからのデコード中にエラーが発生しました: {0}").format(e))
                    # デコードに失敗した場合は通常の処理を続行

            # UI上でテンソルデータの情報を表示
            tensor_info = translate("テンソルデータ ({0}フレーム) を検出しました。動画生成後に後方に結合します。").format(uploaded_tensor.shape[2])
            stream.output_queue.push(('progress', (None, tensor_info, make_progress_bar_html(10, translate('テンソルデータを後方に結合')))))

        # 常に入力画像から通常のエンコーディングを行う
        start_latent = vae_encode(input_image_pt, vae)
        # end_frameも同じタイミングでencode
        if end_frame is not None:
            end_frame_np, end_frame_pt, _, _ = preprocess_image(end_frame, resolution=resolution)
            end_frame_latent = vae_encode(end_frame_pt, vae)
        else:
            end_frame_latent = None

        # create section_latents here
        section_latents = None
        if section_map:
            section_latents = {}
            for sec_num, (img, prm) in section_map.items():
                if img is not None:
                    # 画像をVAE encode
                    img_np, img_pt, _, _ = preprocess_image(img, resolution=resolution)
                    section_latents[sec_num] = vae_encode(img_pt, vae)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("CLIP Vision encoding ...")))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Start sampling ...")))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        # latent_window_sizeが4.5の場合は特別に17フレームとする
        if latent_window_size == 4.5:
            num_frames = 17  # 5 * 4 - 3 = 17
        else:
            num_frames = int(latent_window_size * 4 - 3)

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        # ここでlatent_paddingsを再定義していたのが原因だったため、再定義を削除します


        # -------- LoRA 設定 START ---------

        # LoRAの環境変数設定（PYTORCH_CUDA_ALLOC_CONF）
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            old_env = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            print(translate("CUDA環境変数設定: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (元の値: {0})").format(old_env))

        # LoRA処理
        if use_lora and has_lora_support and (lora_file is not None):
            # LoRAファイルのパスを取得
            lora_path = lora_file.name

            if transformer_lora_state.is_lora_loaded(lora_path):
                print(translate("LoRAはすでに適用されています: {0}").format(lora_path))
            else:
                try:
                    # 明示的な同期ポイントの追加
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # 直接LoRA適用方式を使用
                    from lora_utils.lora_loader import load_and_apply_lora
                    transformer = load_and_apply_lora(
                        transformer,
                        lora_path,
                        lora_scale,
                        device=gpu
                    )
                    print(translate("LoRAを直接適用しました (スケール: {0})").format(lora_scale))
                    transformer_lora_state.set_loaded_lora_path(lora_path)

                    # 診断レポートの出力
                    try:
                        from lora_utils.lora_check_helper import check_lora_applied
                        has_lora, source = check_lora_applied(transformer)
                        print(translate("LoRA適用状況: {0}, 適用方法: {1}").format(has_lora, source))
                    except Exception as diagnostic_error:
                        print(translate("LoRA診断エラー: {0}").format(diagnostic_error))

                    # FP8最適化処理
                    try:
                        if fp8_optimization and transformer_lora_state.is_lora_loaded(lora_path):
                            # FP8サポートのチェック
                            from lora_utils.fp8_optimization_utils import check_fp8_support, optimize_state_dict_with_fp8, apply_fp8_monkey_patch

                            has_e4m3, has_e5m2, has_scaled_mm = check_fp8_support()

                            if not has_e4m3:
                                print(translate("FP8最適化が有効化されていますが、サポートされていません。PyTorch 2.1以上が必要です。"))
                            else:
                                print(translate("FP8最適化を適用します..."))

                                # 状態辞書を取得
                                state_dict = transformer.state_dict()

                                # 最適化のターゲットと除外キーを設定
                                TARGET_KEYS = ["transformer_blocks", "single_transformer_blocks"]
                                EXCLUDE_KEYS = ["norm"]  # LayerNormなどの正規化層を除外

                                # 状態辞書をFP8形式に最適化
                                print(translate("FP8形式で状態辞書を最適化しています..."))
                                state_dict = optimize_state_dict_with_fp8(
                                    state_dict,
                                    gpu,
                                    TARGET_KEYS,
                                    EXCLUDE_KEYS,
                                    move_to_device=False
                                )

                                # モンキーパッチの適用
                                print(translate("FP8モンキーパッチを適用しています..."))
                                use_scaled_mm = has_scaled_mm and has_e5m2
                                apply_fp8_monkey_patch(transformer, state_dict, use_scaled_mm=use_scaled_mm)

                                # 状態辞書を読み込み
                                transformer.load_state_dict(state_dict, strict=True)

                                print(translate("FP8最適化が適用されました！"))
                        else:
                            print(translate("FP8最適化は無効化されています。"))
                    except Exception as e:
                        print(translate("FP8最適化エラー: {0}").format(e))
                        traceback.print_exc()
                        print(translate("FP8最適化に失敗しました。通常モードで続行します。"))
                        # エラー時は元の状態を使用し続ける

                except Exception as e:
                    print(translate("LoRA適用エラー: {0}").format(e))
                    traceback.print_exc()
                    print(translate("LoRA適用に失敗しました。通常モードで続行します。"))

                    # エラー時は元のtransformerを使用（リロード）
                    if not transformer_lora_state.is_lora_loaded(None):
                        reload_transformer_model()
                    transformer_lora_state.set_loaded_lora_path(None)

        else:
            # LoRA未使用時のメッセージ
            if use_lora:
                if not has_lora_support:
                    print(translate("LoRAサポートが無効です。lora_utilsモジュールが必要です。"))
                elif lora_file is None:
                    print(translate("LoRAファイルが指定されていません。通常モードで続行します。"))
            else:
                print(translate("LoRAは使用されません。通常モードで続行します。"))
            
            # 前回Loraが適用されていた場合は、元のtransformerをリロード
            if not transformer_lora_state.is_lora_loaded(None):
                reload_transformer_model()
            transformer_lora_state.set_loaded_lora_path(None)

        # -------- LoRA 設定 END ---------


        for i_section, latent_padding in enumerate(latent_paddings):
            # 先に変数を定義
            is_first_section = i_section == 0

            # オールパディングの場合の特別処理
            if use_all_padding:
                # 最後のセクションの判定
                is_last_section = i_section == len(latent_paddings) - 1

                # 内部処理用に元の値を保存
                orig_padding_value = latent_padding

                # 最後のセクションが0より大きい場合は警告と強制変換
                if is_last_section and float(latent_padding) > 0:
                    print(translate("警告: 最後のセクションのパディング値は内部計算のために0に強制します。"))
                    latent_padding = 0
                elif isinstance(latent_padding, float):
                    # 浮動小数点の場合はそのまま使用（小数点対応）
                    # 小数点1桁に固定のみ行い、丸めは行わない
                    latent_padding = round(float(latent_padding), 1)

                # 値が変更された場合にデバッグ情報を出力
                if float(orig_padding_value) != float(latent_padding):
                    print(translate("パディング値変換: セクション{0}の値を{1}から{2}に変換しました").format(i_section, orig_padding_value, latent_padding))
            else:
                # 通常モードの場合
                is_last_section = latent_padding == 0

            use_end_latent = is_last_section and end_frame is not None
            latent_padding_size = int(latent_padding * latent_window_size)

            # 定義後にログ出力
            padding_info = translate("設定パディング値: {0}").format(all_padding_value) if use_all_padding else translate("パディング値: {0}").format(latent_padding)
            print(translate("\n■ セクション{0}の処理開始 ({1})").format(i_section, padding_info))
            print(translate("  - 現在の生成フレーム数: {0}フレーム").format(total_generated_latent_frames * 4 - 3))
            print(translate("  - 生成予定フレーム数: {0}フレーム").format(num_frames))
            print(translate("  - 最初のセクション?: {0}").format(is_first_section))
            print(translate("  - 最後のセクション?: {0}").format(is_last_section))
            # set current_latent here
            # セクションごとのlatentを使う場合
            if section_map and section_latents is not None and len(section_latents) > 0:
                # i_section以上で最小のsection_latentsキーを探す
                valid_keys = [k for k in section_latents.keys() if k >= i_section]
                if valid_keys:
                    use_key = min(valid_keys)
                    current_latent = section_latents[use_key]
                    print(translate("[section_latent] section {0}: use section {1} latent (section_map keys: {2})").format(i_section, use_key, list(section_latents.keys())))
                    print(translate("[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}").format(id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()))
                else:
                    current_latent = start_latent
                    print(translate("[section_latent] section {0}: use start_latent (no section_latent >= {1})").format(i_section, i_section))
                    print(translate("[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}").format(id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()))
            else:
                current_latent = start_latent
                print(translate("[section_latent] section {0}: use start_latent (no section_latents)").format(i_section))
                print(translate("[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}").format(id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()))

            if is_first_section and end_frame_latent is not None:
                # EndFrame影響度設定を適用（デフォルトは1.0=通常の影響）
                if end_frame_strength != 1.0:
                    # 影響度を適用した潜在表現を生成
                    # 値が小さいほど影響が弱まるように単純な乗算を使用
                    # end_frame_strength=1.0のときは1.0倍（元の値）
                    # end_frame_strength=0.01のときは0.01倍（影響が非常に弱い）
                    modified_end_frame_latent = end_frame_latent * end_frame_strength
                    print(translate("EndFrame影響度を{0}に設定（最終フレームの影響が{1}倍）").format(f"{end_frame_strength:.2f}", f"{end_frame_strength:.2f}"))
                    history_latents[:, :, 0:1, :, :] = modified_end_frame_latent
                else:
                    # 通常の処理（通常の影響）
                    history_latents[:, :, 0:1, :, :] = end_frame_latent

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            # セクション固有のプロンプトがあれば使用する
            current_llama_vec, current_clip_l_pooler, current_llama_attention_mask = process_section_prompt(i_section, section_map, llama_vec, clip_l_pooler, llama_attention_mask)

            print(translate('latent_padding_size = {0}, is_last_section = {1}').format(latent_padding_size, is_last_section))


            # COMMENTED OUT: セクション処理前のメモリ解放（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            # latent_window_sizeが4.5の場合は特別に5を使用
            effective_window_size = 5 if latent_window_size == 4.5 else int(latent_window_size)
            indices = torch.arange(0, sum([1, latent_padding_size, effective_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, effective_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = current_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                # GPUメモリ保存値を明示的に浮動小数点に変換
                preserved_memory = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
                print(translate('Setting transformer memory preservation to: {0} GB').format(preserved_memory))
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory)

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
                hint = translate('Sampling {0}/{1}').format(current_step, steps)
                # セクション情報を追加（現在のセクション/全セクション）
                section_info = translate('セクション: {0}/{1}').format(i_section+1, total_sections)
                desc = f"{section_info} " + translate('生成フレーム数: {total_generated_latent_frames}, 動画長: {video_length:.2f} 秒 (FPS-30). 動画が生成中です ...').format(section_info=section_info, total_generated_latent_frames=int(max(0, total_generated_latent_frames * 4 - 3)), video_length=max(0, (total_generated_latent_frames * 4 - 3) / 30))
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
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=current_llama_vec,  # セクションごとのプロンプトを使用
                prompt_embeds_mask=current_llama_attention_mask,  # セクションごとのマスクを使用
                prompt_poolers=current_clip_l_pooler,  # セクションごとのプロンプトを使用
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

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                # 減圧時に使用するGPUメモリ値も明示的に浮動小数点に設定
                preserved_memory_offload = 8.0  # こちらは固定値のまま
                print(translate('Offloading transformer with memory preservation: {0} GB').format(preserved_memory_offload))
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory_offload)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            # COMMENTED OUT: VAEデコード前のメモリクリア（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()
            #     print(f"VAEデコード前メモリ: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                # latent_window_sizeが4.5の場合は特別に5を使用
                if latent_window_size == 4.5:
                    section_latent_frames = 11 if is_last_section else 10  # 5 * 2 + 1 = 11, 5 * 2 = 10
                    overlapped_frames = 17  # 5 * 4 - 3 = 17
                else:
                    section_latent_frames = int(latent_window_size * 2 + 1) if is_last_section else int(latent_window_size * 2)
                    overlapped_frames = int(latent_window_size * 4 - 3)

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            # COMMENTED OUT: 明示的なCPU転送と不要テンソルの削除（処理速度向上のため）
            # if torch.cuda.is_available():
            #     # 必要なデコード後、明示的にキャッシュをクリア
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            # 各セクションの最終フレームを静止画として保存（セクション番号付き）
            if save_section_frames and history_pixels is not None:
                try:
                    if i_section == 0 or current_pixels is None:
                        # 最初のセクションは history_pixels の最後
                        last_frame = history_pixels[0, :, -1, :, :]
                    else:
                        # 2セクション目以降は current_pixels の最後
                        last_frame = current_pixels[0, :, -1, :, :]
                    last_frame = einops.rearrange(last_frame, 'c h w -> h w c')
                    last_frame = last_frame.cpu().numpy()
                    last_frame = np.clip((last_frame * 127.5 + 127.5), 0, 255).astype(np.uint8)
                    last_frame = resize_and_center_crop(last_frame, target_width=width, target_height=height)
                    if is_first_section and end_frame is None:
                        Image.fromarray(last_frame).save(os.path.join(outputs_folder, f'{job_id}_{i_section}_end.png'))
                    else:
                        Image.fromarray(last_frame).save(os.path.join(outputs_folder, f'{job_id}_{i_section}.png'))
                except Exception as e:
                    print(translate("[WARN] セクション{0}最終フレーム画像保存時にエラー: {1}").format(i_section, e))

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(translate('Decoded. Current latent shape {0}; pixel shape {1}').format(real_history_latents.shape, history_pixels.shape))

            # COMMENTED OUT: セクション処理後の明示的なメモリ解放（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()
            #     import gc
            #     gc.collect()
            #     memory_allocated = torch.cuda.memory_allocated()/1024**3
            #     memory_reserved = torch.cuda.memory_reserved()/1024**3
            #     print(f"セクション後メモリ状態: 割当={memory_allocated:.2f}GB, 予約={memory_reserved:.2f}GB")

            print(translate("■ セクション{0}の処理完了").format(i_section))
            print(translate("  - 現在の累計フレーム数: {0}フレーム").format(int(max(0, total_generated_latent_frames * 4 - 3))))
            print(translate("  - レンダリング時間: {0}秒").format(f"{max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f}"))
            print(translate("  - 出力ファイル: {0}").format(output_filename))

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                combined_output_filename = None
                # 全セクション処理完了後、テンソルデータを後方に結合
                if uploaded_tensor is not None:
                    try:
                        original_frames = real_history_latents.shape[2]  # 元のフレーム数を記録
                        uploaded_frames = uploaded_tensor.shape[2]  # アップロードされたフレーム数

                        print(translate("テンソルデータを後方に結合します: アップロードされたフレーム数 = {uploaded_frames}").format(uploaded_frames=uploaded_frames))
                        # UI上で進捗状況を更新
                        stream.output_queue.push(('progress', (None, translate("テンソルデータ({uploaded_frames}フレーム)の結合を開始します...").format(uploaded_frames=uploaded_frames), make_progress_bar_html(80, translate('テンソルデータ結合準備')))))

                        # テンソルデータを後方に結合する前に、互換性チェック
                        # デバッグログを追加して詳細を出力
                        print(translate("[DEBUG] テンソルデータの形状: {0}, 生成データの形状: {1}").format(uploaded_tensor.shape, real_history_latents.shape))
                        print(translate("[DEBUG] テンソルデータの型: {0}, 生成データの型: {1}").format(uploaded_tensor.dtype, real_history_latents.dtype))
                        print(translate("[DEBUG] テンソルデータのデバイス: {0}, 生成データのデバイス: {1}").format(uploaded_tensor.device, real_history_latents.device))

                        if uploaded_tensor.shape[3] != real_history_latents.shape[3] or uploaded_tensor.shape[4] != real_history_latents.shape[4]:
                            print(translate("警告: テンソルサイズが異なります: アップロード={0}, 現在の生成={1}").format(uploaded_tensor.shape, real_history_latents.shape))
                            print(translate("テンソルサイズの不一致のため、前方結合をスキップします"))
                            stream.output_queue.push(('progress', (None, translate("テンソルサイズの不一致のため、前方結合をスキップしました"), make_progress_bar_html(85, translate('互換性エラー')))))
                        else:
                            # デバイスとデータ型を合わせる
                            processed_tensor = uploaded_tensor.clone()
                            if processed_tensor.device != real_history_latents.device:
                                processed_tensor = processed_tensor.to(real_history_latents.device)
                            if processed_tensor.dtype != real_history_latents.dtype:
                                processed_tensor = processed_tensor.to(dtype=real_history_latents.dtype)

                            # 元の動画を品質を保ちつつ保存
                            original_output_filename = os.path.join(outputs_folder, f'{job_id}_original.mp4')
                            save_bcthw_as_mp4(history_pixels, original_output_filename, fps=30, crf=mp4_crf)
                            print(translate("元の動画を保存しました: {original_output_filename}").format(original_output_filename=original_output_filename))

                            # 元データのコピーを取得
                            combined_history_latents = real_history_latents.clone()
                            combined_history_pixels = history_pixels.clone() if history_pixels is not None else None

                            # 各チャンクの処理前に明示的にメモリ解放
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                print(translate("[MEMORY] チャンク処理前のGPUメモリ確保状態: {memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3))

                            # VAEをGPUに移動
                            if not high_vram and vae.device != torch.device('cuda'):
                                print(translate("[SETUP] VAEをGPUに移動: {0} → cuda").format(vae.device))
                                vae.to('cuda')

                            # 各チャンクを処理
                            # チャンクサイズを設定(各セクションと同等のサイズにする)
                            chunk_size = min(5, uploaded_frames)  # 最大チャンクサイズを5フレームに設定（メモリ使用量を減らすため）

                            # チャンク数を計算
                            num_chunks = (uploaded_frames + chunk_size - 1) // chunk_size

                            # テンソルデータの詳細を出力
                            print(translate("[DEBUG] テンソルデータの詳細分析:"))
                            print(translate("  - 形状: {0}").format(processed_tensor.shape))
                            print(translate("  - 型: {0}").format(processed_tensor.dtype))
                            print(translate("  - デバイス: {0}").format(processed_tensor.device))
                            print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(processed_tensor.min().item(), processed_tensor.max().item(), processed_tensor.mean().item()))
                            print(translate("  - チャンク数: {0}, チャンクサイズ: {1}").format(num_chunks, chunk_size))
                            tensor_size_mb = (processed_tensor.element_size() * processed_tensor.nelement()) / (1024 * 1024)
                            print(translate("  - テンソルデータ全体サイズ: {0:.2f} MB").format(tensor_size_mb))
                            print(translate("  - フレーム数: {0}フレーム（制限無し）").format(uploaded_frames))
                            # 各チャンクを処理
                            for chunk_idx in range(num_chunks):
                                chunk_start = chunk_idx * chunk_size
                                chunk_end = min(chunk_start + chunk_size, uploaded_frames)
                                chunk_frames = chunk_end - chunk_start

                                # 進捗状況を更新
                                chunk_progress = (chunk_idx + 1) / num_chunks * 100
                                progress_message = translate("テンソルデータ結合中: チャンク {0}/{1} (フレーム {2}-{3}/{4})").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames)
                                stream.output_queue.push(('progress', (None, progress_message, make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('テンソルデータ処理中')))))

                                # 現在のチャンクを取得
                                current_chunk = processed_tensor[:, :, chunk_start:chunk_end, :, :]
                                print(translate("チャンク{0}/{1}処理中: フレーム {2}-{3}/{4}").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))

                                # メモリ状態を出力
                                if torch.cuda.is_available():
                                    print(translate("[MEMORY] チャンク{0}処理前のGPUメモリ: {1:.2f}GB/{2:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))
                                    # メモリキャッシュをクリア
                                    torch.cuda.empty_cache()

                                try:
                                    # 各チャンク処理前にGPUメモリを解放
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                                        import gc
                                        gc.collect()
                                    # チャンクをデコード
                                    # VAEデコードは時間がかかるため、進行中であることを表示
                                    print(translate("チャンク{0}のVAEデコード開始...").format(chunk_idx+1))
                                    stream.output_queue.push(('progress', (None, translate("チャンク{0}/{1}のVAEデコード中...").format(chunk_idx+1, num_chunks), make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('デコード処理')))))

                                    # VAEデコード前にテンソル情報を詳しく出力
                                    print(translate("[DEBUG] チャンク{0}のデコード前情報:").format(chunk_idx+1))
                                    print(translate("  - 形状: {0}").format(current_chunk.shape))
                                    print(translate("  - 型: {0}").format(current_chunk.dtype))
                                    print(translate("  - デバイス: {0}").format(current_chunk.device))
                                    print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(current_chunk.min().item(), current_chunk.max().item(), current_chunk.mean().item()))

                                    # 明示的にデバイスを合わせる
                                    if current_chunk.device != vae.device:
                                        print(translate("  - デバイスをVAEと同じに変更: {0} → {1}").format(current_chunk.device, vae.device))
                                        current_chunk = current_chunk.to(vae.device)

                                    # 型を明示的に合わせる
                                    if current_chunk.dtype != torch.float16:
                                        print(translate("  - データ型をfloat16に変更: {0} → torch.float16").format(current_chunk.dtype))
                                        current_chunk = current_chunk.to(dtype=torch.float16)

                                    # VAEデコード処理
                                    chunk_pixels = vae_decode(current_chunk, vae).cpu()
                                    print(translate("チャンク{0}のVAEデコード完了 (フレーム数: {1})").format(chunk_idx+1, chunk_frames))

                                    # デコード後のピクセルデータ情報を出力
                                    print(translate("[DEBUG] チャンク{0}のデコード結果:").format(chunk_idx+1))
                                    print(translate("  - 形状: {0}").format(chunk_pixels.shape))
                                    print(translate("  - 型: {0}").format(chunk_pixels.dtype))
                                    print(translate("  - デバイス: {0}").format(chunk_pixels.device))
                                    print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(chunk_pixels.min().item(), chunk_pixels.max().item(), chunk_pixels.mean().item()))

                                    # メモリ使用量を出力
                                    if torch.cuda.is_available():
                                        print(translate("[MEMORY] チャンク{0}デコード後のGPUメモリ: {1:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3))

                                    # 結合する
                                    if combined_history_pixels is None:
                                        # 初回のチャンクの場合はそのまま設定
                                        combined_history_pixels = chunk_pixels
                                    else:
                                        # 2回目以降は結合
                                        print(translate("[DEBUG] 結合前の情報:"))
                                        print(translate("  - 既存: {0}, 型: {1}, デバイス: {2}").format(combined_history_pixels.shape, combined_history_pixels.dtype, combined_history_pixels.device))
                                        print(translate("  - 新規: {0}, 型: {1}, デバイス: {2}").format(chunk_pixels.shape, chunk_pixels.dtype, chunk_pixels.device))

                                        # 既存データと新規データで型とデバイスを揃える
                                        if combined_history_pixels.dtype != chunk_pixels.dtype:
                                            print(translate("  - データ型の不一致を修正: {0} → {1}").format(combined_history_pixels.dtype, chunk_pixels.dtype))
                                            combined_history_pixels = combined_history_pixels.to(dtype=chunk_pixels.dtype)

                                        # 両方とも必ずCPUに移動してから結合
                                        if combined_history_pixels.device != torch.device('cpu'):
                                            combined_history_pixels = combined_history_pixels.cpu()
                                        if chunk_pixels.device != torch.device('cpu'):
                                            chunk_pixels = chunk_pixels.cpu()

                                        # 結合処理
                                        combined_history_pixels = torch.cat([combined_history_pixels, chunk_pixels], dim=2)

                                    # 結合後のフレーム数を確認
                                    current_total_frames = combined_history_pixels.shape[2]
                                    print(translate("チャンク{0}の結合完了: 現在の組み込みフレーム数 = {1}").format(chunk_idx+1, current_total_frames))

                                    # 中間結果の保存（チャンクごとに保存すると効率が悪いので、最終チャンクのみ保存）
                                    if chunk_idx == num_chunks - 1 or (chunk_idx > 0 and (chunk_idx + 1) % 5 == 0):
                                        # 5チャンクごと、または最後のチャンクで保存
                                        interim_output_filename = os.path.join(outputs_folder, f'{job_id}_combined_interim_{chunk_idx+1}.mp4')
                                        print(translate("中間結果を保存中: チャンク{0}/{1}").format(chunk_idx+1, num_chunks))
                                        stream.output_queue.push(('progress', (None, translate("中間結果のMP4変換中... (チャンク{0}/{1})").format(chunk_idx+1, num_chunks), make_progress_bar_html(int(85 + chunk_progress * 0.1), translate('MP4保存中')))))

                                        # MP4として保存
                                        save_bcthw_as_mp4(combined_history_pixels, interim_output_filename, fps=30, crf=mp4_crf)
                                        print(translate("中間結果を保存しました: {0}").format(interim_output_filename))

                                        # 結合した動画をUIに反映するため、出力フラグを立てる
                                        stream.output_queue.push(('file', interim_output_filename))
                                except Exception as e:
                                    print(translate("チャンク{0}の処理中にエラーが発生しました: {1}").format(chunk_idx+1, e))
                                    traceback.print_exc()

                                    # エラー情報の詳細な出力
                                    print(translate("[ERROR] 詳細エラー情報:"))
                                    print(translate("  - チャンク情報: {0}/{1}, フレーム {2}-{3}/{4}").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))
                                    if 'current_chunk' in locals():
                                        print(translate("  - current_chunk: shape={0}, dtype={1}, device={2}").format(current_chunk.shape, current_chunk.dtype, current_chunk.device))
                                    if 'vae' in globals():
                                        print(translate("  - VAE情報: device={0}, dtype={1}").format(vae.device, next(vae.parameters()).dtype))

                                    # GPUメモリ情報
                                    if torch.cuda.is_available():
                                        print(translate("  - GPU使用量: {0:.2f}GB/{1:.2f}GB").format(torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))

                                    stream.output_queue.push(('progress', (None, translate("エラー: チャンク{0}の処理に失敗しました - {1}").format(chunk_idx+1, str(e)), make_progress_bar_html(90, translate('エラー')))))
                                    break

                            # 処理完了後に明示的にメモリ解放
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                print(translate("[MEMORY] チャンク処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))

                            # 全チャンクの処理が完了したら、最終的な結合動画を保存
                            if combined_history_pixels is not None:
                                # 結合された最終結果の情報を出力
                                print(translate("[DEBUG] 最終結合結果:"))
                                print(translate("  - 形状: {0}").format(combined_history_pixels.shape))
                                print(translate("  - 型: {0}").format(combined_history_pixels.dtype))
                                print(translate("  - デバイス: {0}").format(combined_history_pixels.device))
                                # 最終結果の保存
                                print(translate("最終結果を保存中: 全{0}チャンク完了").format(num_chunks))
                                stream.output_queue.push(('progress', (None, translate("結合した動画をMP4に変換中..."), make_progress_bar_html(95, translate('最終MP4変換処理')))))

                                # 最終的な結合ファイル名
                                combined_output_filename = os.path.join(outputs_folder, f'{job_id}_combined.mp4')

                                # MP4として保存
                                save_bcthw_as_mp4(combined_history_pixels, combined_output_filename, fps=30, crf=mp4_crf)
                                print(translate("最終結果を保存しました: {0}").format(combined_output_filename))
                                print(translate("結合動画の保存場所: {0}").format(os.path.abspath(combined_output_filename)))

                                # 中間ファイルの削除処理
                                print(translate("中間ファイルの削除を開始します..."))
                                deleted_files = []
                                try:
                                    # 現在のジョブIDに関連する中間ファイルを正規表現でフィルタリング
                                    import re
                                    interim_pattern = re.compile(f'{job_id}_combined_interim_\d+\.mp4')

                                    for filename in os.listdir(outputs_folder):
                                        if interim_pattern.match(filename):
                                            interim_path = os.path.join(outputs_folder, filename)
                                            try:
                                                os.remove(interim_path)
                                                deleted_files.append(filename)
                                                print(translate("  - 中間ファイルを削除しました: {0}").format(filename))
                                            except Exception as e:
                                                print(translate("  - ファイル削除エラー ({0}): {1}").format(filename, e))

                                    if deleted_files:
                                        print(translate("合計 {0} 個の中間ファイルを削除しました").format(len(deleted_files)))
                                        # 削除ファイル名をユーザーに表示
                                        files_str = ', '.join(deleted_files)
                                        stream.output_queue.push(('progress', (None, translate("中間ファイルを削除しました: {0}").format(files_str), make_progress_bar_html(97, translate('クリーンアップ完了')))))
                                    else:
                                        print(translate("削除対象の中間ファイルは見つかりませんでした"))
                                except Exception as e:
                                    print(translate("中間ファイル削除中にエラーが発生しました: {0}").format(e))
                                    traceback.print_exc()

                                # 結合した動画をUIに反映するため、出力フラグを立てる
                                stream.output_queue.push(('file', combined_output_filename))

                                # 結合後の全フレーム数を計算して表示
                                combined_frames = combined_history_pixels.shape[2]
                                combined_size_mb = (combined_history_pixels.element_size() * combined_history_pixels.nelement()) / (1024 * 1024)
                                print(translate("結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム").format(uploaded_frames, original_frames, combined_frames))
                                print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                                print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                                # UI上で完了メッセージを表示
                                stream.output_queue.push(('progress', (None, translate("テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒) - サイズ制限なし").format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), make_progress_bar_html(100, translate('結合完了')))))
                            else:
                                print(translate("テンソルデータの結合に失敗しました。"))
                                stream.output_queue.push(('progress', (None, translate("テンソルデータの結合に失敗しました。"), make_progress_bar_html(100, translate('エラー')))))

                            # 正しく結合された動画はすでに生成済みなので、ここでの処理は不要

                            # この部分の処理はすでに上記のチャンク処理で完了しているため不要

                            # real_history_latentsとhistory_pixelsを結合済みのものに更新
                            real_history_latents = combined_history_latents
                            history_pixels = combined_history_pixels

                            # 結合した動画をUIに反映するため、出力フラグを立てる
                            stream.output_queue.push(('file', combined_output_filename))

                            # 出力ファイル名を更新
                            output_filename = combined_output_filename

                            # 結合後の全フレーム数を計算して表示
                            combined_frames = combined_history_pixels.shape[2]
                            combined_size_mb = (combined_history_pixels.element_size() * combined_history_pixels.nelement()) / (1024 * 1024)
                            print(translate("結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム").format(uploaded_frames, original_frames, combined_frames))
                            print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                            print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                            # UI上で完了メッセージを表示
                            stream.output_queue.push(('progress', (None, translate("テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒)").format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), make_progress_bar_html(100, translate('結合完了')))))
                    except Exception as e:
                        print(translate("テンソルデータ結合中にエラーが発生しました: {0}").format(e))
                        traceback.print_exc()
                        stream.output_queue.push(('progress', (None, translate("エラー: テンソルデータ結合に失敗しました - {0}").format(str(e)), make_progress_bar_html(100, translate('エラー')))))

                # 処理終了時に通知
                if HAS_WINSOUND:
                    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                else:
                    print(translate("\n✓ 処理が完了しました！"))  # Linuxでの代替通知

                # メモリ解放を明示的に実行
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    print(translate("[MEMORY] 処理完了後のメモリクリア: {memory:.2f}GB/{total_memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3, total_memory=torch.cuda.get_device_properties(0).total_memory/1024**3))

                # テンソルデータの保存処理
                if save_tensor_data:
                    try:
                        # 結果のテンソルを保存するファイルパス
                        tensor_file_path = os.path.join(outputs_folder, f'{job_id}.safetensors')

                        # 保存するデータを準備
                        print(translate("=== テンソルデータ保存処理開始 ==="))
                        print(translate("保存対象フレーム数: {frames}").format(frames=real_history_latents.shape[2]))

                        # サイズ制限を完全に撤廃し、全フレームを保存
                        tensor_to_save = real_history_latents.clone().cpu()

                        # テンソルデータの保存サイズの概算
                        tensor_size_mb = (tensor_to_save.element_size() * tensor_to_save.nelement()) / (1024 * 1024)

                        print(translate("テンソルデータを保存中... shape: {shape}, フレーム数: {frames}, サイズ: {size:.2f} MB").format(shape=tensor_to_save.shape, frames=tensor_to_save.shape[2], size=tensor_size_mb))
                        stream.output_queue.push(('progress', (None, translate('テンソルデータを保存中... ({frames}フレーム)').format(frames=tensor_to_save.shape[2]), make_progress_bar_html(95, translate('テンソルデータの保存')))))

                        # メタデータの準備（フレーム数も含める）
                        metadata = torch.tensor([height, width, tensor_to_save.shape[2]], dtype=torch.int32)

                        # safetensors形式で保存
                        tensor_dict = {
                            "history_latents": tensor_to_save,
                            "metadata": metadata
                        }
                        sf.save_file(tensor_dict, tensor_file_path)

                        print(translate("テンソルデータを保存しました: {path}").format(path=tensor_file_path))
                        print(translate("保存済みテンソルデータ情報: {frames}フレーム, {size:.2f} MB").format(frames=tensor_to_save.shape[2], size=tensor_size_mb))
                        print(translate("=== テンソルデータ保存処理完了 ==="))
                        stream.output_queue.push(('progress', (None, translate("テンソルデータが保存されました: {path} ({frames}フレーム, {size:.2f} MB)").format(path=os.path.basename(tensor_file_path), frames=tensor_to_save.shape[2], size=tensor_size_mb), make_progress_bar_html(100, translate('処理完了')))))

                        # アップロードされたテンソルデータがあれば、それも結合したものを保存する
                        if tensor_data_input is not None and uploaded_tensor is not None:
                            try:
                                # アップロードされたテンソルデータのファイル名を取得
                                uploaded_tensor_filename = os.path.basename(tensor_data_input.name)
                                tensor_combined_path = os.path.join(outputs_folder, f'{job_id}_combined_tensors.safetensors')

                                print(translate("=== テンソルデータ結合処理開始 ==="))
                                print(translate("生成テンソルと入力テンソルを結合して保存します"))
                                print(translate("生成テンソル: {frames}フレーム").format(frames=tensor_to_save.shape[2]))
                                print(translate("入力テンソル: {frames}フレーム").format(frames=uploaded_tensor.shape[2]))

                                # データ型とデバイスを統一
                                if uploaded_tensor.dtype != tensor_to_save.dtype:
                                    uploaded_tensor = uploaded_tensor.to(dtype=tensor_to_save.dtype)
                                if uploaded_tensor.device != tensor_to_save.device:
                                    uploaded_tensor = uploaded_tensor.to(device=tensor_to_save.device)

                                # サイズチェック
                                if uploaded_tensor.shape[3] != tensor_to_save.shape[3] or uploaded_tensor.shape[4] != tensor_to_save.shape[4]:
                                    print(translate("警告: テンソルサイズが一致しないため結合できません: {uploaded_shape} vs {tensor_shape}").format(uploaded_shape=uploaded_tensor.shape, tensor_shape=tensor_to_save.shape))
                                else:
                                    # 結合（生成テンソルの後にアップロードされたテンソルを追加）
                                    combined_tensor = torch.cat([tensor_to_save, uploaded_tensor], dim=2)
                                    combined_frames = combined_tensor.shape[2]
                                    combined_size_mb = (combined_tensor.element_size() * combined_tensor.nelement()) / (1024 * 1024)

                                    # メタデータ更新
                                    combined_metadata = torch.tensor([height, width, combined_frames], dtype=torch.int32)

                                    # 結合したテンソルを保存
                                    combined_tensor_dict = {
                                        "history_latents": combined_tensor,
                                        "metadata": combined_metadata
                                    }
                                    sf.save_file(combined_tensor_dict, tensor_combined_path)

                                    print(translate("結合テンソルを保存しました: {path}").format(path=tensor_combined_path))
                                    print(translate("結合テンソル情報: 合計{0}フレーム ({1}+{2}), {3:.2f} MB").format(frames, tensor_to_save.shape[2], uploaded_tensor.shape[2], size))
                                    print(translate("=== テンソルデータ結合処理完了 ==="))
                                    stream.output_queue.push(('progress', (None, translate("テンソルデータ結合が保存されました: 合計{frames}フレーム").format(frames=combined_frames), make_progress_bar_html(100, translate('結合テンソル保存完了')))))
                            except Exception as e:
                                print(translate("テンソルデータ結合保存エラー: {0}").format(e))
                                traceback.print_exc()
                    except Exception as e:
                        print(translate("テンソルデータ保存エラー: {0}").format(e))
                        traceback.print_exc()
                        stream.output_queue.push(('progress', (None, translate("テンソルデータの保存中にエラーが発生しました。"), make_progress_bar_html(100, translate('処理完了')))))

                # 全体の処理時間を計算
                process_end_time = time.time()
                total_process_time = process_end_time - process_start_time
                hours, remainder = divmod(total_process_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = ""
                if hours > 0:
                    time_str = translate("{0}時間 {1}分 {2}秒").format(int(hours), int(minutes), f"{seconds:.1f}")
                elif minutes > 0:
                    time_str = translate("{0}分 {1}秒").format(int(minutes), f"{seconds:.1f}")
                else:
                    time_str = translate("{0:.1f}秒").format(seconds)
                print(translate("\n全体の処理時間: {0}").format(time_str))

                # 完了メッセージの設定（結合有無によって変更）
                if combined_output_filename is not None:
                    # テンソル結合が成功した場合のメッセージ
                    combined_filename_only = os.path.basename(combined_output_filename)
                    completion_message = translate("すべてのセクション({sections}/{total_sections})が完了しました。テンソルデータとの後方結合も完了しました。結合ファイル名: {filename}\n全体の処理時間: {time}").format(sections=sections, total_sections=total_sections, filename=combined_filename_only, time=time_str)
                    # 最終的な出力ファイルを結合したものに変更
                    output_filename = combined_output_filename
                else:
                    # 通常の完了メッセージ
                    completion_message = translate("すべてのセクション({sections}/{total_sections})が完了しました。全体の処理時間: {time}").format(sections=total_sections, total_sections=total_sections, time=time_str)

                stream.output_queue.push(('progress', (None, completion_message, make_progress_bar_html(100, translate('処理完了')))))

                # 中間ファイルの削除処理
                if not keep_section_videos:
                    # 最終動画のフルパス
                    final_video_path = output_filename
                    final_video_name = os.path.basename(final_video_path)
                    # job_id部分を取得（タイムスタンプ部分）
                    job_id_part = job_id

                    # ディレクトリ内のすべてのファイルを取得
                    files = os.listdir(outputs_folder)
                    deleted_count = 0

                    for file in files:
                        # 同じjob_idを持つMP4ファイルかチェック
                        # 結合ファイル('combined'を含む)は消さないように保護
                        if file.startswith(job_id_part) and file.endswith('.mp4') \
                           and file != final_video_name \
                           and 'combined' not in file:  # combinedファイルは保護
                            file_path = os.path.join(outputs_folder, file)
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                print(translate("[削除] 中間ファイル: {0}").format(file))
                            except Exception as e:
                                print(translate("[エラー] ファイル削除時のエラー {0}: {1}").format(file, e))

                    if deleted_count > 0:
                        print(translate("[済] {0}個の中間ファイルを削除しました。最終ファイルは保存されています: {1}").format(deleted_count, final_video_name))
                        final_message = translate("中間ファイルを削除しました。最終動画と結合動画は保存されています。")
                        stream.output_queue.push(('progress', (None, final_message, make_progress_bar_html(100, translate('処理完了')))))

                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


# 画像のバリデーション関数
def validate_images(input_image, section_settings, length_radio=None, frame_size_radio=None):
    """入力画像または画面に表示されている最後のキーフレーム画像のいずれかが有効かを確認する"""
    # 入力画像をチェック
    if input_image is not None:
        return True, ""

    # 現在の設定から表示すべきセクション数を計算
    total_display_sections = None
    if length_radio is not None and frame_size_radio is not None:
        try:
            # 動画長を秒数で取得
            seconds = get_video_seconds(length_radio.value)

            # フレームサイズ設定からlatent_window_sizeを計算
            latent_window_size = 4.5 if frame_size_radio.value == translate("0.5秒 (17フレーム)") else 9
            frame_count = latent_window_size * 4 - 3

            # セクション数を計算
            total_frames = int(seconds * 30)
            total_display_sections = int(max(round(total_frames / frame_count), 1))
            print(translate("[DEBUG] 現在の設定によるセクション数: {0}").format(total_display_sections))
        except Exception as e:
            print(translate("[ERROR] セクション数計算エラー: {0}").format(e))

    # 入力画像がない場合、表示されているセクションの中で最後のキーフレーム画像をチェック
    last_visible_section_image = None
    last_visible_section_num = -1

    if section_settings is not None:
        # 有効なセクション番号を収集
        valid_sections = []
        for section in section_settings:
            if section and len(section) > 1 and section[0] is not None:
                try:
                    section_num = int(section[0])
                    # 表示セクション数が計算されていれば、それ以下のセクションのみ追加
                    if total_display_sections is None or section_num < total_display_sections:
                        valid_sections.append((section_num, section[1]))
                except (ValueError, TypeError):
                    continue

        # 有効なセクションがあれば、最大の番号（最後のセクション）を探す
        if valid_sections:
            # 番号でソート
            valid_sections.sort(key=lambda x: x[0])
            # 最後のセクションを取得
            last_visible_section_num, last_visible_section_image = valid_sections[-1]

            print(translate("[DEBUG] 最後のキーフレーム確認: セクション{0} (画像あり: {1})").format(last_visible_section_num, last_visible_section_image is not None))

    # 最後のキーフレーム画像があればOK
    if last_visible_section_image is not None:
        return True, ""

    # どちらの画像もない場合はエラー
    error_html = f"""
    <div style="padding: 15px; border-radius: 10px; background-color: #ffebee; border: 1px solid #f44336; margin: 10px 0;">
        <h3 style="color: #d32f2f; margin: 0 0 10px 0;">{translate('❗️ 画像が選択されていません')}</h3>
        <p>{translate('生成を開始する前に「Image」欄または表示されている最後のキーフレーム画像に画像をアップロードしてください。これはあまねく叡智の始発点となる重要な画像です。')}</p>
    </div>
    """
    error_bar = make_progress_bar_html(100, translate('画像がありません'))
    return False, error_html + error_bar

def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf=16, all_padding_value=1.0, end_frame=None, end_frame_strength=1.0, frame_size_setting="1秒 (33フレーム)", keep_section_videos=False, lora_file=None, lora_scale=0.8, output_dir=None, save_section_frames=False, section_settings=None, use_all_padding=False, use_lora=False, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640):
    global stream

    # バリデーション関数で既にチェック済みなので、ここでの再チェックは不要

    # フレームサイズ設定に応じてlatent_window_sizeを先に調整
    if frame_size_setting == "0.5秒 (17フレーム)":
        # 0.5秒の場合はlatent_window_size=4.5に設定（実際には4.5*4-3=17フレーム≒0.5秒@30fps）
        latent_window_size = 4.5
        print(translate('フレームサイズを0.5秒モードに設定: latent_window_size = {0}').format(latent_window_size))
    else:
        # デフォルトの1秒モードではlatent_window_size=9を使用（9*4-3=33フレーム≒1秒@30fps）
        latent_window_size = 9
        print(translate('フレームサイズを1秒モードに設定: latent_window_size = {0}').format(latent_window_size))

    # 下のように修正：解像度を安全な値に丸めてログ表示
    from diffusers_helper.bucket_tools import SAFE_RESOLUTIONS
    if resolution not in SAFE_RESOLUTIONS:
        closest_resolution = min(SAFE_RESOLUTIONS, key=lambda x: abs(x - resolution))
        print(translate('安全な解像度値ではないため、{0}から{1}に自動調整しました').format(resolution, closest_resolution))
        resolution = closest_resolution
    
    # 解像度設定を出力
    print(translate('解像度を設定: {0}').format(resolution))

    # 動画生成の設定情報をログに出力
    # 4.5の場合は5として計算するための特別処理
    if latent_window_size == 4.5:
        frame_count = 17  # 5 * 4 - 3 = 17
    else:
        frame_count = int(latent_window_size * 4 - 3)
    total_latent_sections = int(max(round((total_second_length * 30) / frame_count), 1))

    mode_name = translate("通常モード") if mode_radio.value == MODE_TYPE_NORMAL else translate("ループモード")

    print(translate("\n==== 動画生成開始 ====="))
    print(translate("\u25c6 生成モード: {0}").format(mode_name))
    print(translate("\u25c6 動画長: {0}秒").format(total_second_length))
    print(translate("\u25c6 フレームサイズ: {0}").format(frame_size_setting))
    print(translate("\u25c6 生成セクション数: {0}回").format(total_latent_sections))
    print(translate("\u25c6 サンプリングステップ数: {0}").format(steps))
    print(translate("\u25c6 TeaCache使用: {0}").format(use_teacache))
    print(translate("\u25c6 LoRA使用: {0}").format(use_lora))

    # FP8最適化設定のログ出力
    print(translate("\u25c6 FP8最適化: {0}").format(fp8_optimization))

    # オールパディング設定のログ出力
    if use_all_padding:
        print(translate("\u25c6 オールパディング: 有効 (値: {0})").format(round(all_padding_value, 1)))
    else:
        print(translate("\u25c6 オールパディング: 無効"))

    # LoRA情報のログ出力
    if use_lora and lora_file is not None:
        print(translate("\u25c6 LoRAファイル: {0}").format(os.path.basename(lora_file.name)))
        print(translate("\u25c6 LoRA適用強度: {0}").format(lora_scale))

    # セクションごとのキーフレーム画像の使用状況をログに出力
    valid_sections = []
    if section_settings is not None:
        for i, sec_data in enumerate(section_settings):
            if sec_data and sec_data[1] is not None:  # 画像が設定されている場合
                valid_sections.append(sec_data[0])

    if valid_sections:
        print(translate("\u25c6 使用するキーフレーム画像: セクション{0}").format(', '.join(map(str, valid_sections))))
    else:
        print(translate("◆ キーフレーム画像: デフォルト設定のみ使用"))

    print("=============================\n")

    if use_random_seed:
        seed = random.randint(0, 2**32 - 1)
        # UIのseed欄もランダム値で更新
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(value=seed)
    else:
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update()

    stream = AsyncStream()

    # GPUメモリの設定値をデバッグ出力し、正しい型に変換
    gpu_memory_value = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
    print(translate('Using GPU memory preservation setting: {0} GB').format(gpu_memory_value))

    # 出力フォルダが空の場合はデフォルト値を使用
    if not output_dir or not output_dir.strip():
        output_dir = "outputs"
    print(translate('Output directory: {0}').format(output_dir))

    # 先に入力データの状態をログ出力（デバッグ用）
    if input_image is not None:
        print(translate("[DEBUG] input_image shape: {0}, type: {1}").format(input_image.shape, type(input_image)))
    if end_frame is not None:
        print(translate("[DEBUG] end_frame shape: {0}, type: {1}").format(end_frame.shape, type(end_frame)))
    if section_settings is not None:
        print(translate("[DEBUG] section_settings count: {0}").format(len(section_settings)))
        valid_images = sum(1 for s in section_settings if s and s[1] is not None)
        print(translate("[DEBUG] Valid section images: {0}").format(valid_images))

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_value, use_teacache, mp4_crf, all_padding_value, end_frame, end_frame_strength, keep_section_videos, lora_file, lora_scale, output_dir, save_section_frames, section_settings, use_all_padding, use_lora, save_tensor_data, tensor_data_input, fp8_optimization, resolution)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            # より明確な更新方法を使用し、preview_imageを明示的にクリア
            yield output_filename, gr.update(value=None, visible=False), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update()

        if flag == 'progress':
            preview, desc, html = data
            # preview_imageを明示的に設定
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update()

        if flag == 'end':
            # 処理終了時に明示的にpreview_imageをクリア
            # 出力ファイル名と処理完了メッセージを返す
            yield output_filename, gr.update(value=None, visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False), gr.update()
            break


def end_process():
    stream.input_queue.push('end')





# 既存のQuick Prompts（初期化時にプリセットに変換されるので、互換性のために残す）
quick_prompts = [
    'A character doing some simple body movements.',
    'A character uses expressive hand gestures and body language.',
    'A character walks leisurely with relaxed movements.',
    'A character performs dynamic movements with energy and flowing motion.',
    'A character moves in unexpected ways, with surprising transitions poses.',
]
quick_prompts = [[x] for x in quick_prompts]


css = get_app_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.HTML('<h1>FramePack<span class="title-suffix">-eichi</span></h1>')

    # デバッグ情報の表示
    # print_keyframe_debug_info()

    # 一番上の行に「生成モード、セクションフレームサイズ、オールパディング、動画長」を配置
    with gr.Row():
        with gr.Column(scale=1):
            mode_radio = gr.Radio(choices=[MODE_TYPE_NORMAL, MODE_TYPE_LOOP], value=MODE_TYPE_NORMAL, label=translate("生成モード"), info=translate("通常：一般的な生成 / ループ：ループ動画用"))
        with gr.Column(scale=1):
            # フレームサイズ切替用のUIコントロール（名前を「セクションフレームサイズ」に変更）
            frame_size_radio = gr.Radio(
                choices=[translate("1秒 (33フレーム)"), translate("0.5秒 (17フレーム)")],
                value=translate("1秒 (33フレーム)"),
                label=translate("セクションフレームサイズ"),
                info=translate("1秒 = 高品質・通常速度 / 0.5秒 = よりなめらかな動き（実験的機能）")
            )
        with gr.Column(scale=1):
            # オールパディング設定
            use_all_padding = gr.Checkbox(label=translate("オールパディング"), value=False, info=translate("数値が小さいほど直前の絵への影響度が下がり動きが増える"), elem_id="all_padding_checkbox")
            all_padding_value = gr.Slider(label=translate("パディング値"), minimum=0.2, maximum=3, value=1, step=0.1, info=translate("すべてのセクションに適用するパディング値（0.2〜3の整数）"), visible=False)

            # オールパディングのチェックボックス状態に応じてスライダーの表示/非表示を切り替える
            def toggle_all_padding_visibility(use_all_padding):
                return gr.update(visible=use_all_padding)

            use_all_padding.change(
                fn=toggle_all_padding_visibility,
                inputs=[use_all_padding],
                outputs=[all_padding_value]
            )
        with gr.Column(scale=1):
            # 設定から動的に選択肢を生成
            length_radio = gr.Radio(choices=get_video_modes(), value=translate("1秒"), label=translate("動画長"), info=translate("キーフレーム画像のコピー範囲と動画の長さを設定"))

    with gr.Row():
        with gr.Column():
            # Final Frameの上に説明を追加
            gr.Markdown(translate("**Finalは最後の画像、Imageは最初の画像(最終キーフレーム画像といずれか必須)となります。**"))
            end_frame = gr.Image(sources='upload', type="numpy", label=translate("Final Frame (Optional)"), height=320)

            # テンソルデータ設定をグループ化して灰色のタイトルバーに変更
            with gr.Group():
                gr.Markdown(translate("### テンソルデータ設定"))

                # テンソルデータ使用有無のチェックボックス
                use_tensor_data = gr.Checkbox(label=translate("テンソルデータを使用する"), value=False, info=translate("チェックをオンにするとテンソルデータをアップロードできます"))

                # テンソルデータ設定コンポーネント（初期状態では非表示）
                with gr.Group(visible=False) as tensor_data_group:
                    tensor_data_input = gr.File(
                        label=translate("テンソルデータアップロード (.safetensors) - 生成動画の後方(末尾)に結合されます"),
                        file_types=[".safetensors"]
                    )

                    gr.Markdown(translate("※ テンソルデータをアップロードすると通常の動画生成後に、その動画の後方（末尾）に結合されます。\n結合した動画は「元のファイル名_combined.mp4」として保存されます。\n※ テンソルデータの保存機能を有効にすると、生成とアップロードのテンソルを結合したデータも保存されます。\n※ テンソルデータの結合は別ツール `python eichi_utils/tensor_combiner.py --ui` でもできます。"))

                # チェックボックスの状態によってテンソルデータ設定の表示/非表示を切り替える関数
                def toggle_tensor_data_settings(use_tensor):
                    return gr.update(visible=use_tensor)

                # チェックボックスの変更イベントに関数を紐づけ
                use_tensor_data.change(
                    fn=toggle_tensor_data_settings,
                    inputs=[use_tensor_data],
                    outputs=[tensor_data_group]
                )
                
            # テンソルデータ設定の下に解像度スライダーを追加
            with gr.Group():
                resolution = gr.Slider(
                    label=translate("解像度"), 
                    minimum=512, 
                    maximum=768, 
                    value=640, 
                    step=128, 
                    info=translate("出力動画の基準解像度。現在は512か640か768のいずれかのみ対応（640推奨）")
                )

            # 開始・終了ボタン
            with gr.Row():
                start_button = gr.Button(value=translate("Start Generation"))
                end_button = gr.Button(value=translate("End Generation"), interactive=False)

            # セクション入力用のリストを初期化
            section_number_inputs = []
            section_image_inputs = []
            section_prompt_inputs = []  # プロンプト入力欄用のリスト
            section_row_groups = []  # 各セクションのUI行を管理するリスト

            # 設定から最大キーフレーム数を取得
            max_keyframes = get_max_keyframes_count()

            # 現在の動画モードで必要なセクション数を取得する関数
            def get_current_sections_count():
                mode_value = length_radio.value
                if mode_value in VIDEO_MODE_SETTINGS:
                    # sections値をそのまま使用 - 注：これは0から始めた場合の最大値となる
                    return VIDEO_MODE_SETTINGS[mode_value]["sections"]
                return max_keyframes  # デフォルト値

            # 現在の必要セクション数を取得
            initial_sections_count = get_current_sections_count()
            # セクション設定タイトルの定義と動的な更新用の関数
            # 現在のセクション数に応じたMarkdownを返す関数
            def generate_section_title(total_sections):
                last_section = total_sections - 1
                return translate('### セクション設定（逆順表示）\n\nセクションは逆時系列で表示されています。Image(始点)は必須でFinal(終点)から遡って画像を設定してください。**最終キーフレームの画像は、Image(始点)より優先されます。総数{0}**').format(total_sections)

            # 動画のモードとフレームサイズに基づいてセクション数を計算し、タイトルを更新する関数
            def update_section_title(frame_size, mode, length):
                seconds = get_video_seconds(length)
                latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                frame_count = latent_window_size * 4 - 3
                total_frames = int(seconds * 30)
                total_sections = int(max(round(total_frames / frame_count), 1))
                # 表示セクション数の設定
                # 例: 総セクション数が5の場合、4～0の5個のセクションが表示される
                display_sections = total_sections
                return generate_section_title(display_sections)

            # 初期タイトルを計算
            initial_title = update_section_title(translate("1秒 (33フレーム)"), MODE_TYPE_NORMAL, translate("1秒"))

            with gr.Accordion(translate("セクション設定"), open=False, elem_classes="section-accordion"):
                with gr.Group(elem_classes="section-container"):
                    section_title = gr.Markdown(initial_title)

                    # セクション番号0の上にコピー機能チェックボックスを追加（ループモード時のみ表示）
                    with gr.Row(visible=(mode_radio.value == MODE_TYPE_LOOP)) as copy_button_row:
                        keyframe_copy_checkbox = gr.Checkbox(label="キーフレーム自動コピー機能を有効にする", value=True, info="オンにするとキーフレーム間の自動コピーが行われます")

                    for i in range(max_keyframes):
                        with gr.Row(visible=(i < initial_sections_count), elem_classes="section-row") as row_group:
                            # 左側にセクション番号とプロンプトを配置
                            with gr.Column(scale=1):
                                section_number = gr.Number(label=translate("セクション番号 {0}").format(i), value=i, precision=0)
                                section_prompt = gr.Textbox(label=translate("セクションプロンプト {0}").format(i), placeholder=translate("セクション固有のプロンプト（空白の場合は共通プロンプトを使用）"), lines=2)

                            # 右側にキーフレーム画像のみ配置
                            with gr.Column(scale=2):
                                section_image = gr.Image(label=translate("キーフレーム画像 {0}").format(i), sources="upload", type="numpy", height=200)
                            section_number_inputs.append(section_number)
                            section_image_inputs.append(section_image)
                            section_prompt_inputs.append(section_prompt)
                            section_row_groups.append(row_group)  # 行全体をリストに保存

                    # ※ enable_keyframe_copy変数は後で使用するため、ここで定義（モードに応じた初期値設定）
                    enable_keyframe_copy = gr.State(mode_radio.value == MODE_TYPE_LOOP) # ループモードの場合はTrue、通常モードの場合はFalse

                    # キーフレーム自動コピーチェックボックスの変更をenable_keyframe_copyに反映させる関数
                    def update_keyframe_copy_state(value):
                        return value

                    # チェックボックスの変更がenable_keyframe_copyに反映されるようにイベントを設定
                    keyframe_copy_checkbox.change(
                        fn=update_keyframe_copy_state,
                        inputs=[keyframe_copy_checkbox],
                        outputs=[enable_keyframe_copy]
                    )

                    # チェックボックス変更時に赤枠/青枠の表示を切り替える
                    def update_frame_visibility_from_checkbox(value, mode):
                    #   print(f"チェックボックス変更: 値={value}, モード={mode}")
                        # モードとチェックボックスの両方に基づいて枠表示を決定
                        is_loop = (mode == MODE_TYPE_LOOP)

                        # 通常モードでは常に赤枠/青枠を非表示 (最優先で確認)
                        if not is_loop:
                        #   print(f"通常モード (チェックボックス値={value}): 赤枠/青枠を強制的に非表示にします")
                            # 通常モードでは常にelm_classesを空にして赤枠/青枠を非表示に確定する
                            return gr.update(elem_classes=""), gr.update(elem_classes="")

                        # ループモードでチェックボックスがオンの場合のみ枠を表示
                        if value:
                        #   print(f"ループモード + チェックボックスオン: 赤枠/青枠を表示します")
                            return gr.update(elem_classes="highlighted-keyframe-red"), gr.update(elem_classes="highlighted-keyframe-blue")
                        else:
                        #   print(f"ループモード + チェックボックスオフ: 赤枠/青枠を非表示にします")
                            # ループモードでもチェックがオフなら必ずelem_classesを空にして赤枠/青枠を非表示にする
                            return gr.update(elem_classes=""), gr.update(elem_classes="")

                    keyframe_copy_checkbox.change(
                        fn=update_frame_visibility_from_checkbox,
                        inputs=[keyframe_copy_checkbox, mode_radio],
                        outputs=[section_image_inputs[0], section_image_inputs[1]]
                    )

                    # モード切り替え時にチェックボックスの値と表示状態を制御する
                    def toggle_copy_checkbox_visibility(mode):
                        """モード切り替え時にチェックボックスの表示/非表示を切り替える"""
                        is_loop = (mode == MODE_TYPE_LOOP)
                        # 通常モードの場合はチェックボックスを非表示に設定、コピー機能を必ずFalseにする
                        if not is_loop:
                        #   print(f"モード切替: {mode} -> チェックボックス非表示、コピー機能を無効化")
                            return gr.update(visible=False, value=False), gr.update(visible=False), False
                        # ループモードの場合は表示し、デフォルトでオンにする
                    #   print(f"モード切替: {mode} -> チェックボックス表示かつオンに設定")
                        return gr.update(visible=True, value=True), gr.update(visible=True), True

                    # モード切り替え時にチェックボックスの表示/非表示と値を制御するイベントを設定
                    mode_radio.change(
                        fn=toggle_copy_checkbox_visibility,
                        inputs=[mode_radio],
                        outputs=[keyframe_copy_checkbox, copy_button_row, enable_keyframe_copy]
                    ) # ループモードに切替時は常にチェックボックスをオンにし、通常モード時は常にオフにする

                    # モード切り替え時に赤枠/青枠の表示を更新
                    def update_frame_visibility_from_mode(mode):
                        # モードに基づいて枠表示を決定
                        is_loop = (mode == MODE_TYPE_LOOP)

                        # 通常モードでは無条件で赤枠/青枠を非表示 (最優先で確定)
                        if not is_loop:
                        #   print(f"モード切替: 通常モード -> 枠を強制的に非表示")
                            return gr.update(elem_classes=""), gr.update(elem_classes="")
                        else:
                            # ループモードではチェックボックスが常にオンになるので枠を表示
                        #   print(f"モード切替: ループモード -> チェックボックスオンなので枠を表示")
                            return gr.update(elem_classes="highlighted-keyframe-red"), gr.update(elem_classes="highlighted-keyframe-blue")

                    mode_radio.change(
                        fn=update_frame_visibility_from_mode,
                        inputs=[mode_radio],
                        outputs=[section_image_inputs[0], section_image_inputs[1]]
                    )

            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)

            # LoRA設定グループを追加
            with gr.Group(visible=has_lora_support) as lora_settings_group:
                gr.Markdown(translate("### LoRA設定"))

                # LoRA使用有無のチェックボックス
                use_lora = gr.Checkbox(label=translate("LoRAを使用する"), value=False, info=translate("チェックをオンにするとLoRAを使用します（要16GB VRAM以上）"))

                # LoRA設定コンポーネント（初期状態では非表示）
                lora_file = gr.File(label=translate("LoRAファイル (.safetensors, .pt, .bin)"),
                            file_types=[".safetensors", ".pt", ".bin"],
                            visible=False)
                lora_scale = gr.Slider(label=translate("LoRA適用強度"), minimum=0.0, maximum=1.0,
                           value=0.8, step=0.01, visible=False)
                fp8_optimization = gr.Checkbox(
                    label=translate("FP8最適化"),
                    value=False,
                    info=translate("メモリ使用量を削減し、速度を改善します（PyTorch 2.1以上が必要）"),
                    visible=False
                )
                lora_blocks_type = gr.Dropdown(
                    label=translate("LoRAブロック選択"),
                    choices=["all", "single_blocks", "double_blocks", "db0-9", "db10-19", "sb0-9", "sb10-19", "important"],
                    value="all",
                    info=translate("選択するブロックタイプ（all=すべて、その他=メモリ節約）"),
                    visible=False
                )

                # チェックボックスの状態によって他のLoRA設定の表示/非表示を切り替える関数
                def toggle_lora_settings(use_lora):
                    return [
                        gr.update(visible=use_lora),  # lora_file
                        gr.update(visible=use_lora),  # lora_scale
                        gr.update(visible=use_lora),  # fp8_optimization
                    ]

                # チェックボックスの変更イベントに関数を紋づけ
                use_lora.change(fn=toggle_lora_settings,
                           inputs=[use_lora],
                           outputs=[lora_file, lora_scale, fp8_optimization])

                # LoRAサポートが無効の場合のメッセージ
                if not has_lora_support:
                    gr.Markdown(translate("LoRAサポートは現在無効です。lora_utilsモジュールが必要です。"))

            prompt = gr.Textbox(label=translate("Prompt"), value=get_default_startup_prompt(), lines=6)

            # プロンプト管理パネルの追加
            with gr.Group(visible=True) as prompt_management:
                gr.Markdown(translate("### プロンプト管理"))

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

            # プリセットの説明文を削除

            # 互換性のためにQuick Listも残しておくが、非表示にする
            with gr.Row(visible=False):
                example_quick_prompts = gr.Dataset(samples=quick_prompts, label=translate("Quick List"), samples_per_page=1000, components=[prompt])
                example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            # 以下の設定ブロックは右カラムに移動しました

                # セクション設定のリストは既にアコーディオン内で初期化されています
                # section_number_inputs
                # section_image_inputs
                # section_prompt_inputs
                # section_row_groups

                # section_settingsは入力欄の値をまとめてリスト化
                def collect_section_settings(*args):
                    # args: [num1, img1, prompt1, num2, img2, prompt2, ...]
                    return [[args[i], args[i+1], args[i+2]] for i in range(0, len(args), 3)]

                section_settings = gr.State([[None, None, ""] for _ in range(max_keyframes)])
                section_inputs = []
                for i in range(max_keyframes):
                    section_inputs.extend([section_number_inputs[i], section_image_inputs[i], section_prompt_inputs[i]])

                # section_inputsをまとめてsection_settings Stateに格納
                def update_section_settings(*args):
                    return collect_section_settings(*args)

                # section_inputsが変化したらsection_settings Stateを更新
                for inp in section_inputs:
                    inp.change(fn=update_section_settings, inputs=section_inputs, outputs=section_settings)

                # フレームサイズ変更時の処理を追加
                def update_section_calculation(frame_size, mode, length):
                    """フレームサイズ変更時にセクション数を再計算して表示を更新"""
                    # 動画長を取得
                    seconds = get_video_seconds(length)

                    # latent_window_sizeを設定
                    latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                    frame_count = latent_window_size * 4 - 3

                    # セクション数を計算
                    total_frames = int(seconds * 30)
                    total_sections = int(max(round(total_frames / frame_count), 1))

                    # 計算詳細を表示するHTMLを生成
                    html = f"""<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 14px;'>
                    {translate('<strong>計算詳細</strong>: モード={0}, フレームサイズ={1}, 総フレーム数={2}, セクションあたり={3}フレーム, 必要セクション数={4}').format(length, frame_size, total_frames, frame_count, total_sections)}
                    <br>
                    {translate('動画モード {0} とフレームサイズ {1} で必要なセクション数: <strong>{2}</strong>').format(length, frame_size, total_sections)}
                    </div>"""

                    # デバッグ用ログ
                    print(translate("計算結果: モード={0}, フレームサイズ={1}, latent_window_size={2}, 総フレーム数={3}, 必要セクション数={4}").format(length, frame_size, latent_window_size, total_frames, total_sections))

                    return html

                # 初期化時にも計算を実行
                initial_html = update_section_calculation(frame_size_radio.value, mode_radio.value, length_radio.value)
                section_calc_display = gr.HTML(value=initial_html, label="")

                # フレームサイズ変更イベント - HTML表示の更新とセクションタイトルの更新を行う
                frame_size_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # フレームサイズ変更時にセクションタイトルも更新
                frame_size_radio.change(
                    fn=update_section_title,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_title]
                )

                # セクションの表示/非表示のみを制御する関数
                def update_section_visibility(mode, length, frame_size=None):
                    """画像は初期化せずにセクションの表示/非表示のみを制御する関数"""
                    # フレームサイズに基づくセクション数計算
                    seconds = get_video_seconds(length)
                    latent_window_size_value = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                    frame_count = latent_window_size_value * 4 - 3
                    total_frames = int(seconds * 30)
                    total_sections = int(max(round(total_frames / frame_count), 1))

                    # 通常モードの場合は全てのセクションの赤枠青枠を強制的にクリア
                    is_normal_mode = (mode == MODE_TYPE_NORMAL)
                    section_image_updates = []

                    print(translate("セクション視認性更新: モード={mode}, 長さ={length}, 必要セクション数={total_sections}").format(mode=mode, length=length, total_sections=total_sections))

                    for i in range(len(section_image_inputs)):
                        if is_normal_mode:
                            # 通常モードではすべてのセクション画像の赤枠青枠を強制的にクリア
                            # 重要: 通常モードでは無条件で済む結果を返す
                            # print(translate("  セクション{i}: 通常モードなので赤枠/青枠を強制的にクリア").format(i=i))
                            section_image_updates.append(gr.update(elem_classes=""))  # 必ずelem_classesを空に設定
                        else:
                            # ループモードではセクション0と1に赤枠青枠を設定
                            # ループモードではチェックボックスが常にオンになることを利用
                            if i == 0:
                                # print(translate("  セクション{i}: ループモードのセクション0に赤枠を設定").format(i=i))
                                section_image_updates.append(gr.update(elem_classes="highlighted-keyframe-red"))
                            elif i == 1:
                                # print(translate("  セクション{i}: ループモードのセクション1に青枠を設定").format(i=i))
                                section_image_updates.append(gr.update(elem_classes="highlighted-keyframe-blue"))
                            else:
                                # print(translate("  セクション{i}: ループモードの他セクションは空枠に設定").format(i=i))
                                section_image_updates.append(gr.update(elem_classes=""))

                    # 各セクションの表示/非表示のみを更新
                    section_row_updates = []
                    for i in range(len(section_row_groups)):
                        section_row_updates.append(gr.update(visible=(i < total_sections)))

                    # 返値の設定 - input_imageとend_frameは更新せず
                    return [gr.update()] * 2 + section_image_updates + [gr.update(value=seconds)] + section_row_updates

                # 注意: この関数のイベント登録は、total_second_lengthのUIコンポーネント定義後に行うため、
                # ここでは関数の定義のみ行い、実際のイベント登録はUIコンポーネント定義後に行います。

                # 動画長変更イベントでもセクション数計算を更新
                length_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # 動画長変更時にセクションタイトルも更新
                length_radio.change(
                    fn=update_section_title,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_title]
                )

                # モード変更時にも計算を更新
                mode_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # モード変更時にセクションタイトルも更新
                mode_radio.change(
                    fn=update_section_title,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_title]
                )

                # モード変更時の処理もtotal_second_lengthコンポーネント定義後に行います

                # 動画長変更時のセクション表示更新もtotal_second_lengthコンポーネント定義後に行います

                # 入力画像変更時の処理 - ループモード用に復活
                # 通常モードでセクションにコピーする処理はコメント化したまま
                # ループモードのLastにコピーする処理のみ復活

                # 終端フレームハンドラ関数（FinalからImageへのコピーのみ実装）
                def loop_mode_final_handler(img, mode, length):
                    """end_frameの変更時、ループモードの場合のみコピーを行う関数"""
                    if img is None:
                        # 画像が指定されていない場合は何もしない
                        return gr.update()

                    # ループモードかどうかで処理を分岐
                    if mode == MODE_TYPE_LOOP:
                        # ループモード: ImageにFinalFrameをコピー
                        return gr.update(value=img)  # input_imageにコピー
                    else:
                        # 通常モード: 何もしない
                        return gr.update()

                # 終端フレームの変更ハンドラを登録
                end_frame.change(
                    fn=loop_mode_final_handler,
                    inputs=[end_frame, mode_radio, length_radio],
                    outputs=[input_image]
                )

                # 各キーフレーム画像の変更イベントを個別に設定
                # 一度に複数のコンポーネントを更新する代わりに、個別の更新関数を使用
                def create_single_keyframe_handler(src_idx, target_idx):
                    def handle_single_keyframe(img, mode, length, enable_copy):
                        # ループモード以外では絶対にコピーを行わない
                        if mode != MODE_TYPE_LOOP:
                            # 通常モードでは絶対にコピーしない
                        #   print(translate("通常モードでのコピー要求を拒否: src={src_idx}, target={target_idx}").format(src_idx=src_idx, target_idx=target_idx))
                            return gr.update()

                        # コピー条件をチェック
                        if img is None or not enable_copy:
                            return gr.update()

                        # 現在のセクション数を動的に計算
                        seconds = get_video_seconds(length)
                        # フレームサイズに応じたlatent_window_sizeの調整（ここではUIの設定によらず計算）
                        frame_size = frame_size_radio.value
                        latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                        frame_count = latent_window_size * 4 - 3
                        total_frames = int(seconds * 30)
                        total_sections = int(max(round(total_frames / frame_count), 1))

                        # 対象セクションが有効範囲を超えている場合はコピーしない(項目数的に+1)
                        if target_idx >= total_sections:
                        #   print(translate("コピー対象セクション{target_idx}が有効範囲({total_sections}まで)を超えています").format(target_idx=target_idx, total_sections=total_sections))
                            return gr.update()

                        # コピー先のチェック - セクション0は偶数番号に、セクション1は奇数番号にコピー
                        if src_idx == 0 and target_idx % 2 == 0 and target_idx != 0:
                            # 詳細ログ出力
                        #   print(translate("赤枠(0)から偶数セクション{target_idx}へのコピー実行 (動的セクション数:{total_sections})").format(target_idx=target_idx, total_sections=total_sections))
                            return gr.update(value=img)
                        elif src_idx == 1 and target_idx % 2 == 1 and target_idx != 1:
                            # 詳細ログ出力
                        #   print(translate("青枠(1)から奇数セクション{target_idx}へのコピー実行 (動的セクション数:{total_sections})").format(target_idx=target_idx, total_sections=total_sections))
                            return gr.update(value=img)

                        # 条件に合わない場合
                        return gr.update()
                    return handle_single_keyframe

                # 各キーフレームについて、影響を受ける可能性のある後続のキーフレームごとに個別のイベントを設定
                # ここではイベント登録の定義のみ行い、実際の登録はUIコンポーネント定義後に行う

                # キーフレーム自動コピーの初期値はStateでデフォルトでTrueに設定済み
                # enable_keyframe_copyは既にTrueに初期化されているのでここでは特に何もしない

                # モード切り替え時に赤枠/青枠の表示を切り替える関数
                # トグル関数は不要になったため削除
                # 代わりにcheckbox値のみに依存するシンプルな条件分岐を各関数で直接実装

        with gr.Column():
            result_video = gr.Video(label=translate("Finished Frames"), autoplay=True, show_share_button=False, height=512, loop=True)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)

            # フレームサイズ切替用のUIコントロールは上部に移動したため削除

            # 計算結果を表示するエリア
            section_calc_display = gr.HTML("", label="")

            use_teacache = gr.Checkbox(label=translate('Use TeaCache'), value=True, info=translate('Faster speed, but often makes hands and fingers slightly worse.'))

            # Use Random Seedの初期値
            use_random_seed_default = True
            seed_default = random.randint(0, 2**32 - 1) if use_random_seed_default else 1

            use_random_seed = gr.Checkbox(label=translate("Use Random Seed"), value=use_random_seed_default)

            n_prompt = gr.Textbox(label=translate("Negative Prompt"), value="", visible=False)  # Not used
            seed = gr.Number(label=translate("Seed"), value=seed_default, precision=0)

            def set_random_seed(is_checked):
                if is_checked:
                    return random.randint(0, 2**32 - 1)
                else:
                    return gr.update()
            use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)

            total_second_length = gr.Slider(label=translate("Total Video Length (Seconds)"), minimum=1, maximum=120, value=1, step=1)
            latent_window_size = gr.Slider(label=translate("Latent Window Size"), minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
            steps = gr.Slider(label=translate("Steps"), minimum=1, maximum=100, value=25, step=1, info=translate('Changing this value is not recommended.'))

            cfg = gr.Slider(label=translate("CFG Scale"), minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
            gs = gr.Slider(label=translate("Distilled CFG Scale"), minimum=1.0, maximum=32.0, value=10.0, step=0.01, info=translate('Changing this value is not recommended.'))
            rs = gr.Slider(label=translate("CFG Re-Scale"), minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

            gpu_memory_preservation = gr.Slider(label=translate("GPU Memory to Preserve (GB) (smaller = more VRAM usage)"), minimum=6, maximum=128, value=10, step=0.1, info=translate("空けておくGPUメモリ量を指定。小さい値=より多くのVRAMを使用可能=高速、大きい値=より少ないVRAMを使用=安全"))

            # MP4圧縮設定スライダーを追加
            mp4_crf = gr.Slider(label=translate("MP4 Compression"), minimum=0, maximum=100, value=16, step=1, info=translate("数値が小さいほど高品質になります。0は無圧縮。黒画面が出る場合は16に設定してください。"))

            # セクションごとの動画保存チェックボックスを追加（デフォルトOFF）
            keep_section_videos = gr.Checkbox(label=translate("完了時にセクションごとの動画を残す - チェックがない場合は最終動画のみ保存されます（デフォルトOFF）"), value=False)

            # テンソルデータ保存チェックボックス违加
            save_tensor_data = gr.Checkbox(
                label=translate("完了時にテンソルデータ(.safetensors)も保存 - このデータを別の動画の後に結合可能"),
                value=False,
                info=translate("チェックすると、生成されたテンソルデータを保存します。アップロードされたテンソルがあれば、結合したテンソルデータも保存されます。")
            )

            # セクションごとの静止画保存チェックボックスを追加（デフォルトOFF）
            save_section_frames = gr.Checkbox(label=translate("Save Section Frames"), value=False, info=translate("各セクションの最終フレームを静止画として保存します（デフォルトOFF）"))

            # UIコンポーネント定義後のイベント登録
            # mode_radio.changeの登録 - セクションの表示/非表示と赤枠青枠の表示を同時に更新
            mode_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups
            )

            # frame_size_radio.changeの登録 - セクションの表示/非表示のみを更新
            frame_size_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups
            )

            # length_radio.changeの登録 - セクションの表示/非表示のみを更新
            length_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups
            )

            # mode_radio.changeの登録 - 拡張モード変更ハンドラを使用
            mode_radio.change(
                fn=lambda mode, length: extended_mode_length_change_handler(mode, length, section_number_inputs, section_row_groups),
                inputs=[mode_radio, length_radio],
                outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups
            )


            # EndFrame影響度調整スライダー
            with gr.Group():
                gr.Markdown(translate("### EndFrame影響度調整"))
                end_frame_strength = gr.Slider(
                    label=translate("EndFrame影響度"),
                    minimum=0.01,
                    maximum=1.00,
                    value=1.00,
                    step=0.01,
                    info=translate("最終フレームが動画全体に与える影響の強さを調整します。値を小さくすると最終フレームの影響が弱まり、最初のフレームに早く移行します。1.00が通常の動作です。")
                )

            # 出力フォルダ設定
            gr.Markdown(translate("※ 出力先は `webui` 配下に限定されます"))
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    # フォルダ名だけを入力欄に設定
                    output_dir = gr.Textbox(
                        label=translate("出力フォルダ名"),
                        value=output_folder_name,  # 設定から読み込んだ値を使用
                        info=translate("動画やキーフレーム画像の保存先フォルダ名"),
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

            # フォルダを開くボタンのイベント
            def handle_open_folder_btn(folder_name):
                """フォルダ名を保存し、そのフォルダを開く"""
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

            open_folder_btn.click(fn=handle_open_folder_btn, inputs=[output_dir], outputs=[output_dir, path_display])

            # プロンプト管理パネル（右カラムから左カラムに移動済み）

    # 実行前のバリデーション関数
    def validate_and_process(*args):
        """入力画像または最後のキーフレーム画像のいずれかが有効かどうかを確認し、問題がなければ処理を実行する"""
        input_img = args[0]  # 入力の最初が入力画像
        section_settings = args[24]  # section_settings引数のインデックス

        # 現在の動画長設定とフレームサイズ設定を渡す
        is_valid, error_message = validate_images(input_img, section_settings, length_radio, frame_size_radio)

        if not is_valid:
            # 画像が無い場合はエラーメッセージを表示して終了
            yield None, gr.update(visible=False), translate("エラー: 画像が選択されていません"), error_message, gr.update(interactive=True), gr.update(interactive=False), gr.update()
            return

        # 画像がある場合は通常の処理を実行
        # process関数のジェネレータを返す
        yield from process(*args)

    # 実行ボタンのイベント
    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf, all_padding_value, end_frame, end_frame_strength, frame_size_radio, keep_section_videos, lora_file, lora_scale, output_dir, save_section_frames, section_settings, use_all_padding, use_lora, save_tensor_data, tensor_data_input, fp8_optimization, resolution]
    start_button.click(fn=validate_and_process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed])
    end_button.click(fn=end_process)

    # キーフレーム画像変更時のイベント登録
    # セクション0（赤枚)からの自動コピー処理
    for target_idx in range(1, max_keyframes):
        # 偶数セクションにのみコピー
        if target_idx % 2 == 0:  # 偶数先セクション
            single_handler = create_single_keyframe_handler(0, target_idx)
            section_image_inputs[0].change(
                fn=single_handler,
                inputs=[section_image_inputs[0], mode_radio, length_radio, enable_keyframe_copy],
                outputs=[section_image_inputs[target_idx]]
            )

    # セクション1（青枠)からの自動コピー処理
    for target_idx in range(2, max_keyframes):
        # 奇数セクションにのみコピー
        if target_idx % 2 == 1:  # 奇数先セクション
            single_handler = create_single_keyframe_handler(1, target_idx)
            section_image_inputs[1].change(
                fn=single_handler,
                inputs=[section_image_inputs[1], mode_radio, length_radio, enable_keyframe_copy],
                outputs=[section_image_inputs[target_idx]]
            )

    # 注: create_single_keyframe_handler関数はフレームサイズや動画長に基づいた動的セクション数を計算します
    # UIでフレームサイズや動画長を変更すると、動的に計算されたセクション数に従ってコピー処理が行われます

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

    # 保存ボタンのクリックイベントを接続
    save_btn.click(
        fn=save_button_click_handler,
        inputs=[edit_name, edit_prompt],
        outputs=[result_message, preset_dropdown, prompt]
    )

    # クリアボタン処理
    def clear_fields():
        return gr.update(value=""), gr.update(value="")

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[edit_name, edit_prompt]
    )

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

    preset_dropdown.change(
        fn=load_preset_handler_wrapper,
        inputs=[preset_dropdown],
        outputs=[edit_name, edit_prompt]
    )

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

    apply_preset_btn.click(
        fn=apply_to_prompt,
        inputs=[edit_prompt],
        outputs=[prompt]
    )

    delete_preset_btn.click(
        fn=delete_preset_handler,
        inputs=[preset_dropdown],
        outputs=[result_message, preset_dropdown]
    )

# enable_keyframe_copyの初期化（グローバル変数）
enable_keyframe_copy = True

allowed_paths = [os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './outputs')))]

# 起動コード
try:
    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        allowed_paths=allowed_paths,
        inbrowser=args.inbrowser,
    )
except OSError as e:
    if "Cannot find empty port" in str(e):
        print("\n======================================================")
        print(translate("エラー: FramePack-eichiは既に起動しています。"))
        print(translate("同時に複数のインスタンスを実行することはできません。"))
        print(translate("現在実行中のアプリケーションを先に終了してください。"))
        print("======================================================\n")
        input(translate("続行するには何かキーを押してください..."))
    else:
        # その他のOSErrorの場合は元のエラーを表示
        print(translate("\nエラーが発生しました: {e}").format(e=e))
        input(translate("続行するには何かキーを押してください..."))
