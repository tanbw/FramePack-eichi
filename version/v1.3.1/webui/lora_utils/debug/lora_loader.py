"""FramePack-eichi LoRA loader module
Provides LoRA application functionality for HunyuanVideo model
Added memory usage optimization features and selective block loading
"""

import os
import torch
import logging
import traceback
import safetensors.torch as sf
import json
import hashlib
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
from time import time

# 環境変数による設定読み込み
import os
ENV_SELECTIVE_APP = os.environ.get('LORA_SELECTIVE_APP', '1')
ENV_DEBUG_LEVEL = os.environ.get('LORA_DEBUG_LEVEL', 'INFO')
ENV_FORCE_TRANSFORMER_LORA = os.environ.get('FORCE_TRANSFORMER_LORA', '0')
ENV_LOW_VRAM_LORA = os.environ.get('LOW_VRAM_LORA', '0')

# デバッグ設定
SELECTIVE_APPLICATION_DEFAULT = ENV_SELECTIVE_APP.lower() in ('1', 'true', 'yes', 'on')
DEBUG_ENABLED = ENV_DEBUG_LEVEL.upper() in ('DEBUG', 'TRACE', 'ALL')
FORCE_TRANSFORMER_LORA = ENV_FORCE_TRANSFORMER_LORA.lower() in ('1', 'true', 'yes', 'on')
LOW_VRAM_LORA = ENV_LOW_VRAM_LORA.lower() in ('1', 'true', 'yes', 'on')

# ロギング設定
logger = logging.getLogger("lora_loader")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 環境変数に基づくログレベル設定
log_level = logging.DEBUG if DEBUG_ENABLED else logging.INFO
logger.setLevel(log_level)

# 設定情報のログ出力
logger.info(f"===== LoRA設定情報 =====")
logger.info(f"選択的適用デフォルト: {SELECTIVE_APPLICATION_DEFAULT}")
logger.info(f"デバッグモード: {DEBUG_ENABLED}")
logger.info(f"強制transformer_lora使用: {FORCE_TRANSFORMER_LORA}")
logger.info(f"低VRAM LoRAモード: {LOW_VRAM_LORA}")
logger.info(f"ログレベル: {logging.getLevelName(log_level)}")
logger.info(f"=========================")

# ブロック選択の定義
BSINGLE = "single_blocks"
BDOUBLE = "double_blocks"
PRESET_BLOCKS = { # Name = single / double, accepted layers.
    "all": (None, None),  # すべてのブロックを対象
    "single_blocks": (BSINGLE, None),  # すべてのsingle_blocks
    "double_blocks": (BDOUBLE, None),  # すべてのdouble_blocks
    "db0-9": (BDOUBLE, list(range(0, 10))),  # double_blocksの0-9
    "db10-19": (BDOUBLE, list(range(10, 20))),  # double_blocksの10-19
    "sb0-9": (BSINGLE, list(range(0, 10))),  # single_blocksの0-9
    "sb10-19": (BSINGLE, list(range(10, 20))),  # single_blocksの10-19
    "important": (None, None)  # 重要なレイヤーのみ（別途フィルタリング）
}

# LoRAファイルのキャッシュ
_lora_cache = {}

# テンソル状態トラッキング用のグローバル変数
_tensor_stats = {}

# テンソル統計の記録
def track_tensor_stats(name, tensor, operation="初期化"):
    """テンソルの統計情報を記録する"""
    if DEBUG_ENABLED and isinstance(tensor, torch.Tensor):
        try:
            stats = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
                "non_zero": int((tensor != 0).sum().item()),
                "total": int(tensor.numel()),
                "operation": operation,
                "memory_mb": float(tensor.element_size() * tensor.numel() / 1024**2)
            }
            _tensor_stats[name] = stats
            if DEBUG_ENABLED:
                logger.debug(f"テンソル統計 [{name}] {operation}: shape={stats['shape']}, 非ゼロ={stats['non_zero']}/{stats['total']} ({stats['non_zero']/stats['total']*100:.2f}%), min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, メモリ={stats['memory_mb']:.2f}MB")
        except Exception as e:
            logger.debug(f"テンソル統計収集エラー [{name}]: {e}")

# モデルの差分を検出するヘルパー関数
def detect_model_differences(model1, model2, prefix="", max_params=10):
    """2つのモデル間の差分を検出してログに出力"""
    if not DEBUG_ENABLED:
        return
    
    if model1 is None or model2 is None:
        logger.debug(f"{prefix}モデル比較: いずれかのモデルがNone")
        return
        
    # 名前付きパラメータのリストを取得
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    # 共通キーを見つける
    common_keys = set(params1.keys()).intersection(set(params2.keys()))
    diff_count = 0
    diff_details = []
    
    for key in common_keys:
        if diff_count >= max_params:
            break
            
        p1 = params1[key]
        p2 = params2[key]
        
        # 比較前に同じデバイスとデータ型に移動
        p2 = p2.to(device=p1.device, dtype=p1.dtype)
        
        if not torch.allclose(p1, p2, rtol=1e-3, atol=1e-3):
            diff = torch.abs(p1 - p2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            diff_count += 1
            diff_details.append((key, max_diff, mean_diff))
    
    # 結果を出力
    if diff_count > 0:
        logger.debug(f"{prefix}モデル比較: {diff_count}/{len(common_keys)} パラメータで差異を検出")
        for key, max_diff, mean_diff in diff_details:
            logger.debug(f"  - [{key}]: 最大差={max_diff:.6e}, 平均差={mean_diff:.6e}")
    else:
        logger.debug(f"{prefix}モデル比較: 差異なし")
    
    return diff_count > 0

# LoRAの適用効果を判定する関数
def verify_lora_application(model, original_model=None, lora_state_dict=None):
    """LoRAが適用されたかどうかを検証"""
    if not DEBUG_ENABLED:
        return True
        
    verification_result = True
    
    # 1. 元のモデルがあれば比較
    if original_model is not None:
        diff_detected = detect_model_differences(model, original_model, "LoRA適用検証: ")
        if not diff_detected:
            logger.warning("LoRA検証: モデルに変化が検出されませんでした！")
            verification_result = False
    
    # 2. LoRA辞書があれば、該当するパラメータが変化しているか確認
    if lora_state_dict is not None:
        # LoRAキーのモジュールパス（例: layer.0.weight.lora_up → layer.0.weight）を抽出
        lora_target_modules = set()
        for key in lora_state_dict.keys():
            if '.lora_down' in key or '.lora_up' in key:
                module_path = key.split('.lora_')[0]
                lora_target_modules.add(module_path)
        
        # これらのモジュールが実際に変更されているか確認（一部だけをチェック）
        module_sample = list(lora_target_modules)[:5]  # 最初の5つだけチェック
        if module_sample and original_model is not None:
            orig_params = dict(original_model.named_parameters())
            new_params = dict(model.named_parameters())
            
            changed_modules = 0
            for module_path in module_sample:
                if module_path in orig_params and module_path in new_params:
                    if not torch.allclose(orig_params[module_path], new_params[module_path]):
                        changed_modules += 1
            
            if changed_modules == 0 and len(module_sample) > 0:
                logger.warning(f"LoRA検証: LoRA対象モジュールに変化が検出されませんでした！ (サンプル: {len(module_sample)})")
                verification_result = False
            else:
                logger.debug(f"LoRA検証: {changed_modules}/{len(module_sample)} モジュールに変化を確認")
    
    return verification_result

# モデルの差分を検出するヘルパー関数
def detect_model_differences(model1, model2, prefix="", max_params=10):
    """2つのモデル間の差分を検出してログに出力"""
    if not DEBUG_ENABLED:
        return
    
    if model1 is None or model2 is None:
        logger.debug(f"{prefix}モデル比較: いずれかのモデルがNone")
        return
        
    # 名前付きパラメータのリストを取得
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    # 共通キーを見つける
    common_keys = set(params1.keys()).intersection(set(params2.keys()))
    diff_count = 0
    diff_details = []
    
    for key in common_keys:
        if diff_count >= max_params:
            break
            
        p1 = params1[key]
        p2 = params2[key]
        
        # 比較前に同じデバイスとデータ型に移動
        p2 = p2.to(device=p1.device, dtype=p1.dtype)
        
        if not torch.allclose(p1, p2, rtol=1e-3, atol=1e-3):
            diff = torch.abs(p1 - p2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            diff_count += 1
            diff_details.append((key, max_diff, mean_diff))
    
    # 結果を出力
    if diff_count > 0:
        logger.debug(f"{prefix}モデル比較: {diff_count}/{len(common_keys)} パラメータで差異を検出")
        for key, max_diff, mean_diff in diff_details:
            logger.debug(f"  - [{key}]: 最大差={max_diff:.6e}, 平均差={mean_diff:.6e}")
    else:
        logger.debug(f"{prefix}モデル比較: 差異なし")
    
    return diff_count > 0

# LoRAの適用効果を判定する関数
def verify_lora_application(model, original_model=None, lora_state_dict=None):
    """LoRAが適用されたかどうかを検証"""
    if not DEBUG_ENABLED:
        return True
        
    verification_result = True
    
    # 1. 元のモデルがあれば比較
    if original_model is not None:
        diff_detected = detect_model_differences(model, original_model, "LoRA適用検証: ")
        if not diff_detected:
            logger.warning("LoRA検証: モデルに変化が検出されませんでした！")
            verification_result = False
    
    # 2. LoRA辞書があれば、該当するパラメータが変化しているか確認
    if lora_state_dict is not None:
        # LoRAキーのモジュールパス（例: layer.0.weight.lora_up → layer.0.weight）を抽出
        lora_target_modules = set()
        for key in lora_state_dict.keys():
            if '.lora_down' in key or '.lora_up' in key:
                module_path = key.split('.lora_')[0]
                lora_target_modules.add(module_path)
        
        # これらのモジュールが実際に変更されているか確認（一部だけをチェック）
        module_sample = list(lora_target_modules)[:5]  # 最初の5つだけチェック
        if module_sample and original_model is not None:
            orig_params = dict(original_model.named_parameters())
            new_params = dict(model.named_parameters())
            
            changed_modules = 0
            for module_path in module_sample:
                if module_path in orig_params and module_path in new_params:
                    if not torch.allclose(orig_params[module_path], new_params[module_path]):
                        changed_modules += 1
            
            if changed_modules == 0 and len(module_sample) > 0:
                logger.warning(f"LoRA検証: LoRA対象モジュールに変化が検出されませんでした！ (サンプル: {len(module_sample)})")
                verification_result = False
            else:
                logger.debug(f"LoRA検証: {changed_modules}/{len(module_sample)} モジュールに変化を確認")
    
    return verification_result

# LoRAファイルのハッシュ計算
def compute_file_hash(file_path):
    """ファイルのMD5ハッシュを計算"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)  # 64K chunks
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def load_lora_weights(lora_path: str) -> Dict[str, torch.Tensor]:
    """
    LoRAファイルから重みを読み込む（キャッシュ機能付き）
    
    Args:
        lora_path: LoRAファイルへのパス (.safetensors, .pt, .bin)
    
    Returns:
        Dict[str, torch.Tensor]: LoRAの状態辞書
    """
    global _lora_cache
    
    # キャッシュチェック
    start_time = time()
    file_size_mb = os.path.getsize(lora_path) / (1024 * 1024) if os.path.exists(lora_path) else 0
    file_hash = compute_file_hash(lora_path) if os.path.exists(lora_path) else None
    
    if lora_path in _lora_cache:
        logger.info(f"キャッシュからLoRAを読み込み中: {os.path.basename(lora_path)} (サイズ: {file_size_mb:.2f}MB, ハッシュ: {file_hash[:8]}...)")
        return _lora_cache[lora_path]
    
    logger.info(f"LoRAファイルを読み込み中: {lora_path}")
    
    if not os.path.exists(lora_path):
        logger.error(f"LoRAファイルが見つかりません: {lora_path}")
        raise FileNotFoundError(f"LoRAファイルが見つかりません: {lora_path} (カレントディレクトリ: {os.getcwd()})")
    
    _, ext = os.path.splitext(lora_path.lower())
    
    try:
        logger.info(f"LoRAファイル読み込み開始: {lora_path} (サイズ: {file_size_mb:.2f}MB, ハッシュ: {file_hash[:8]}...)")
        load_start = time()
        
        if ext == '.safetensors':
            state_dict = sf.load_file(lora_path)
            load_time = time() - load_start
            logger.info(f"safetensorsフォーマットからロード完了 (所要時間: {load_time:.2f}秒)")
        else:  # .pt, .bin
            state_dict = torch.load(lora_path, map_location='cpu')
            if isinstance(state_dict, torch.nn.Module):
                state_dict = state_dict.state_dict()
            load_time = time() - load_start
            logger.info(f"Torchフォーマットからロード完了 (所要時間: {load_time:.2f}秒)")
        
        # 詳細なファイル情報をログ出力
        if DEBUG_ENABLED:
            lora_keys = list(state_dict.keys())
            tensor_count = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
            total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
            total_memory_mb = sum(v.element_size() * v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)) / (1024 * 1024)
            
            logger.debug(f"LoRAファイル詳細情報:")
            logger.debug(f"  - テンソル数: {tensor_count}")
            logger.debug(f"  - 総パラメータ数: {total_params:,}")
            logger.debug(f"  - 総メモリサイズ: {total_memory_mb:.2f}MB")
            logger.debug(f"  - キーサンプル: {', '.join(lora_keys[:5]) if len(lora_keys) > 0 else 'なし'}...")
            
            # キーの最初の10個のテンソル統計情報を記録
            for i, (key, value) in enumerate(state_dict.items()):
                if i >= 10 or not isinstance(value, torch.Tensor):
                    continue
                track_tensor_stats(f"lora_file[{key}]", value, "読み込み")
        
        # キーの種類を確認してフォーマットを自動判定
        format_type = detect_lora_format(state_dict)
        logger.info(f"検出されたLoRAフォーマット: {format_type}")
        
        # キャッシュに保存
        _lora_cache[lora_path] = state_dict
        
        return state_dict
    except Exception as e:
        logger.error(f"LoRAファイル読み込みエラー: {e}")
        logger.error(traceback.format_exc())
        raise

def detect_lora_format(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    状態辞書からLoRAフォーマットを検出する
    
    Args:
        state_dict: LoRAの状態辞書
    
    Returns:
        str: フォーマットタイプ ('diffusers', 'hunyuan', 'kohya', 'musubi', 'unknown')
    """
    # キーのサンプルを取得
    keys = list(state_dict.keys())
    sample_keys = keys[:min(10, len(keys))]
    
    # Diffusersフォーマットの検出
    diffusers_pattern = any(['lora_A' in k or 'lora_B' in k for k in sample_keys])
    
    # Hunyuanフォーマットの検出
    hunyuan_pattern = any(['hunyuan_video' in k.lower() for k in sample_keys])
    
    # Kohyaフォーマットの検出
    kohya_pattern = any(['lora_down' in k or 'lora_up' in k for k in sample_keys])
    
    # Musubiフォーマットの検出
    musubi_pattern = any(['lora_unet_' in k and ('alpha' in k or 'lora_down' in k or 'lora_up' in k) for k in sample_keys])
    
    if musubi_pattern:
        return 'musubi'
    elif diffusers_pattern:
        return 'diffusers'
    elif hunyuan_pattern:
        return 'hunyuan'
    elif kohya_pattern:
        return 'kohya'
    else:
        return 'unknown'

def convert_diffusers_lora_to_hunyuan(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    DiffusersフォーマットのLoRAをHunyuan互換形式に変換
    
    Args:
        state_dict: Diffusersフォーマットの状態辞書
    
    Returns:
        Dict[str, torch.Tensor]: Hunyuan互換の状態辞書
    """
    logger.info("DiffusersフォーマットからHunyuanフォーマットへ変換中...")
    converted_dict = {}
    
    # フォーマット変換ロジック
    for key, value in state_dict.items():
        if 'lora_A' in key:
            new_key = key.replace('lora_A', 'lora_down')
            converted_dict[new_key] = value
        elif 'lora_B' in key:
            new_key = key.replace('lora_B', 'lora_up')
            converted_dict[new_key] = value
        else:
            converted_dict[key] = value
    
    logger.info(f"変換完了: {len(converted_dict)} パラメータ")
    return converted_dict

def check_for_musubi(lora_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Musubiフォーマットを検出して変換
    
    Args:
        lora_weights: LoRA重み
    
    Returns:
        Dict[str, torch.Tensor]: 変換後のLoRA重み
    """
    prefix = "lora_unet_"
    musubi = False
    lora_alphas = {}
    
    # Musubiフォーマット検出
    for key, value in lora_weights.items():
        if key.startswith(prefix):
            lora_name = key.split(".", 1)[0]
            if lora_name not in lora_alphas and "alpha" in key:
                lora_alphas[lora_name] = value
                musubi = True
    
    if not musubi:
        return lora_weights
        
    logger.info("Musubiチューナーフォーマットのロード中...")
    converted_lora = {}
    
    for key, weight in lora_weights.items():
        if key.startswith(prefix):
            if "alpha" in key:
                continue
            lora_name = key.split(".", 1)[0]
            module_name = lora_name[len(prefix):]  # remove "lora_unet_"
            module_name = module_name.replace("_", ".")  # replace "_" with "."
            module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
            module_name = module_name.replace("single.blocks.", "single_blocks.")  # fix single blocks
            module_name = module_name.replace("img.", "img_")  # fix img
            module_name = module_name.replace("txt.", "txt_")  # fix txt
            module_name = module_name.replace("attn.", "attn_")  # fix attn
            diffusers_prefix = "diffusion_model"
            
            if "lora_down" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
                dim = weight.shape[0]
            elif "lora_up" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
                dim = weight.shape[1]
            else:
                logger.info(f"予期しないキー: {key} (Musubiフォーマット)")
                continue
                
            # スケーリング
            if lora_name in lora_alphas:
                scale = lora_alphas[lora_name] / dim
                scale = scale.sqrt()
                weight = weight * scale
            else:
                logger.info(f"alpha情報が見つかりません: {lora_name}")
                
            converted_lora[new_key] = weight
    
    logger.info(f"Musubi変換完了: {len(converted_lora)} パラメータ")
    return converted_lora

def filter_lora_weights_by_block_type(lora_weights: Dict[str, torch.Tensor], blocks_type: str = "all") -> Dict[str, torch.Tensor]:
    """
    ブロックタイプに基づいてLoRA重みをフィルタリング
    
    Args:
        lora_weights: フィルタリング前のLoRA重み
        blocks_type: フィルタリングするブロックタイプ
        
    Returns:
        Dict[str, torch.Tensor]: フィルタリング後のLoRA重み
    """
    # パラメータが空の場合はそのまま返す
    if not lora_weights:
        logger.warning("引数が空です。フィルタリングをスキップします。")
        return lora_weights
    
    # すべてのブロックを対象とする場合はフィルタリングなし
    if blocks_type == "all":
        return lora_weights
    
    # 特定のブロックタイプに対するフィルタリング
    filtered_weights = {}
    total_keys = len(lora_weights)
    kept_keys = 0
    
    # プリセットの取得
    base_name, base_layer = PRESET_BLOCKS.get(blocks_type, (None, None))
    
    # 重要なレイヤーのみのフィルタリング
    if blocks_type == "important":
        return filter_lora_weights_by_important_layers(lora_weights)
    
    # ブロックタイプに基づくフィルタリング
    for key, value in lora_weights.items():
        # キーを変換して処理
        parts = key.split('.')
        
        if base_name is None:
            # フィルタリングなし
            filtered_weights[key] = value
            kept_keys += 1
            continue
            
        # ブロック名検出
        if base_name in key:
            if base_layer is None:
                # 指定ブロックタイプのすべてのレイヤー
                filtered_weights[key] = value
                kept_keys += 1
            else:
                # 指定ブロックタイプの特定インデックスレイヤーのみ
                try:
                    # インデックス抽出
                    base_split = key.split('.')
                    block_index = -1
                    for i, part in enumerate(base_split):
                        if part == base_name.split('.')[-1] and i+1 < len(base_split):
                            try:
                                block_index = int(base_split[i+1])
                                break
                            except ValueError:
                                continue
                    
                    if block_index in base_layer:
                        filtered_weights[key] = value
                        kept_keys += 1
                except Exception as e:
                    logger.warning(f"レイヤーインデックス抽出エラー: {key}, {e}")
    
    # フィルタリング後にパラメータが0の場合は、空のセットを返し、LoRA適用を中止
    if not filtered_weights:
        logger.warning(f"ブロックタイプ '{blocks_type}' でのフィルタリング後にパラメータが0になりました。LoRAは適用されません。")
        # デバッグ情報を追加
        if DEBUG_ENABLED:
            logger.debug(f"元キー数: {len(lora_weights)}")
            # キーのサンプルを出力
            key_samples = list(lora_weights.keys())[:10]
            logger.debug(f"元キーサンプル: {key_samples}")
            # base_nameとbase_layerの値を出力
            logger.debug(f"検索対象ブロック名: {base_name}, 検索対象レイヤー: {base_layer}")
        return {}
        
    # パラメータの削減率を計算
    if total_keys > 0:
        filter_percent = (kept_keys / total_keys) * 100.0
        logger.info(f"ブロックタイプ '{blocks_type}' による選択的適用: {kept_keys}/{total_keys} パラメータ保持 ({filter_percent:.2f}%)")
    else:
        logger.warning("全パラメータ数が0です")
        
    return filtered_weights

def filter_lora_weights_by_important_layers(lora_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    重要な層のみにLoRA重みをフィルタリング
    
    Args:
        lora_weights: フィルタリング前のLoRA重み
        
    Returns:
        Dict[str, torch.Tensor]: フィルタリング後のLoRA重み
    """
    # パラメータが空の場合はそのまま返す
    if not lora_weights:
        logger.warning("引数が空です。フィルタリングをスキップします。")
        return lora_weights
        
    important_layers = [
        # 変換ブロック全体（より広範なカバレッジ）
        "transformer_blocks", "single_blocks", "double_blocks", "single_transformer_blocks",
        
        # 注意機構と正規化層（視覚効果に重要）
        "attn", "norm", "rgb_linear",
        
        # 出力・入力層（必須）
        "norm_out", "proj_out", "input_blocks", "output_blocks",
        
        # 一般的なLoRAターゲット層
        "conv", "to_q", "to_k", "to_v", "to_out"
    ]
    
    filtered_weights = {}
    total_keys = len(lora_weights)
    kept_keys = 0
    
    for key, value in lora_weights.items():
        # キーからモジュールパスを抽出
        parts = key.split('.')
        if len(parts) >= 2:
            module_path = f"{parts[0]}.{parts[1]}" if len(parts) > 1 else parts[0]
            
            # 重要層リストと照合
            if any(layer in module_path for layer in important_layers) or any(layer in key for layer in important_layers):
                filtered_weights[key] = value
                kept_keys += 1
    
    # フィルタリング後にパラメータが0の場合は、空のセットを返し、LoRA適用を中止
    if not filtered_weights:
        logger.warning("選択的適用後にパラメータが0になりました。LoRAは適用されません。")
        # デバッグ情報を追加
        if DEBUG_ENABLED:
            logger.debug(f"元キー数: {len(lora_weights)}")
            # キーのサンプルを出力
            key_samples = list(lora_weights.keys())[:10]
            logger.debug(f"元キーサンプル: {key_samples}")
            # 重要レイヤーリストを出力
            logger.debug(f"重要レイヤーリスト: {important_layers}")
            # マッチングしたレイヤーがあるかチェック
            matched_layers = []
            for key in list(lora_weights.keys())[:20]:  # 最初の20キーをチェック
                for layer in important_layers:
                    if layer in key:
                        matched_layers.append((key, layer))
            if matched_layers:
                logger.debug(f"検出されたマッチング例: {matched_layers[:5]}")
            else:
                logger.debug("最初の20キーでマッチするレイヤーがありません。")
        return {}
        
    # パラメータの削減率を計算
    if total_keys > 0:
        filter_percent = (kept_keys / total_keys) * 100.0
        logger.info(f"選択的適用: {kept_keys}/{total_keys} パラメータ保持 ({filter_percent:.2f}%)")
    else:
        logger.warning("全パラメータ数が0です")
        
    return filtered_weights

def prune_lora_weights(lora_weights: Dict[str, torch.Tensor], threshold: float = 0.0005) -> Dict[str, torch.Tensor]:
    """
    閾値以下の小さなLoRAパラメータを0に設定
    
    Args:
        lora_weights: プルーニング前のLoRA重み
        threshold: プルーニング閾値（絶対値）
        
    Returns:
        Dict[str, torch.Tensor]: プルーニング後のLoRA重み
    """
    # パラメータが空の場合はそのまま返す
    if not lora_weights:
        logger.warning("引数が空です。プルーニングをスキップします。")
        return lora_weights
        
    pruned_weights = {}
    total_params = 0
    pruned_params = 0
    
    for key, value in lora_weights.items():
        if isinstance(value, torch.Tensor):
            total_params += value.numel()
            # 閾値以下の値をゼロにマスク
            mask = torch.abs(value) >= threshold
            pruned_tensor = value * mask
            pruned_params += (value.numel() - mask.sum().item())
            pruned_weights[key] = pruned_tensor
        else:
            pruned_weights[key] = value
    
    # ゼロ除算を確実に回避
    if total_params > 0:  # 0除算を避ける
        prune_percent = (pruned_params / total_params) * 100.0
        logger.info(f"プルーニング: {pruned_params}/{total_params} パラメータ削減 ({prune_percent:.2f}%)")
    else:
        logger.warning("プルーニング対象のパラメータが0です。プルーニングをスキップします。")
    
    return pruned_weights

def log_memory_usage(message: str) -> None:
    """
    現在のGPUメモリ使用状況をログに出力
    
    Args:
        message: ログメッセージのプレフィックス
    """
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"{message}: GPU使用メモリ {allocated:.2f}GB (予約: {reserved:.2f}GB, 最大: {max_memory:.2f}GB)")
            
            # デバッグモードの場合は詳細情報を出力
            if DEBUG_ENABLED:
                # 各デバイスごとのメモリ使用状況
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    device_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    device_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.debug(f"  CUDA:{i} - 使用: {device_allocated:.2f}GB, 予約: {device_reserved:.2f}GB")
                
                # キャッシュメモリもクリア
                torch.cuda.empty_cache()
                logger.debug(f"  キャッシュクリア後 - 使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB, 予約: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        except Exception as e:
            logger.warning(f"メモリ使用状況の取得に失敗: {e}")
            logger.warning(traceback.format_exc())
            
def detailed_pruning_log(original_weights, pruned_weights):
    """
    プルーニング前後の状態を詳細に分析してログ出力
    
    Args:
        original_weights: プルーニング前の重み
        pruned_weights: プルーニング後の重み
    """
    total_elements = sum(t.numel() for t in original_weights.values() if isinstance(t, torch.Tensor))
    non_zero_after = sum((t != 0).sum().item() for t in pruned_weights.values() if isinstance(t, torch.Tensor))
    
    memory_before = sum(t.element_size() * t.numel() for t in original_weights.values() if isinstance(t, torch.Tensor)) / 1024**2
    memory_after = sum(t.element_size() * t.numel() for t in pruned_weights.values() if isinstance(t, torch.Tensor)) / 1024**2
    
    logger.info(f"プルーニング詳細: 非ゼロ要素 {non_zero_after}/{total_elements} ({non_zero_after/total_elements*100:.2f}%)")
    logger.info(f"メモリ削減: {memory_before:.2f}MB → {memory_after:.2f}MB (節約: {memory_before-memory_after:.2f}MB)")

def apply_lora_to_model(
    model: torch.nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    scale: float = 0.8,
    format_type: Optional[str] = None
) -> torch.nn.Module:
    """
    モデルにLoRAを適用する
    
    Args:
        model: ベースモデル
        lora_state_dict: LoRAの状態辞書
        scale: LoRAの適用強度 (0.0-1.0)
        format_type: 明示的なフォーマットタイプ ('diffusers', 'hunyuan', 'kohya', None)
        
    Returns:
        torch.nn.Module: LoRAが適用されたモデル
    """
    logger.info(f"LoRAをモデルに適用中 (スケール係数: {scale})")
    
    # フォーマット自動検出（指定されていない場合）
    if format_type is None:
        format_type = detect_lora_format(lora_state_dict)
    
    # フォーマットに応じて変換
    if format_type == 'diffusers':
        lora_state_dict = convert_diffusers_lora_to_hunyuan(lora_state_dict)
    elif format_type == 'musubi':
        lora_state_dict = check_for_musubi(lora_state_dict)
    
    # モデルの現在の状態を保存
    original_state = {}
    modified_modules = []
    
    # パラメータ解析用の統計をクリア
    if DEBUG_ENABLED:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.debug(f"モデル情報: パラメータ数={total_params:,}, 学習可能={trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.debug(f"適用するLoRAの情報: キー数={len(lora_state_dict)}, スケール={scale}")
        
        # モデル構造のサマリー
        model_structure = {}
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type not in model_structure:
                model_structure[module_type] = 0
            model_structure[module_type] += 1
        
        # 上位10種類のモジュールタイプを出力
        sorted_modules = sorted(model_structure.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.debug(f"モデル構造(Top10): {sorted_modules}")
    
    try:
        # LoRAの適用
        with torch.no_grad():
            for name, param in model.named_parameters():
                # LoRAに対応するキーを探す
                lora_down_key = f"{name}.lora_down"
                lora_up_key = f"{name}.lora_up"
                
                if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                    # 元の値を保存
                    original_state[name] = param.data.clone()
                    
                    # LoRAの重みを取得
                    lora_down = lora_state_dict[lora_down_key].to(param.device)
                    lora_up = lora_state_dict[lora_up_key].to(param.device)
                    
                    # LoRAの演算
                    delta = torch.matmul(lora_up, lora_down) * scale
                    
                    # 形状を確認して調整
                    if delta.shape != param.shape:
                        logger.warning(f"形状不一致: {delta.shape} != {param.shape}, スキップ: {name}")
                        continue
                    
                    # 適用前のパラメータ統計情報を記録
                    if DEBUG_ENABLED:
                        track_tensor_stats(f"before_{name}", param.data, "LoRA適用前")
                        track_tensor_stats(f"delta_{name}", delta, "LoRAデルタ")
                    
                    # 元のパラメータに適用
                    param.data += delta
                    modified_modules.append(name)
                    
                    # 適用後のパラメータ統計情報を記録
                    if DEBUG_ENABLED:
                        track_tensor_stats(f"after_{name}", param.data, "LoRA適用後")
                        
                        # LoRA適用による変化率を計算
                        orig_norm = torch.norm(original_state[name])
                        delta_norm = torch.norm(delta)
                        if orig_norm > 0:
                            change_ratio = delta_norm / orig_norm * 100
                            logger.debug(f"LoRA影響度 [{name}]: 変化率={change_ratio:.4f}%, 元ノルム={orig_norm:.6f}, デルタノルム={delta_norm:.6f}")
        
        logger.info(f"LoRA適用完了: {len(modified_modules)} モジュールが修正されました")
        return model
    
    except Exception as e:
        # エラー時には元の状態に戻す
        logger.error(f"LoRA適用エラー: {e}")
        logger.error(traceback.format_exc())
        
        with torch.no_grad():
            for name, original_data in original_state.items():
                param = dict(model.named_parameters())[name]
                param.data.copy_(original_data)
        
        logger.info("エラーにより元の状態に復元しました")
        raise

def load_and_apply_lora(
    model: torch.nn.Module,
    lora_path: str,
    scale: float = 0.8,
    is_diffusers: bool = False,
    selective_application: bool = None,  # Noneを許容し、デフォルト値を環境設定から取得
    pruning: bool = True,
    pruning_threshold: float = 0.0005,
    blocks_type: str = "all"
) -> torch.nn.Module:
    """
    LoRAをロードしてモデルに適用する便利関数
    
    Args:
        model: ベースモデル
        lora_path: LoRAファイルへのパス
        scale: LoRAの適用強度 (0.0-1.0)
        is_diffusers: Diffusersフォーマットかどうか
        selective_application: 選択的なLoRA適用を行うかどうか
        pruning: LoRAパラメータのプルーニングを行うかどうか
        pruning_threshold: プルーニングの閾値
        blocks_type: フィルタリングするブロックタイプ
        
    Returns:
        torch.nn.Module: LoRAが適用されたモデル
    """
    # selective_applicationのデフォルト値を環境設定から取得
    if selective_application is None:
        selective_application = SELECTIVE_APPLICATION_DEFAULT
        
    logger.info(f"LoRAを読み込み・適用: {lora_path}, スケール: {scale}, ブロックタイプ: {blocks_type}, 選択的適用: {selective_application}")
    
    # LoRAの読み込み
    lora_state_dict = load_lora_weights(lora_path)
    
    # 明示的なフォーマット指定（オプション）
    format_type = 'diffusers' if is_diffusers else None
    format_type = format_type or detect_lora_format(lora_state_dict)
    
    # フォーマットに応じて変換
    if format_type == 'diffusers':
        lora_state_dict = convert_diffusers_lora_to_hunyuan(lora_state_dict)
    elif format_type == 'musubi':
        lora_state_dict = check_for_musubi(lora_state_dict)
        
    logger.info(f"元のLoRA: {len(lora_state_dict)} パラメータ")
    
    # ブロックタイプに基づくフィルタリング
    if blocks_type != "all" and lora_state_dict:
        try:
            filtered_state_dict = filter_lora_weights_by_block_type(lora_state_dict, blocks_type)
            if filtered_state_dict:  # 空でない場合のみ更新
                lora_state_dict = filtered_state_dict
        except Exception as e:
            logger.warning(f"ブロックタイプによるフィルタリング中にエラーが発生しました: {e}")
            logger.warning(traceback.format_exc())
            logger.info("ブロックフィルタリングをスキップし、標準的な選択的適用を使用します")
            
            # 標準的な選択的適用にフォールバック
            if selective_application:
                try:
                    filtered_state_dict = filter_lora_weights_by_important_layers(lora_state_dict)
                    if filtered_state_dict:  # 空でない場合のみ更新
                        lora_state_dict = filtered_state_dict
                except Exception as e:
                    logger.warning(f"選択的適用中にエラーが発生しました: {e}")
                    logger.warning(traceback.format_exc())
                    logger.info("選択的適用をスキップし、全パラメータを使用します")
        
    # フィルタリングなしの場合の選択的適用
    elif selective_application and blocks_type == "all" and lora_state_dict:
        try:
            filtered_state_dict = filter_lora_weights_by_important_layers(lora_state_dict)
            if filtered_state_dict:  # 空でない場合のみ更新
                lora_state_dict = filtered_state_dict
        except Exception as e:
            logger.warning(f"選択的適用中にエラーが発生しました: {e}")
            logger.warning(traceback.format_exc())
            logger.info("選択的適用をスキップし、全パラメータを使用します")
    
    # LoRAパラメータのプルーニング
    if pruning and lora_state_dict:
        try:
            log_memory_usage("プルーニング前")
            start_time = time()
            
            # プルーニング前のテンソル統計
            if DEBUG_ENABLED:
                lora_down_tensors = [v for k, v in lora_state_dict.items() if 'lora_down' in k]
                lora_up_tensors = [v for k, v in lora_state_dict.items() if 'lora_up' in k]
                
                if lora_down_tensors and lora_up_tensors:
                    down_stats = {
                        "count": len(lora_down_tensors),
                        "total_elements": sum(t.numel() for t in lora_down_tensors),
                        "non_zero": sum((t != 0).sum().item() for t in lora_down_tensors),
                        "min": min(t.min().item() for t in lora_down_tensors),
                        "max": max(t.max().item() for t in lora_down_tensors),
                        "mean_abs": np.mean([torch.abs(t).mean().item() for t in lora_down_tensors])
                    }
                    
                    up_stats = {
                        "count": len(lora_up_tensors),
                        "total_elements": sum(t.numel() for t in lora_up_tensors),
                        "non_zero": sum((t != 0).sum().item() for t in lora_up_tensors),
                        "min": min(t.min().item() for t in lora_up_tensors),
                        "max": max(t.max().item() for t in lora_up_tensors),
                        "mean_abs": np.mean([torch.abs(t).mean().item() for t in lora_up_tensors])
                    }
                    
                    logger.debug(f"lora_down統計: キー数={down_stats['count']}, 総要素数={down_stats['total_elements']:,}, 非ゼロ={down_stats['non_zero']:,} ({down_stats['non_zero']/down_stats['total_elements']*100:.2f}%), 絶対値平均={down_stats['mean_abs']:.6f}")
                    logger.debug(f"lora_up統計: キー数={up_stats['count']}, 総要素数={up_stats['total_elements']:,}, 非ゼロ={up_stats['non_zero']:,} ({up_stats['non_zero']/up_stats['total_elements']*100:.2f}%), 絶対値平均={up_stats['mean_abs']:.6f}")
            
            # プルーニング実行
            pruned_state_dict = prune_lora_weights(lora_state_dict, pruning_threshold)
            if pruned_state_dict:  # 空でない場合のみ更新
                # 詳細ログ出力
                detailed_pruning_log(lora_state_dict, pruned_state_dict)
                lora_state_dict = pruned_state_dict
                logger.info(f"プルーニング処理時間: {time() - start_time:.2f}秒")
                log_memory_usage("プルーニング後")
                
                # プルーニング後のテンソル統計
                if DEBUG_ENABLED:
                    pruned_down_tensors = [v for k, v in lora_state_dict.items() if 'lora_down' in k]
                    pruned_up_tensors = [v for k, v in lora_state_dict.items() if 'lora_up' in k]
                    
                    if pruned_down_tensors and pruned_up_tensors:
                        p_down_stats = {
                            "count": len(pruned_down_tensors),
                            "total_elements": sum(t.numel() for t in pruned_down_tensors),
                            "non_zero": sum((t != 0).sum().item() for t in pruned_down_tensors),
                        }
                        
                        p_up_stats = {
                            "count": len(pruned_up_tensors),
                            "total_elements": sum(t.numel() for t in pruned_up_tensors),
                            "non_zero": sum((t != 0).sum().item() for t in pruned_up_tensors),
                        }
                        
                        down_reduction = (1 - p_down_stats['non_zero'] / down_stats['non_zero']) * 100 if down_stats['non_zero'] > 0 else 0
                        up_reduction = (1 - p_up_stats['non_zero'] / up_stats['non_zero']) * 100 if up_stats['non_zero'] > 0 else 0
                        
                        logger.debug(f"プルーニング後lora_down: 非ゼロ={p_down_stats['non_zero']:,} ({p_down_stats['non_zero']/p_down_stats['total_elements']*100:.2f}%), 削減率={down_reduction:.2f}%")
                        logger.debug(f"プルーニング後lora_up: 非ゼロ={p_up_stats['non_zero']:,} ({p_up_stats['non_zero']/p_up_stats['total_elements']*100:.2f}%), 削減率={up_reduction:.2f}%")
        except Exception as e:
            logger.warning(f"プルーニング中にエラーが発生しました: {e}")
            logger.warning(traceback.format_exc())
            logger.info("プルーニングをスキップします")
    
    # モデルに適用
    if lora_state_dict:
        # 適用前のモデルパラメータ統計
        if DEBUG_ENABLED:
            log_memory_usage("LoRA適用前")
            logger.debug(f"LoRA適用前モデルID: {id(model)}")
            # 重要パラメータの保存（比較用）
            before_params = {}
            for name, param in model.named_parameters():
                if len(before_params) < 5:  # 最初の5つだけ保存
                    before_params[name] = param.data.clone()
        
        # LoRA適用
        model = apply_lora_to_model(model, lora_state_dict, scale)
        
        # 適用後のパラメータ検証
        if DEBUG_ENABLED:
            log_memory_usage("LoRA適用後")
            logger.debug(f"LoRA適用後モデルID: {id(model)}")
            # パラメータ変化を確認
            changes_detected = 0
            for name, param in model.named_parameters():
                if name in before_params:
                    if not torch.allclose(before_params[name], param.data):
                        diff = torch.abs(before_params[name] - param.data)
                        logger.debug(f"パラメータ変化検出 [{name}]: 最大差={diff.max().item():.8f}, 平均差={diff.mean().item():.8f}")
                        changes_detected += 1
            logger.debug(f"パラメータ変化検出: {changes_detected}/5")
    else:
        logger.warning("LoRAパラメータが空です。モデルに適用しません。")
    
    return model
