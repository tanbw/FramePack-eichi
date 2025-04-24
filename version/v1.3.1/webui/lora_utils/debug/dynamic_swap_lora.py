# endframe_ichi.pyのためのヘルパー関数

# 低VRAMモードでもtransformer_loraを使用するかチェックする関数
def should_use_transformer_lora() -> bool:
    """
    低VRAMモードでもtransformer_loraを使用するかどうかをチェック
    
    Returns:
        bool: transformer_loraを使用すべきかどうか
    """
    return FORCE_TRANSFORMER_LORA

# LoRA適用後の確認用関数
def verify_lora_effect(original_model, modified_model, name=""):
    """
    LoRAの適用による変化を確認する
    
    Args:
        original_model: 元のモデル
        modified_model: 変更後のモデル
        name: 検証名
    
    Returns:
        bool: 変化があればTrue
    """
    if not DEBUG_ENABLED:
        return True
        
    # モデルが同じオブジェクトだった場合は比較できない
    if id(original_model) == id(modified_model):
        logger.debug(f"{name} - モデルはIDが同じなので内部変化のみを確認しました。ID: {id(original_model)}")
        return True
    
    # パラメータの比較
    orig_params = dict(original_model.named_parameters())
    mod_params = dict(modified_model.named_parameters())
    
    total_params = 0
    different_params = 0
    max_diff = 0.0
    avg_diff = 0.0
    diff_params = []
    
    # 共通パラメータの比較
    common_params = set(orig_params.keys()).intersection(set(mod_params.keys()))
    for param_name in common_params:
        total_params += 1
        orig_param = orig_params[param_name]
        mod_param = mod_params[param_name]
        
        # デバイスとデータ型を合わせる
        if orig_param.device != mod_param.device or orig_param.dtype != mod_param.dtype:
            mod_param = mod_param.to(device=orig_param.device, dtype=orig_param.dtype)
        
        # 差分計算
        if not torch.allclose(orig_param, mod_param, rtol=1e-5, atol=1e-5, equal_nan=True):
            different_params += 1
            diff = torch.abs(orig_param - mod_param)
            current_max_diff = diff.max().item()
            current_avg_diff = diff.mean().item()
            max_diff = max(max_diff, current_max_diff)
            avg_diff += current_avg_diff
            
            # 最初の10個だけ保存
            if len(diff_params) < 10:
                diff_params.append((param_name, current_max_diff, current_avg_diff))
    
    # 平均差分の計算
    if different_params > 0:
        avg_diff /= different_params
    
    # 結果をログに出力
    if different_params > 0:
        logger.debug(f"{name} - LoRA適用確認: {different_params}/{total_params} パラメータが変化しました")
        logger.debug(f"  - 最大差分: {max_diff}, 平均差分: {avg_diff}")
        
        # 最大差分を持つパラメータを出力
        for param_name, max_d, avg_d in diff_params:
            logger.debug(f"  - {param_name}: 最大差={max_d:.8f}, 平均差={avg_d:.8f}")
        
        return True
    else:
        logger.warning(f"{name} - LoRA適用確認失敗: 変化したパラメータがありません！")
        return False

# 低VRAMモードでもtransformer_loraを使用するかチェックする関数
def should_use_transformer_lora() -> bool:
    """
    低VRAMモードでもtransformer_loraを使用するかどうかをチェック
    
    Returns:
        bool: transformer_loraを使用すべきかどうか
    """
    return FORCE_TRANSFORMER_LORA

# LoRA適用後の確認用関数
def verify_lora_effect(original_model, modified_model, name=""):
    """
    LoRAの適用による変化を確認する
    
    Args:
        original_model: 元のモデル
        modified_model: 変更後のモデル
        name: 検証名
    
    Returns:
        bool: 変化があればTrue
    """
    if not DEBUG_ENABLED:
        return True
        
    # モデルが同じオブジェクトだった場合は比較できない
    if id(original_model) == id(modified_model):
        logger.debug(f"{name} - モデルはIDが同じなので内部変化のみを確認しました。ID: {id(original_model)}")
        return True
    
    # パラメータの比較
    orig_params = dict(original_model.named_parameters())
    mod_params = dict(modified_model.named_parameters())
    
    total_params = 0
    different_params = 0
    max_diff = 0.0
    avg_diff = 0.0
    diff_params = []
    
    # 共通パラメータの比較
    common_params = set(orig_params.keys()).intersection(set(mod_params.keys()))
    for param_name in common_params:
        total_params += 1
        orig_param = orig_params[param_name]
        mod_param = mod_params[param_name]
        
        # デバイスとデータ型を合わせる
        if orig_param.device != mod_param.device or orig_param.dtype != mod_param.dtype:
            mod_param = mod_param.to(device=orig_param.device, dtype=orig_param.dtype)
        
        # 差分計算
        if not torch.allclose(orig_param, mod_param, rtol=1e-5, atol=1e-5, equal_nan=True):
            different_params += 1
            diff = torch.abs(orig_param - mod_param)
            current_max_diff = diff.max().item()
            current_avg_diff = diff.mean().item()
            max_diff = max(max_diff, current_max_diff)
            avg_diff += current_avg_diff
            
            # 最初の10個だけ保存
            if len(diff_params) < 10:
                diff_params.append((param_name, current_max_diff, current_avg_diff))
    
    # 平均差分の計算
    if different_params > 0:
        avg_diff /= different_params
    
    # 結果をログに出力
    if different_params > 0:
        logger.debug(f"{name} - LoRA適用確認: {different_params}/{total_params} パラメータが変化しました")
        logger.debug(f"  - 最大差分: {max_diff}, 平均差分: {avg_diff}")
        
        # 最大差分を持つパラメータを出力
        for param_name, max_d, avg_d in diff_params:
            logger.debug(f"  - {param_name}: 最大差={max_d:.8f}, 平均差={avg_d:.8f}")
        
        return True
    else:
        logger.warning(f"{name} - LoRA適用確認失敗: 変化したパラメータがありません！")
        return False"""
FramePack-eichi DynamicSwap対応LoRAモジュール
DynamicSwapメモリ管理システムと互換性のあるLoRA適用機能を提供します。
拡張機能: ブロック選択とMusubiフォーマット対応
"""

import os
import torch
import logging
import traceback
import time
import numpy as np
import json
from typing import Dict, List, Optional, Union, Tuple, Any

# 環境変数による設定読み込み
ENV_DEBUG_LEVEL = os.environ.get('LORA_DEBUG_LEVEL', 'INFO')
ENV_FORCE_TRANSFORMER_LORA = os.environ.get('FORCE_TRANSFORMER_LORA', '0')
ENV_LOW_VRAM_LORA = os.environ.get('LOW_VRAM_LORA', '0')

# デバッグ設定
DEBUG_ENABLED = ENV_DEBUG_LEVEL.upper() in ('DEBUG', 'TRACE', 'ALL')
FORCE_TRANSFORMER_LORA = ENV_FORCE_TRANSFORMER_LORA.lower() in ('1', 'true', 'yes', 'on')
LOW_VRAM_LORA = ENV_LOW_VRAM_LORA.lower() in ('1', 'true', 'yes', 'on')

# LoRAローダーをインポート
from .lora_loader import (load_lora_weights, detect_lora_format, convert_diffusers_lora_to_hunyuan, 
                        check_for_musubi, filter_lora_weights_by_important_layers, 
                        filter_lora_weights_by_block_type, prune_lora_weights, PRESET_BLOCKS)

# ロギング設定
logger = logging.getLogger("dynamic_swap_lora")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 環境変数に基づくログレベル設定
log_level = logging.DEBUG if DEBUG_ENABLED else logging.INFO
logger.setLevel(log_level)

# 設定情報のログ出力
logger.info(f"===== DynamicSwapLoRA設定情報 =====")
logger.info(f"デバッグモード: {DEBUG_ENABLED}")
logger.info(f"強制transformer_lora使用: {FORCE_TRANSFORMER_LORA}")
logger.info(f"低 VRAMモード: {LOW_VRAM_LORA}")
logger.info(f"ログレベル: {logging.getLevelName(log_level)}")
logger.info(f"==================================")

# デバッグ用データストア
_debug_data = {
    "operation_times": {},      # 処理時間記録
    "memory_usage": {},        # メモリ使用状況
    "lora_stats": {},          # LoRA統計情報
    "application_counts": {},  # 適用したレイヤー数
    "errors": []               # エラー記録
}

# 処理時間を記録するデコレータ
def timing_decorator(func):
    """Function execution time measuring decorator"""
    def wrapper(*args, **kwargs):
        if not DEBUG_ENABLED:
            return func(*args, **kwargs)
            
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            func_name = func.__name__
            if func_name not in _debug_data["operation_times"]:
                _debug_data["operation_times"][func_name] = []
            _debug_data["operation_times"][func_name].append(execution_time)
            logger.debug(f"[TIMING] {func_name}: {execution_time:.4f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            func_name = func.__name__
            logger.error(f"[TIMING-ERROR] {func_name}: {execution_time:.4f}秒 - {str(e)}")
            if func_name not in _debug_data["operation_times"]:
                _debug_data["operation_times"][func_name] = []
            _debug_data["operation_times"][func_name].append(execution_time)
            _debug_data["errors"].append({"function": func_name, "error": str(e), "traceback": traceback.format_exc()})
            raise
    return wrapper

# テンソル統計情報の収集
def collect_tensor_stats(tensor, name="テンソル"):
    """Collect and return tensor statistics"""
    if not DEBUG_ENABLED or not isinstance(tensor, torch.Tensor):
        return {}
        
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
            "memory_mb": float(tensor.element_size() * tensor.numel() / (1024 * 1024))
        }
        
        logger.debug(f"テンソル統計 [{name}]: shape={stats['shape']}, 非ゼロ={stats['non_zero']}/{stats['total']} ({stats['non_zero']/stats['total']*100:.2f}%), min={stats['min']:.6f}, max={stats['max']:.6f}")
        return stats
    except Exception as e:
        logger.debug(f"テンソル統計収集エラー [{name}]: {e}")
        return {}

# デバッグ情報の保存
def save_debug_info(filename="lora_debug_info.json"):
    """Save debug information to a file"""
    if not DEBUG_ENABLED:
        return
        
    try:
        # 保存先のパスを生成
        current_dir = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.join(current_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        filepath = os.path.join(debug_dir, filename)
        
        # 現在の時分を追加
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        _debug_data["timestamp"] = timestamp
        
        # 処理時間の平均計算
        for func_name, times in _debug_data["operation_times"].items():
            if times:
                _debug_data["operation_times"][func_name] = {
                    "calls": len(times),
                    "total": sum(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times)
                }
        
        # JSONファイルに保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(_debug_data, f, ensure_ascii=False, indent=2)
            
        logger.debug(f"デバッグ情報を保存しました: {filepath}")
    except Exception as e:
        logger.error(f"デバッグ情報の保存中にエラーが発生しました: {e}")

class OptimizedDynamicSwapLoRA:
    """
    メモリ使用量を最適化したDynamicSwapに対応するLoRAクラス
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.original_state = {}
        self.applied_keys = set()
        
        # デバッグ用統計情報
        self.stats = {
            "params_modified": 0,
            "params_skipped": 0,
            "errors": 0,
            "total_param_size": 0,
            "start_time": time.time()
        }
        
        # デバッグモードならモデル情報を出力
        if DEBUG_ENABLED:
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model_memory = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 * 1024)
                
                logger.debug(f"OptimizedDynamicSwapLoRA初期化: モデルID={id(model)}")
                logger.debug(f"  - パラメータ数: {total_params:,}")
                logger.debug(f"  - 学習可能パラメータ: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
                logger.debug(f"  - メモリ使用量: {model_memory:.2f}MB")
            except Exception as e:
                logger.debug(f"モデル情報収集エラー: {e}")
    
    @timing_decorator
    def install(self, lora_weights: Dict[str, torch.Tensor], scale: float = 1.0) -> torch.nn.Module:
        """
        最適化されたDynamicSwapモードでLoRAをインストール
        
        Args:
            lora_weights: LoRA重み
            scale: 適用スケール
            
        Returns:
            torch.nn.Module: LoRAが適用されたモデル
        """
        if DEBUG_ENABLED:
            logger.debug(f"LoRAインストール開始: 重み数={len(lora_weights)}, スケール={scale}")
            self.log_memory_usage("インストール前")
        
        start_time = time.time()
        
        # 元の状態を最小限だけ保存
        self._save_minimal_original_state(lora_weights)
        
        # LoRAを適用
        self._apply_weights(lora_weights, scale)
        
        # 統計情報を更新
        elapsed_time = time.time() - start_time
        self.stats["elapsed_time"] = elapsed_time
        
        if DEBUG_ENABLED:
            self.log_memory_usage("インストール後")
            logger.debug(f"LoRAインストール完了: 所要時間={elapsed_time:.2f}秒, 変更パラメータ={self.stats['params_modified']}, スキップ={self.stats['params_skipped']}")
            
        logger.info(f"最適化済みDynamicSwapモードでLoRAをインストールしました (所要時間: {elapsed_time:.2f}秒, パラメータ変更数: {self.stats['params_modified']})")        
        return self.model
    
    @timing_decorator
    def _save_minimal_original_state(self, lora_weights: Dict[str, torch.Tensor]) -> None:
        """
        変更される部分のみ元の状態を保存
        
        Args:
            lora_weights: LoRA重み
        """
        saved_params = 0
        saved_memory = 0
        lora_keys = set(lora_weights.keys())
        
        # 事前に変更対象となるモジュールの一覧を抽出
        target_modules = set()
        for key in lora_keys:
            if ".lora_down" in key or ".lora_up" in key:
                parts = key.split('.')
                # 先頭2つのセグメントをモジュール名として抽出
                if len(parts) >= 3:
                    module_path = '.'.join(parts[:-2])
                    target_modules.add(module_path)
        
        if DEBUG_ENABLED:
            logger.debug(f"変更対象モジュール数: {len(target_modules)}")
            # 最初の5つのモジュール名を表示
            sample_modules = list(target_modules)[:5]
            logger.debug(f"モジュールサンプル: {sample_modules}")
        
        for name, module in self.model.named_modules():
            # ターゲットモジュールのいずれかにマッチするか確認
            if not any(name == target_module or name.startswith(target_module + '.') for target_module in target_modules):
                continue
                
            # このモジュールがLoRAで変更される場合のみ保存
            if name not in self.original_state:
                param_count = 0
                self.original_state[name] = {}
                
                for param_name, param in module.named_parameters(recurse=False):
                    if param.numel() > 0:
                        # CPUメモリに保存してGPUメモリを節約
                        self.original_state[name][param_name] = param.detach().cpu().clone()
                        saved_memory += param.element_size() * param.numel()
                        param_count += 1
                
                if param_count > 0:
                    saved_params += 1
                    if DEBUG_ENABLED and saved_params <= 5:  # 最初の5つのみログ出力
                        logger.debug(f"  保存モジュール: {name} (パラメータ数: {param_count})")
        
        # 保存統計情報を更新
        self.stats["saved_modules"] = saved_params
        self.stats["saved_memory_mb"] = saved_memory / (1024 * 1024)
        
        logger.info(f"元の状態を保存: {saved_params} モジュール（最適化済み）, メモリ: {self.stats['saved_memory_mb']:.2f}MB")
    
    @timing_decorator
    def _apply_weights(self, lora_weights: Dict[str, torch.Tensor], scale: float) -> None:
        """
        LoRA重みをモデルに適用
        
        Args:
            lora_weights: LoRA重み
            scale: 適用スケール
        """
        params_modified = 0
        params_skipped = 0
        total_delta_norm = 0.0
        total_param_norm = 0.0
        total_param_size = 0
        
        # LoRAキーをモジュールごとにグループ化
        module_lora_keys = {}
        for key in lora_weights.keys():
            if ".lora_down" in key or ".lora_up" in key:
                parts = key.split('.')
                if len(parts) >= 3:
                    module_path = '.'.join(parts[:-2])
                    param_name = parts[-2]
                    lora_type = "lora_down" if ".lora_down" in key else "lora_up"
                    
                    if module_path not in module_lora_keys:
                        module_lora_keys[module_path] = {}
                    if param_name not in module_lora_keys[module_path]:
                        module_lora_keys[module_path][param_name] = {}
                    
                    module_lora_keys[module_path][param_name][lora_type] = key
        
        if DEBUG_ENABLED:
            logger.debug(f"LoRA適用対象モジュール数: {len(module_lora_keys)}")
        
        # モジュールごとに処理
        with torch.no_grad():
            for module_path, params in module_lora_keys.items():
                # モジュールを取得
                module = self.model
                valid_module = True
                for name in module_path.split('.'):
                    if not name:
                        continue
                    try:
                        module = getattr(module, name)
                    except AttributeError:
                        logger.warning(f"モジュール {name} が見つかりません: {module_path}")
                        valid_module = False
                        break
                
                if not valid_module:
                    params_skipped += len(params)
                    continue
                
                # パラメータごとに処理
                for param_name, lora_types in params.items():
                    # 両方のタイプがある場合のみ処理
                    if "lora_down" in lora_types and "lora_up" in lora_types:
                        if not hasattr(module, param_name):
                            params_skipped += 1
                            continue
                            
                        # パラメータを取得
                        param = getattr(module, param_name)
                        if not isinstance(param, torch.nn.Parameter):
                            params_skipped += 1
                            continue
                        
                        # LoRAの重みを取得
                        lora_down_key = lora_types["lora_down"]
                        lora_up_key = lora_types["lora_up"]
                        lora_down = lora_weights[lora_down_key].to(param.device)
                        lora_up = lora_weights[lora_up_key].to(param.device)
                        
                        # LoRAデルタの計算
                        try:
                            delta = torch.matmul(lora_up, lora_down) * scale
                            
                            # 形状確認
                            if delta.shape != param.shape:
                                params_skipped += 1
                                logger.warning(f"形状不一致: {delta.shape} != {param.shape}, スキップ: {module_path}.{param_name}")
                                continue
                                
                            # デバッグ時のみ詳細統計を記録
                            if DEBUG_ENABLED:
                                param_norm = torch.norm(param.data)
                                delta_norm = torch.norm(delta)
                                delta_ratio = (delta_norm / param_norm * 100) if param_norm > 0 else 0
                                
                                if params_modified < 5:  # 最初の5つだけ詳細ログを出力
                                    logger.debug(f"LoRA適用 [{module_path}.{param_name}]: 形状={param.shape}, 変化率={delta_ratio:.4f}%, 原ノルム={param_norm:.6f}, デルタノルム={delta_norm:.6f}")
                                
                                total_delta_norm += delta_norm.item()
                                total_param_norm += param_norm.item()
                                total_param_size += param.numel() * param.element_size()
                            
                            # 実際にパラメータに適用
                            param.data += delta
                            params_modified += 1
                            
                            # 適用済みキーを記録
                            if module_path not in self.applied_keys:
                                self.applied_keys.add(module_path)
                                
                        except Exception as e:
                            params_skipped += 1
                            logger.error(f"LoRA適用エラー [{module_path}.{param_name}]: {str(e)}")
                            if DEBUG_ENABLED:
                                logger.debug(f"lora_down: {lora_down.shape}, lora_up: {lora_up.shape}, param: {param.shape}")
                    else:
                        # 片方のタイプしかない場合はスキップ
                        params_skipped += 1
        
        # 統計情報を更新
        self.stats["params_modified"] = params_modified
        self.stats["params_skipped"] = params_skipped
        self.stats["total_param_size"] = total_param_size / (1024 * 1024)  # MB単位
        
        if DEBUG_ENABLED and params_modified > 0:
            avg_change_ratio = (total_delta_norm / total_param_norm * 100) if total_param_norm > 0 else 0
            logger.debug(f"LoRA適用統計: 変更パラメータ数={params_modified}, スキップ={params_skipped}, 平均変化率={avg_change_ratio:.4f}%, 総メモリ={self.stats['total_param_size']:.2f}MB")
    
    @timing_decorator
    def uninstall(self) -> torch.nn.Module:
        """
        LoRAを削除し、元の状態に戻す
        
        Returns:
            torch.nn.Module: 元の状態に戻されたモデル
        """
        with torch.no_grad():
            for module_path, params in self.original_state.items():
                # モジュールを取得（再帰的に探索）
                module = self.model
                for name in module_path.split('.'):
                    if not name:
                        continue
                    try:
                        module = getattr(module, name)
                    except AttributeError:
                        logger.warning(f"モジュール {name} が見つかりません: {module_path}")
                        break
                
                # 元の状態に戻す
                for param_name, original_value in params.items():
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        if isinstance(param, torch.nn.Parameter):
                            # 保存された元の状態がCPUにある場合は、デバイスを合わせる
                            if original_value.device != param.device:
                                original_value = original_value.to(param.device)
                            
                            # 現在のパラメータを元の状態に戻す
                            param.data.copy_(original_value)
                            logger.debug(f"LoRA削除: {module_path}.{param_name}")
        
        # 状態をリセット
        self.original_state = {}
        self.applied_keys.clear()
        
        logger.info("LoRAをアンインストールしました")
        return self.model


class DynamicSwapLoRAManager:
    """
    DynamicSwapと互換性のあるLoRA管理クラス
    """
    
    def __init__(self):
        # LoRAの状態を保持
        self.lora_state_dict = None
        # 適用スケール
        self.scale = 0.0
        # 適用したレイヤーの記録
        self.applied_layers = set()
        # LoRAの適用有無フラグ
        self.is_active = False
        # 処理統計情報
        self.stats = {
            "start_time": time.time(),
            "load_time": 0,
            "apply_count": 0,
            "modified_layers": 0,
            "errors": [],
            "memory_usage": {}
        }
        
        # デバッグ時のみログ出力
        if DEBUG_ENABLED:
            logger.debug(f"DynamicSwapLoRAManager初期化: FORCE_TRANSFORMER_LORA={FORCE_TRANSFORMER_LORA}, LOW_VRAM_LORA={LOW_VRAM_LORA}")
    
    @timing_decorator
    def load_lora(self, lora_path: str, is_diffusers: bool = False, selective_application: bool = None, pruning: bool = True, pruning_threshold: float = 0.0005, hierarchical_pruning: bool = True, blocks_type: str = "all") -> None:
        """
        LoRAファイルを読み込む
        
        Args:
            lora_path: LoRAファイルへのパス
            is_diffusers: Diffusersフォーマットかどうか
            selective_application: 選択的なLoRA適用を行うかどうか
            pruning: LoRAパラメータのプルーニングを行うかどうか
            pruning_threshold: プルーニングの閾値
            hierarchical_pruning: 階層的プルーニングを使用するか
            blocks_type: 適用するブロックタイプ
        """
        # デフォルト設定の適用
        from .lora_loader import SELECTIVE_APPLICATION_DEFAULT
        if selective_application is None:
            selective_application = SELECTIVE_APPLICATION_DEFAULT
            
        load_start_time = time.time()
        
        # 統計情報を記録
        self.stats["load_params"] = {
            "lora_path": lora_path,
            "is_diffusers": is_diffusers,
            "selective_application": selective_application,
            "pruning": pruning,
            "pruning_threshold": pruning_threshold,
            "hierarchical_pruning": hierarchical_pruning,
            "blocks_type": blocks_type
        }
        
        # 開始ログ
        logger.info(f"LoRAファイルを読み込み中: {lora_path}")
        logger.info(f"  - 選択的適用: {selective_application}")
        logger.info(f"  - プルーニング: {pruning} (閾値: {pruning_threshold})")
        logger.info(f"  - 階層的プルーニング: {hierarchical_pruning}")
        logger.info(f"  - ブロックタイプ: {blocks_type}")
        
        if DEBUG_ENABLED:
            self.log_memory_usage("ロード開始前")
        
        # LoRAの読み込み
        try:
            # パスが存在するか確認
            if not os.path.exists(lora_path):
                error_msg = f"LoRAファイルが見つかりません: {lora_path} (カレントディレクトリ: {os.getcwd()})"
                logger.error(error_msg)
                self.stats["errors"].append({"type": "file_not_found", "message": error_msg})
                raise FileNotFoundError(error_msg)
            
            lora_state_dict = load_lora_weights(lora_path)
            
            # フォーマット変換（必要な場合）
            format_type = detect_lora_format(lora_state_dict)
            logger.info(f"検出されたLoRAフォーマット: {format_type}")

            if format_type == 'diffusers' or is_diffusers:
                lora_state_dict = convert_diffusers_lora_to_hunyuan(lora_state_dict)
            elif format_type == 'musubi':
                lora_state_dict = check_for_musubi(lora_state_dict)
            
            # 統計情報を更新
            orig_params_count = len(lora_state_dict)
            self.stats["original_params_count"] = orig_params_count
            
            # デバッグ時は詳細情報を出力
            if DEBUG_ENABLED:
                tensor_count = sum(1 for v in lora_state_dict.values() if isinstance(v, torch.Tensor))
                total_params = sum(v.numel() for v in lora_state_dict.values() if isinstance(v, torch.Tensor))
                total_memory_mb = sum(v.element_size() * v.numel() for v in lora_state_dict.values() if isinstance(v, torch.Tensor)) / (1024 * 1024)
                lora_down_count = sum(1 for k in lora_state_dict.keys() if '.lora_down' in k)
                lora_up_count = sum(1 for k in lora_state_dict.keys() if '.lora_up' in k)
                
                logger.debug(f"LoRAファイル統計:")
                logger.debug(f"  - テンソル数: {tensor_count}")
                logger.debug(f"  - 総パラメータ数: {total_params:,}")
                logger.debug(f"  - lora_downキー数: {lora_down_count}")
                logger.debug(f"  - lora_upキー数: {lora_up_count}")
                logger.debug(f"  - 総メモリサイズ: {total_memory_mb:.2f}MB")
                
                # キーの分布を分析
                key_patterns = {}
                for key in lora_state_dict.keys():
                    parts = key.split('.')
                    if len(parts) > 2:
                        pattern = '.'.join(parts[:2])  # 最初の2つのセグメントをグループ化
                        if pattern not in key_patterns:
                            key_patterns[pattern] = 0
                        key_patterns[pattern] += 1
                
                # キーの分布を出力
                top_patterns = sorted(key_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.debug(f"  - 主要キーパターン: {top_patterns}")
            
            logger.info(f"元のLoRA: {orig_params_count} パラメータ")
            
            # ブロックタイプに基づくフィルタリング
            if blocks_type != "all" and lora_state_dict:
                try:
                    logger.info(f"ブロックタイプ '{blocks_type}' に基づくフィルタリングを適用中...")
                    filtered_weights = filter_lora_weights_by_block_type(lora_state_dict, blocks_type)
                    # 空でない場合のみ更新
                    if filtered_weights:  
                        lora_state_dict = filtered_weights
                    else:
                        # 空の場合はLoRAを適用しない
                        logger.warning(f"ブロックタイプ '{blocks_type}' によるフィルタリング後にパラメータが見つからないため、LoRAは無効化されます")
                        self.lora_state_dict = None
                        self.is_active = False
                        return
                except Exception as e:
                    logger.warning(f"ブロックフィルタリング中にエラーが発生しました: {e}")
                    logger.warning(traceback.format_exc())
                    logger.info("ブロックフィルタリングをスキップし、標準的な選択的適用を使用します")
                    
                    # エラー発生時は通常の選択的適用にフォールバック
                    if selective_application:
                        try:
                            filtered_weights = filter_lora_weights_by_important_layers(lora_state_dict)
                            if filtered_weights:
                                lora_state_dict = filtered_weights
                            else:
                                logger.warning("選択的適用によりパラメータが見つからないため、LoRAは無効化されます")
                                self.lora_state_dict = None
                                self.is_active = False
                                return
                        except Exception as e:
                            logger.warning(f"選択的適用中にエラーが発生しました: {e}")
                            logger.warning(traceback.format_exc())
            
            # 標準的な選択的適用（ブロックフィルタリングを使用しない場合）
            elif selective_application and blocks_type == "all" and lora_state_dict:
                try:
                    filtered_weights = filter_lora_weights_by_important_layers(lora_state_dict)
                    # 空でない場合のみ更新
                    if filtered_weights:  
                        lora_state_dict = filtered_weights
                    else:
                        # 空の場合はLoRAを適用しない
                        logger.warning("選択的適用によりパラメータが見つからないため、LoRAは無効化されます")
                        self.lora_state_dict = None
                        self.is_active = False
                        return
                except Exception as e:
                    logger.warning(f"選択的適用中にエラーが発生しました: {e}")
                    logger.warning(traceback.format_exc())
                    logger.info("選択的適用をスキップします")
            
            # LoRAパラメータのプルーニング
            if pruning and lora_state_dict:
                try:
                    # メモリ使用状況のログ出力
                    self.log_memory_usage("プルーニング前")
                    
                    # 階層的プルーニングを適用
                    pruned_weights = self.prune_lora_weights(lora_state_dict, pruning_threshold, hierarchical_pruning)
                    if pruned_weights:  # 空でない場合のみ更新
                        lora_state_dict = pruned_weights
                        
                    # メモリ使用状況のログ出力
                    self.log_memory_usage("プルーニング後")
                    
                    # テンソルキャッシュのクリア
                    self.clear_tensor_cache()
                except Exception as prune_error:
                    logger.warning(f"プルーニング中にエラーが発生しました: {prune_error}")
                    logger.warning(traceback.format_exc())
                    logger.info("プルーニングをスキップします")
            
            # 最終的な状態設定
            try:
                if lora_state_dict and len(lora_state_dict) > 0:
                    self.lora_state_dict = lora_state_dict
                    
                    # 統計情報を更新
                    load_time = time.time() - load_start_time
                    self.stats["load_time"] = load_time
                    self.stats["final_params_count"] = len(lora_state_dict)
                    
                    logger.info(f"LoRA読み込み完了: {len(lora_state_dict)} パラメータ (所要時間: {load_time:.2f}秒)")
                    
                    # 適用記録をクリア
                    self.applied_layers.clear()
                    self.is_active = True
                    
                    # デバッグ情報の保存
                    if DEBUG_ENABLED:
                        save_debug_info()
                else:
                    logger.warning("LoRAパラメータが空です。LoRAは有効化されません。")
                    self.lora_state_dict = None
                    self.is_active = False
                    self.stats["errors"].append({"type": "empty_params", "message": "LoRAパラメータが空です"})
            except Exception as e:
                logger.error(f"最終状態設定中にエラーが発生しました: {e}")
                logger.error(traceback.format_exc())
                self.lora_state_dict = None
                self.is_active = False
            
        except Exception as e:
            logger.error(f"LoRA読み込みエラー: {e}")
            logger.error(traceback.format_exc())
            self.lora_state_dict = None
            self.is_active = False
            raise
    
    @timing_decorator
    def set_scale(self, scale: float) -> None:
        """
        LoRA適用スケールを設定
        
        Args:
            scale: 適用スケール (0.0-1.0)
        """
        self.scale = max(0.0, min(1.0, scale))
        logger.info(f"LoRA適用スケールを設定: {self.scale}")
    
    @timing_decorator
    def apply_to_layer(self, layer_name: str, layer: torch.nn.Module) -> torch.nn.Module:
        """
        単一レイヤーにLoRAを適用
        
        Args:
            layer_name: レイヤー名
            layer: レイヤーモジュール
            
        Returns:
            torch.nn.Module: LoRAが適用されたレイヤー
        """
        if not self.is_active or self.lora_state_dict is None:
            return layer
        
        if layer_name in self.applied_layers:
            logger.debug(f"レイヤーは既にLoRA適用済み: {layer_name}")
            return layer
        
        try:
            with torch.no_grad():
                # レイヤーのパラメータを取得
                for param_name, param in layer.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    
                    # メモリ使用状況をログ出力（最初のパラメータのみ）
                    if param_name == next(iter(layer.named_parameters()))[0]:
                        self.log_memory_usage(f"レイヤー適用前({layer_name})")
                        
                    # LoRAに対応するキーを探す
                    lora_down_key = f"{full_param_name}.lora_down"
                    lora_up_key = f"{full_param_name}.lora_up"
                    
                    if lora_down_key in self.lora_state_dict and lora_up_key in self.lora_state_dict:
                        # LoRAの重みを取得
                        lora_down = self.lora_state_dict[lora_down_key].to(param.device)
                        lora_up = self.lora_state_dict[lora_up_key].to(param.device)
                        
                        # LoRAの演算
                        delta = torch.matmul(lora_up, lora_down) * self.scale
                        
                        # 形状を確認して調整
                        if delta.shape != param.shape:
                            logger.warning(f"形状不一致: {delta.shape} != {param.shape}, スキップ: {full_param_name}")
                            continue
                        
                        # デバッグ時は詳細情報を記録
                        if DEBUG_ENABLED:
                            before_data = param.data.clone()
                            
                        # 元のパラメータに適用
                        param.data += delta
                        
                        # 統計情報を更新
                        self.stats["apply_count"] = self.stats.get("apply_count", 0) + 1
                        
                        if DEBUG_ENABLED:
                            # 適用前後の変化を計算
                            after_data = param.data
                            param_norm = torch.norm(before_data)
                            delta_norm = torch.norm(delta)
                            change_ratio = (delta_norm / param_norm * 100) if param_norm > 0 else 0
                            
                            # 最初の数個だけ詳細ログを出力
                            if self.stats["apply_count"] <= 5:
                                logger.debug(f"LoRA適用 [{full_param_name}]: 変化率={change_ratio:.4f}%, 元ノルム={param_norm:.6f}, デルタノルム={delta_norm:.6f}")
                        
                        logger.debug(f"LoRA適用: {full_param_name}")
            
            # メモリ使用状況をログ出力（最後のパラメータのみ）
            if list(layer.named_parameters()) and param_name == list(layer.named_parameters())[-1][0]:
                self.log_memory_usage(f"レイヤー適用後({layer_name})")
                
            # 適用記録を更新
            self.applied_layers.add(layer_name)
            return layer
            
        except Exception as e:
            logger.error(f"レイヤーへのLoRA適用エラー: {layer_name}, {e}")
            logger.error(traceback.format_exc())
            return layer
    
    @timing_decorator
    def remove_from_layer(self, layer_name: str, layer: torch.nn.Module, original_state: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """
        レイヤーからLoRAの効果を削除（元の状態に戻す）
        
        Args:
            layer_name: レイヤー名
            layer: レイヤーモジュール
            original_state: 元のパラメータ状態
            
        Returns:
            torch.nn.Module: 元の状態に戻されたレイヤー
        """
        if layer_name not in self.applied_layers:
            return layer
        
        try:
            with torch.no_grad():
                # レイヤーのパラメータを元に戻す
                for param_name, param in layer.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    
                    if full_param_name in original_state:
                        # 保存された元の状態がCPUにある場合は、デバイスを合わせる
                        state_tensor = original_state[full_param_name]
                        if state_tensor.device != param.device:
                            state_tensor = state_tensor.to(param.device)
                        
                        # 現在のパラメータを元の状態に戻す
                        param.data.copy_(state_tensor)
                        logger.debug(f"LoRA効果を削除: {full_param_name}")
            
            # 適用記録を更新
            self.applied_layers.remove(layer_name)
            return layer
            
        except Exception as e:
            logger.error(f"レイヤーからのLoRA削除エラー: {layer_name}, {e}")
            logger.error(traceback.format_exc())
            return layer

    @timing_decorator
    def prune_lora_weights(self, lora_weights: Dict[str, torch.Tensor], threshold: float = 0.005, use_hierarchical: bool = True) -> Dict[str, torch.Tensor]:
        """
        閾値以下の小さなLoRAパラメータを0に設定
        
        Args:
            lora_weights: プルーニング前のLoRA重み
            threshold: プルーニング閾値（絶対値）
            use_hierarchical: 階層的プルーニングを使用するか
            
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
        # デバッグ用の統計情報
        if DEBUG_ENABLED:
            tensor_stats_before = {}
            for key, value in lora_weights.items():
                if isinstance(value, torch.Tensor) and len(tensor_stats_before) < 10:  # 最初の10個だけ記録
                    tensor_stats_before[key] = collect_tensor_stats(value, f"prune_before[{key}]")
        
        # 階層的プルーニングのための層ごとの閾値設定
        if use_hierarchical:
            # 特に重要な層（視覚効果に大きく影響）は低い閾値を使用
            critical_layers = [
                "transformer_blocks", "single_transformer_blocks", 
                "norm_out", "proj_out", "to_q", "to_k", "to_v", "to_out"
            ]
            # 中程度の重要性の層
            important_layers = ["attn", "norm", "rgb_linear", "conv", "input_blocks", "output_blocks", "single_blocks", "double_blocks"]
            # その他の層（高い閾値＝より積極的な削減）
            
            logger.info(f"階層的プルーニングを使用: 重要度に応じて異なる閾値を適用")
            
            # デバッグ時は設定される閾値を記録
            if DEBUG_ENABLED:
                thresholds = {
                    "critical": 0.00025,
                    "important": 0.0005,
                    "normal": 0.001,
                    "default": threshold
                }
                logger.debug(f"階層的閾値設定: {thresholds}")
        
        for key, value in lora_weights.items():
            if isinstance(value, torch.Tensor):
                total_params += value.numel()
                
                # 階層的プルーニングの場合、層ごとに異なる閾値を適用
                current_threshold = threshold
                if use_hierarchical:
                    if any(layer in key for layer in critical_layers):
                        # 最重要な層は最も低い閾値を使用（保持率高）
                        current_threshold = 0.00025    # 最重要層（固定値）
                    elif any(layer in key for layer in important_layers):
                        # 重要な層もやや低い閾値を使用
                        current_threshold = 0.0005    # 重要層（固定値）
                    else:
                        # その他の層は標準閾値
                        current_threshold = 0.001     # その他の層（固定値）
                
                # 閾値以下の値をゼロにマスク
                mask = torch.abs(value) >= current_threshold
                pruned_tensor = value * mask
                pruned_params += (value.numel() - mask.sum().item())
                pruned_weights[key] = pruned_tensor
            else:
                pruned_weights[key] = value
        
        # ゼロ除算を確実に回避
        if total_params > 0:  # 0除算を避ける
            prune_percent = (pruned_params / total_params) * 100.0
            logger.info(f"プルーニング: {pruned_params}/{total_params} パラメータ削減 ({prune_percent:.2f}%)")
            
            # メモリ削減量の計算と出力
            tensor_size = 4  # float32 = 4 bytes
            memory_saved_mb = (pruned_params * tensor_size) / (1024 * 1024)  # MB単位
            logger.info(f"推定メモリ削減量: {memory_saved_mb:.2f}MB")
            
            # デバッグ統計情報を更新
            _debug_data["lora_stats"]["pruning"] = {
                "total_params": total_params,
                "pruned_params": pruned_params,
                "prune_percent": prune_percent,
                "memory_saved_mb": memory_saved_mb,
                "threshold": threshold,
                "hierarchical": use_hierarchical
            }
            
            # デバッグ時はプルーニング後のテンソル統計も記録
            if DEBUG_ENABLED:
                tensor_stats_after = {}
                for key, value in pruned_weights.items():
                    if key in tensor_stats_before and isinstance(value, torch.Tensor):
                        tensor_stats_after[key] = collect_tensor_stats(value, f"prune_after[{key}]")
                        
                        # プルーニングの影響をログに出力
                        before = tensor_stats_before[key]
                        after = tensor_stats_after[key]
                        if before and after and "non_zero" in before and "non_zero" in after:
                            reduction = 100 - (after["non_zero"] / before["non_zero"] * 100) if before["non_zero"] > 0 else 0
                            logger.debug(f"プルーニング結果 [{key}]: 非ゼロ要素 {after['non_zero']}/{after['total']} ({after['non_zero']/after['total']*100:.2f}%), 削減率 {reduction:.2f}%")
        else:
            logger.warning("プルーニング対象のパラメータが0です。プルーニングをスキップします。")
            
            # エラー情報を記録
            _debug_data["errors"].append({
                "type": "pruning_error",
                "message": "プルーニング対象パラメータが0"
            })
        
        return pruned_weights

    @timing_decorator
    def log_memory_usage(self, message: str) -> None:
        """
        現在のGPUメモリ使用状況をログに出力
        
        Args:
            message: ログメッセージのプレフィックス
        """
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                
                # 統計情報を記録
                self.stats["memory_usage"][message] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "max_allocated_gb": max_allocated,
                    "timestamp": time.time()
                }
                
                # デバッグ用グローバル変数にも記録
                _debug_data["memory_usage"][message] = self.stats["memory_usage"][message].copy()
                
                # デバッグ時は詳細情報を出力
                if DEBUG_ENABLED:
                    # 各デバイスごとのメモリ使用状況
                    device_count = torch.cuda.device_count()
                    for i in range(device_count):
                        dev_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        dev_reserved = torch.cuda.memory_reserved(i) / 1024**3
                        logger.debug(f"  CUDA:{i} - 使用: {dev_allocated:.2f}GB, 予約: {dev_reserved:.2f}GB")
                
                logger.info(f"{message}: GPU使用メモリ {allocated:.2f}GB (予約: {reserved:.2f}GB, 最大: {max_allocated:.2f}GB)")
            except Exception as e:
                logger.warning(f"メモリ使用状況の取得に失敗: {e}")
                logger.warning(traceback.format_exc())
                
                # エラー情報を記録
                self.stats["errors"].append({
                    "type": "memory_error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                })
    
    @timing_decorator
    def clear_tensor_cache(self) -> None:
        """
        Pytorchのテンソルキャッシュをクリア
        """
        if torch.cuda.is_available():
            try:
                # クリア前のメモリ使用量を記録
                before_allocated = torch.cuda.memory_allocated() / 1024**3
                before_reserved = torch.cuda.memory_reserved() / 1024**3
                
                # GPUキャッシュのクリア
                torch.cuda.empty_cache()
                
                # クリア後のメモリ使用量を記録
                after_allocated = torch.cuda.memory_allocated() / 1024**3
                after_reserved = torch.cuda.memory_reserved() / 1024**3
                
                # 統計情報を記録
                cache_clear_stats = {
                    "before_allocated_gb": before_allocated,
                    "before_reserved_gb": before_reserved,
                    "after_allocated_gb": after_allocated,
                    "after_reserved_gb": after_reserved,
                    "freed_allocated_gb": before_allocated - after_allocated,
                    "freed_reserved_gb": before_reserved - after_reserved,
                    "timestamp": time.time()
                }
                
                # デバッグ用グローバル変数に記録
                _debug_data["memory_usage"]["cache_clear"] = cache_clear_stats
                
                freed_mb = (before_reserved - after_reserved) * 1024
                
                logger.info(f"GPUテンソルキャッシュをクリアしました (解放: {freed_mb:.2f}MB)")
                
                # デバッグ時は詳細情報を出力
                if DEBUG_ENABLED:
                    logger.debug(f"  キャッシュクリア前: 割り当て={before_allocated:.2f}GB, 予約={before_reserved:.2f}GB")
                    logger.debug(f"  キャッシュクリア後: 割り当て={after_allocated:.2f}GB, 予約={after_reserved:.2f}GB")
                    
            except Exception as e:
                logger.warning(f"キャッシュクリア中にエラー: {e}")
                logger.warning(traceback.format_exc())
                
                # エラー情報を記録
                _debug_data["errors"].append({
                    "type": "cache_clear_error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                })
    
    @timing_decorator
    def reset(self) -> None:
        """
        LoRA状態をリセット
        """
        # 現在の状態をログ出力
        if DEBUG_ENABLED:
            logger.debug(f"LoRA状態リセット前: is_active={self.is_active}, 適用レイヤー数={len(self.applied_layers)}")
        
        self.lora_state_dict = None
        self.scale = 0.0
        self.applied_layers.clear()
        self.is_active = False
        
        # 統計リセット
        self.stats["end_time"] = time.time()
        self.stats["total_time"] = self.stats["end_time"] - self.stats["start_time"]
        
        # デバッグ統計情報の保存
        if DEBUG_ENABLED:
            save_debug_info()
        
        # キャッシュもクリア
        self.clear_tensor_cache()
        logger.info(f"LoRA状態をリセットしました (総活動時間: {self.stats['total_time']:.2f}秒)")

    @timing_decorator
    def install_hooks(self, model: torch.nn.Module) -> None:
        """
        DynamicSwapフックをインストールして動的にLoRAを適用
        
        Args:
            model: DynamicSwapが設定されたモデル
        """
        if not self.is_active or self.lora_state_dict is None:
            logger.info("LoRAが読み込まれていないか非アクティブです。フック設定をスキップします。")
            return
        
        # モデルの元の状態を保存（最適化版）
        original_states = {}
        
        def collect_minimal_original_states(module, module_name):
            # LoRAで変更される予定のパラメータのみを保存
            lora_keys = set(self.lora_state_dict.keys())
            for param_name, param in module.named_parameters():
                full_name = f"{module_name}.{param_name}"
                # このパラメータにLoRAが適用されるかチェック
                lora_down_key = f"{full_name}.lora_down"
                lora_up_key = f"{full_name}.lora_up"
                if lora_down_key in lora_keys and lora_up_key in lora_keys:
                    # CPUメモリに保存してGPUメモリを節約
                    original_states[full_name] = param.data.detach().cpu().clone()
        
        # 元の状態を収集（最適化版）
        saved_modules = 0
        for name, module in model.named_modules():
            if len(list(module.parameters(recurse=False))) > 0:
                collect_minimal_original_states(module, name)
                saved_modules += 1
        
        logger.info(f"最適化済み状態保存: {len(original_states)} パラメータ（{saved_modules}モジュール中）")
        
        # 読み込み前フックを定義（LoRA適用）
        def pre_load_hook(module, hook_name):
            if not self.is_active:
                return
            
            logger.debug(f"読み込み前フック呼び出し: {hook_name}")
            self.apply_to_layer(hook_name, module)
        
        # アンロード後フックを定義（元の状態に戻す）
        def post_unload_hook(module, hook_name):
            if not self.is_active:
                return
            
            if hook_name in self.applied_layers:
                logger.debug(f"アンロード後フック呼び出し: {hook_name}")
                self.remove_from_layer(hook_name, module, original_states)
        
        # DynamicSwapの内部状態を取得
        # 注意: 実際の実装ではDynamicSwapの構造に応じた調整が必要
        for name, module in model.named_modules():
            if hasattr(module, '_hook_register_handle_pre_load'):
                # 既存のフックに加えて、LoRAフックを登録
                old_pre_load = module._hook_register_handle_pre_load
                
                def combined_pre_load(mod):
                    old_pre_load(mod)
                    pre_load_hook(mod, name)
                
                module._hook_register_handle_pre_load = combined_pre_load
            
            if hasattr(module, '_hook_register_handle_post_unload'):
                # 既存のフックに加えて、LoRAフックを登録
                old_post_unload = module._hook_register_handle_post_unload
                
                def combined_post_unload(mod):
                    post_unload_hook(mod, name)
                    old_post_unload(mod)
                
                module._hook_register_handle_post_unload = combined_post_unload
        
        # 統計情報を更新
        self.stats["hook_installation"] = {
            "saved_modules": saved_modules,
            "original_states": len(original_states),
            "timestamp": time.time()
        }
        
        logger.info(f"DynamicSwapフックを設定しました (保存モジュール数: {saved_modules}, パラメータ数: {len(original_states)})")
