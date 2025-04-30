"""
FramePack-eichi deepcopy代替モジュール

このモジュールは、トランスフォーマーモデルのコピー操作を最適化し、
メモリ使用量を削減するための代替手法を提供します。
標準のPython copy.deepcopy関数の代わりに使用します。

使用例:
    # 元のコード：
    # import copy
    # transformer_lora = copy.deepcopy(transformer)
    
    # 最適化バージョン：
    from lora_utils.deep_copy_alternatives import replace_deepcopy
    transformer_lora = replace_deepcopy(transformer, high_vram=high_vram)
"""

import torch
import gc
import warnings
import os
import psutil
import traceback
from typing import Dict, Any, Optional, Union, List

def get_system_memory_info():
    """
    システムのメモリ情報を取得します。
    
    Returns:
        dict: RAMとVRAMの情報を含む辞書
    """
    # RAMメモリ使用量
    try:
        import psutil
        process = psutil.Process()
        ram_usage_bytes = process.memory_info().rss
        ram_usage_gb = ram_usage_bytes / (1024 ** 3)
        ram_percent = psutil.virtual_memory().percent
    except ImportError:
        ram_usage_gb = -1
        ram_percent = -1
    
    # VRAMメモリ使用量
    if torch.cuda.is_available():
        vram_usage_bytes = torch.cuda.memory_allocated()
        vram_usage_gb = vram_usage_bytes / (1024 ** 3)
        
        # デバイスの総メモリを取得
        device_props = torch.cuda.get_device_properties(0)
        total_memory_bytes = device_props.total_memory
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        
        vram_percent = (vram_usage_bytes / total_memory_bytes) * 100
    else:
        vram_usage_gb = 0
        total_memory_gb = 0
        vram_percent = 0
    
    return {
        "ram": {
            "usage_gb": ram_usage_gb,
            "percent": ram_percent
        },
        "vram": {
            "usage_gb": vram_usage_gb,
            "total_gb": total_memory_gb,
            "percent": vram_percent
        }
    }

def log_memory_usage(prefix=""):
    """
    現在のメモリ使用状況をログに出力します。
    
    Args:
        prefix: ログメッセージの接頭辞
    """
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"{prefix} VRAM: 割当={vram_allocated:.2f}GB, 予約={vram_reserved:.2f}GB")

def log_memory_detailed(message="", model=None, lora_path=None):
    """
    詳細なメモリ使用状況をログに出力する関数
    
    Args:
        message: ログメッセージ
        model: メモリ使用量を計測するモデル（オプション）
        lora_path: LoRAファイルパス（オプション）
        
    Returns:
        dict: メモリ使用状況の詳細情報
    """
    # RAMメモリ使用量の取得
    process = psutil.Process(os.getpid())
    ram_usage_bytes = process.memory_info().rss
    ram_usage_gb = ram_usage_bytes / (1024 ** 3)
    ram_percent = psutil.virtual_memory().percent
    
    # VRAMメモリ使用量の取得
    vram_info = {}
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        vram_max = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        vram_percent = (vram_allocated / vram_max) * 100
        vram_info = {
            "allocated_gb": vram_allocated,
            "reserved_gb": vram_reserved,
            "total_gb": vram_max,
            "percent": vram_percent
        }
    
    # モデルパラメータ数（もし提供されていれば）
    params_info = {}
    if model is not None:
        params_count = sum(p.numel() for p in model.parameters())
        params_gb = params_count * 4 / (1024 ** 3)  # float32想定
        params_info = {
            "count": params_count,
            "millions": params_count / 1000000,
            "size_gb": params_gb
        }
    
    # LoRAファイルサイズ（もし提供されていれば）
    lora_info = {}
    if lora_path is not None and os.path.exists(lora_path):
        lora_size_bytes = os.path.getsize(lora_path)
        lora_size_mb = lora_size_bytes / (1024 ** 2)
        lora_info = {
            "file_path": lora_path,
            "file_size_mb": lora_size_mb
        }
    
    # 詳細なログ出力
    log_lines = [
        f"===== メモリ詳細レポート: {message} =====",
        f"RAM: {ram_usage_gb:.2f}GB ({ram_percent:.1f}%)"
    ]
    
    if vram_info:
        log_lines.append(f"VRAM: 割当={vram_info['allocated_gb']:.2f}GB, "
                         f"予約={vram_info['reserved_gb']:.2f}GB, "
                         f"合計={vram_info['total_gb']:.2f}GB ({vram_info['percent']:.1f}%)")
    
    if params_info:
        log_lines.append(f"モデルパラメータ: {params_info['millions']:.2f}M パラメータ "
                         f"({params_info['size_gb']:.2f}GB 相当)")
    
    if lora_info:
        log_lines.append(f"LoRAファイル: {os.path.basename(lora_info['file_path'])}, "
                         f"サイズ: {lora_info['file_size_mb']:.2f}MB")
    
    log_lines.append("=" * 50)
    print("\n".join(log_lines))
    
    return {
        "ram": {"usage_gb": ram_usage_gb, "percent": ram_percent},
        "vram": vram_info,
        "params": params_info,
        "lora": lora_info
    }

def analyze_model_memory(model, name="model"):
    """
    モデルの階層別メモリ使用量を分析
    
    Args:
        model: 分析対象のモデル
        name: モデル名（ログ出力用）
        
    Returns:
        dict: 階層別メモリ使用量情報
    """
    results = {}
    total_params = 0
    total_size_mb = 0
    
    for module_name, module in model.named_children():
        # パラメータが一つもないモジュールをスキップ
        if not any(True for _ in module.parameters()):
            continue
            
        params = sum(p.numel() for p in module.parameters())
        size_mb = params * 4 / (1024 * 1024)  # float32想定
        
        # デバイス情報の取得（パラメータがある場合のみ）
        device = "unknown"
        for p in module.parameters():
            device = p.device
            break
            
        results[f"{name}.{module_name}"] = {
            "params": params,
            "size_mb": size_mb,
            "device": device
        }
        total_params += params
        total_size_mb += size_mb
        
        # 再帰的に子モジュールも分析
        if list(module.children()):
            sub_results = analyze_model_memory(module, f"{name}.{module_name}")
            results.update(sub_results)
    
    if name == "model":
        print(f"総パラメータ数: {total_params:,} ({total_size_mb:.2f}MB)")
        
    return results

def detect_lora_changes(model_before, model_after):
    """
    LoRA適用によるパラメータの変化を検出
    
    Args:
        model_before: LoRA適用前のモデル
        model_after: LoRA適用後のモデル
        
    Returns:
        list: 変更されたパラメータの情報
    """
    changes = []
    
    # 両方のモデルから状態辞書を取得
    state_before = model_before.state_dict()
    state_after = model_after.state_dict()
    
    for key in state_before:
        if key in state_after:
            # テンソルが変更されたかチェック
            before_tensor = state_before[key]
            after_tensor = state_after[key]
            
            # 形状が同じ場合のみ比較
            if before_tensor.shape == after_tensor.shape:
                # 同じデバイスに移動して比較
                if before_tensor.device != after_tensor.device:
                    before_tensor = before_tensor.to(after_tensor.device)
                
                # 変更がある場合のみ記録
                if not torch.allclose(before_tensor, after_tensor, rtol=1e-5, atol=1e-5):
                    diff = (after_tensor - before_tensor)
                    changes.append({
                        "key": key,
                        "max_diff": diff.abs().max().item(),
                        "mean_diff": diff.abs().mean().item(),
                        "shape": list(before_tensor.shape),
                        "size_mb": before_tensor.numel() * 4 / (1024 * 1024)
                    })
    
    # 変更の多い順にソート
    changes.sort(key=lambda x: x["max_diff"], reverse=True)
    
    # 結果出力
    print(f"検出された変更: {len(changes)} パラメータ")
    for i, change in enumerate(changes[:10]):  # 上位10件のみ表示
        print(f"{i+1}. {change['key']}: 最大差={change['max_diff']:.6f}, "
              f"平均差={change['mean_diff']:.6f}, サイズ={change['size_mb']:.2f}MB")
              
    return changes

def create_efficient_model_copy(model, selected_modules=None):
    """
    transformerモデルの効率的なコピーを作成します。
    完全なdeepcopyの代わりに、必要な重みのみをコピーします。
    コピー時にデバイスの一貫性を確保します。
    
    Args:
        model: コピー元のtransformerモデル
        selected_modules: コピーする特定のモジュール名のリスト（Noneの場合はすべてコピー）
    
    Returns:
        コピーされたモデル（オリジナルと同じ構造だが一部の重みのみコピー）
    """
    # モデルの初期化パラメータを取得
    # モデルクラスがget_configメソッドを持つ場合は、それを使用
    if hasattr(model, 'get_config') and callable(getattr(model, 'get_config')):
        config = model.get_config()
        if hasattr(model.__class__, 'from_config') and callable(getattr(model.__class__, 'from_config')):
            # モデルがfrom_configメソッドを持つ場合は、それを使用
            new_model = model.__class__.from_config(config)
        else:
            # from_configメソッドがない場合は、コンフィグを引数として直接渡す
            new_model = model.__class__(**config)
    else:
        # トランスフォーマーモデルの一般的な属性を取得してみる
        config = {}
        for attr_name in ['dim', 'hidden_size', 'size', 'width', 'num_layers', 'n_layers']:
            if hasattr(model, attr_name):
                config[attr_name] = getattr(model, attr_name)
        
        # 特殊な属性を特別にチェック
        if hasattr(model, 'config'):
            # Hugging Faceモデル等では、configオブジェクトが存在する場合がある
            try:
                # 同じデバイスでモデルを作成
                device = next(model.parameters()).device
                
                # FrozenDict対応：config が to_dict メソッドを持っている場合は通常の辞書に変換
                if hasattr(model.config, 'to_dict'):
                    config_dict = model.config.to_dict()
                    # コンフィグを使用して新しいモデルを作成
                    new_model = model.__class__(config_dict).to(device)
                else:
                    # 通常のコンフィグを使用
                    new_model = model.__class__(model.config).to(device)
                    
                print("configオブジェクトを使用してモデルを作成")
                # 状態辞書をコピーして戻る
                # モデルの状態辞書を取得
                original_state_dict = model.state_dict()
                # 新しい状態辞書を作成
                new_state_dict = {}
                # すべてのパラメータを選択的にコピー
                if selected_modules is not None:
                    for key in original_state_dict:
                        if any(module_name in key for module_name in selected_modules):
                            new_state_dict[key] = original_state_dict[key].clone()
                        else:
                            new_state_dict[key] = original_state_dict[key]
                else:
                    for key, value in original_state_dict.items():
                        if isinstance(value, torch.Tensor):
                            new_state_dict[key] = value.clone()
                        else:
                            new_state_dict[key] = value
                # 状態辞書をロード
                new_model.load_state_dict(new_state_dict)
                return new_model
            except Exception as e:
                print(f"configを使用したモデル作成に失敗: {e}")
                traceback.print_exc()
        
        # デフォルト値で新しいモデルを作成
        try:
            # 取得した属性でモデルを初期化
            if config:
                new_model = model.__class__(**config)
                print(f"取得したパラメータでモデルを作成: {config}")
            else:
                # パラメータが取得できない場合は、デフォルトの初期化を使用
                new_model = model.__class__()
                print("デフォルトパラメータでモデルを作成。状態辞書の不一致が発生する可能性があります")
        except Exception as e:
            # モデル作成が失敗した場合、通常のdeepcopyにフォールバック
            import copy
            print(f"モデルの初期化に失敗したため、deepcopyを使用します: {e}")
            return copy.deepcopy(model)
    
    # 元のモデルのデバイスを取得
    device = next(model.parameters()).device
    print(f"元のモデルのデバイス: {device}")
    
    # モデルの状態辞書を取得
    original_state_dict = model.state_dict()
    
    # 新しい状態辞書を初期化
    new_state_dict = {}
    
    # 選択的にモジュールをコピー
    if selected_modules is not None:
        for key in original_state_dict:
            # 指定されたモジュールに関連するパラメータのみをコピー
            if any(module_name in key for module_name in selected_modules):
                if isinstance(original_state_dict[key], torch.Tensor):
                    # デバイスを維持しながらクローン
                    new_state_dict[key] = original_state_dict[key].clone().to(device)
                else:
                    new_state_dict[key] = original_state_dict[key]
            else:
                # それ以外は参照をそのまま使用（メモリ節約）
                new_state_dict[key] = original_state_dict[key]
    else:
        # すべてのパラメータをコピー（それでもdeepcopyより効率的）
        for key, value in original_state_dict.items():
            if isinstance(value, torch.Tensor):
                # テンソルの場合はcloneを使用し、同じデバイスに配置
                new_state_dict[key] = value.clone().to(device)
            else:
                # その他の型はそのままコピー
                new_state_dict[key] = value
    
    try:
        # 新しいモデルをデバイスに移動
        new_model = new_model.to(device)
        
        # 新しいモデルに状態辞書をロード
        new_model.load_state_dict(new_state_dict)
        
        # デバイスの一貫性を再確認
        for param in new_model.parameters():
            if param.device != device:
                print(f"警告: パラメータがデバイス {param.device} に存在しますが、期待されるデバイスは {device} です。修正します。")
                param.data = param.data.to(device)
                
    except Exception as e:
        import copy
        print(f"状態辞書のロードに失敗したため、deepcopyを使用します: {e}")
        return copy.deepcopy(model)
    
    print(f"モデルが正常にコピーされ、すべてのパラメータが {device} に配置されました")
    return new_model

def create_memory_efficient_copy(model):
    """
    メモリ効率を最大限に高めたモデルコピー手法です。
    LoRA適用時に必要な重み（線形レイヤーの重みのみ）だけをコピーします。
    コピー時にデバイスの一貫性を確保します。
    
    Args:
        model: コピー元のtransformerモデル
    
    Returns:
        コピーされたモデル（LoRA適用に必要な重みのみコピー）
    """
    # LoRA適用に必要なモジュールキーワードを定義
    lora_relevant_keywords = [
        'attn', 'mlp', 'linear', 'proj', 'norm', 'transformer_blocks', 
        'single_transformer_blocks'
    ]
    
    # 元のモデルのデバイスを取得
    device = next(model.parameters()).device
    print(f"メモリ効率モード: 元のモデルのデバイス: {device}")
    
    # create_efficient_model_copyを利用してコピーを作成
    try:
        # モデルの初期化とコピーは、create_efficient_model_copyと同じ機能を使用
        if hasattr(model, 'get_config') and callable(getattr(model, 'get_config')):
            config = model.get_config()
            if hasattr(model.__class__, 'from_config') and callable(getattr(model.__class__, 'from_config')):
                new_model = model.__class__.from_config(config)
            else:
                new_model = model.__class__(**config)
        else:
            # トランスフォーマーモデルの一般的な属性を取得
            config = {}
            for attr_name in ['dim', 'hidden_size', 'size', 'width', 'num_layers', 'n_layers']:
                if hasattr(model, attr_name):
                    config[attr_name] = getattr(model, attr_name)
            
            # 特殊な属性を特別にチェック
            if hasattr(model, 'config'):
                try:
                    # FrozenDict対応：config が to_dict メソッドを持っている場合は通常の辞書に変換
                    if hasattr(model.config, 'to_dict'):
                        config_dict = model.config.to_dict()
                        # コンフィグを使用して新しいモデルを作成
                        new_model = model.__class__(config_dict).to(device)
                    else:
                        # 通常のコンフィグを使用
                        new_model = model.__class__(model.config).to(device)
                        
                    print("メモリ効率モード: configオブジェクトを使用してモデルを作成")
                    
                    # LoRA関連の重みだけをコピー
                    original_state_dict = model.state_dict()
                    new_state_dict = {}
                    
                    for key, value in original_state_dict.items():
                        if any(keyword in key for keyword in lora_relevant_keywords) and key.endswith('.weight'):
                            # 重みテンソルをクローンし、正しいデバイスに移動
                            new_state_dict[key] = value.clone().to(device)
                        else:
                            # その他の値は参照を保持
                            new_state_dict[key] = value
                    
                    new_model.load_state_dict(new_state_dict)
                    return new_model
                except Exception as e:
                    print(f"メモリ効率モード: configを使用したモデル作成に失敗: {e}")
                    traceback.print_exc()
            
            # 取得した属性でモデルを初期化
            if config:
                new_model = model.__class__(**config)
                print(f"メモリ効率モード: 取得したパラメータでモデルを作成: {config}")
            else:
                # パラメータが取得できない場合は、デフォルトの初期化を使用
                new_model = model.__class__()
                print("メモリ効率モード: デフォルトパラメータでモデルを作成")
    except Exception as e:
        # モデル作成が失敗した場合、通常のdeepcopyにフォールバック
        import copy
        print(f"メモリ効率モード: モデルの初期化に失敗したため、deepcopyを使用します: {e}")
        return copy.deepcopy(model)
    
    # モデルの状態辞書を取得
    original_state_dict = model.state_dict()
    
    # 新しい状態辞書を初期化
    new_state_dict = {}
    
    # LoRA適用に必要な重みのみをコピー
    for key, value in original_state_dict.items():
        # LoRA関連の重みのみをコピー
        if any(keyword in key for keyword in lora_relevant_keywords) and key.endswith('.weight'):
            # デバイスの一貫性を確保しながらクローン
            new_state_dict[key] = value.clone().to(device)
        else:
            # その他のパラメータは参照をそのまま使用
            new_state_dict[key] = value
    
    try:
        # 新しいモデルを正しいデバイスに移動
        new_model = new_model.to(device)
        
        # 新しいモデルに状態辞書をロード
        new_model.load_state_dict(new_state_dict)
        
        # デバイスの一貫性を再確認
        for param in new_model.parameters():
            if param.device != device:
                print(f"メモリ効率モード: 警告 - パラメータがデバイス {param.device} に存在しますが、期待されるデバイスは {device} です。修正します。")
                param.data = param.data.to(device)
    except Exception as e:
        import copy
        print(f"メモリ効率モード: 状態辞書のロードに失敗したため、deepcopyを使用します: {e}")
        return copy.deepcopy(model)
    
    print(f"メモリ効率モード: モデルが正常にコピーされ、すべての重要パラメータが {device} に配置されました")
    return new_model

def create_shared_copy_with_backup(model):
    """
    オリジナルの重みを保存し、必要に応じて復元できるようにする手法です。
    deepcopyを使用せず、オリジナルのモデルに直接変更を加えます。
    
    Args:
        model: 変更を加えるモデル
    
    Returns:
        (model, restore_func) - モデル自体と、元の状態に戻すための関数
    """
    print("共有コピーモード: 元のモデルに直接変更を加え、復元関数を作成します")
    
    # LoRA適用に必要な重みのみをバックアップしてメモリを節約
    lora_relevant_keywords = [
        'attn', 'mlp', 'linear', 'proj', 'norm', 'transformer_blocks', 
        'single_transformer_blocks'
    ]
    
    # 変更される可能性のあるパラメータをバックアップ
    backup = {}
    backed_up_size = 0  # バックアップした重みの総量を記録
    
    try:
        for name, param in model.named_parameters():
            if 'weight' in name and any(keyword in name for keyword in lora_relevant_keywords):
                # メモリ使用量を節約するため、CPUメモリにバックアップ
                backup[name] = param.clone().detach().cpu()
                backed_up_size += param.numel() * param.element_size()
        
        backed_up_size_mb = backed_up_size / (1024 * 1024)
        print(f"共有コピーモード: {len(backup)}個のパラメータ ({backed_up_size_mb:.2f} MB) をバックアップしました")
        
        # バックアップが全くない場合は、modelの構造が期待と異なる
        if len(backup) == 0:
            print("共有コピーモード: 警告 - バックアップするパラメータが見つかりませんでした。モデル構造が予期と異なる可能性があります")
            # 代替手段として、モデル全体の状態辞書をバックアップ
            backup = {"_full_state_dict": {k: v.clone().detach().cpu() if isinstance(v, torch.Tensor) else v 
                                          for k, v in model.state_dict().items()}}
            print("共有コピーモード: 完全状態辞書バックアップに切り替えました")
    except Exception as e:
        print(f"共有コピーモード: バックアップ作成中にエラーが発生しました: {e}")
        # エラーが発生した場合、deepcopyにフォールバック
        import copy
        copied_model = copy.deepcopy(model)
        return copied_model, lambda: None  # 空の復元関数
    
    # 復元関数を定義
    def restore_original_weights():
        try:
            # 完全状態辞書バックアップの場合
            if "_full_state_dict" in backup:
                print("共有コピーモード: 完全状態辞書からモデルを復元します")
                model.load_state_dict(backup["_full_state_dict"])
            else:
                # 個別パラメータの復元
                for name, original_param in backup.items():
                    # パラメータ名からモジュールとパラメータ名を取得
                    module_path, param_name = name.rsplit('.', 1)
                    module = model
                    for part in module_path.split('.'):
                        if hasattr(module, part):
                            module = getattr(module, part)
                        else:
                            print(f"共有コピーモード: 警告 - モジュール {part} がパス {module_path} に存在しません")
                            break
                    
                    # 元の重みを復元
                    if hasattr(module, param_name):
                        # CPUから元のデバイスに移動して復元
                        original_device = getattr(module, param_name).device
                        getattr(module, param_name).data.copy_(original_param.to(original_device))
            
            # 復元完了メッセージ
            print("共有コピーモード: モデルの重みを元の状態に復元しました")
            
        except Exception as e:
            print(f"共有コピーモード: 復元中にエラーが発生しました: {e}")
        finally:
            # 明示的にメモリ解放
            backup.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    
    return model, restore_original_weights

def hybrid_copying_strategy(model, high_vram=False):
    """
    環境に応じてコピー戦略を選択するハイブリッドアプローチ
    
    Args:
        model: コピー元のtransformerモデル
        high_vram: 高VRAMモードかどうか
    
    Returns:
        コピー戦略に応じたモデルとオプションの復元関数
    """
    # システム情報を取得
    memory_info = get_system_memory_info()
    print(f"[LoRA] メモリ情報: RAM {memory_info['ram']['usage_gb']:.2f}GB ({memory_info['ram']['percent']:.1f}%), "
          f"VRAM {memory_info['vram']['usage_gb']:.2f}GB/{memory_info['vram']['total_gb']:.2f}GB ({memory_info['vram']['percent']:.1f}%)")
    
    # 利用可能なRAMを確認
    available_ram_gb = memory_info["ram"]["percent"] < 70
    
    # 利用可能なVRAMを確認
    available_vram_gb = memory_info["vram"]["percent"] < 70
    
    # 環境に応じてコピー戦略を選択
    if memory_info["ram"]["usage_gb"] > 0 and available_ram_gb and available_vram_gb:
        # 十分なメモリがある場合は効率的なコピーを使用
        print("[LoRA] 十分なメモリが利用可能: 効率的なコピー戦略を使用")
        return create_efficient_model_copy(model), None
    elif high_vram and available_vram_gb:
        # 高VRAMモードで中程度のVRAMがある場合はメモリ効率の良いコピーを使用
        print("[LoRA] 高VRAMモード: メモリ効率の良いコピー戦略を使用")
        return create_memory_efficient_copy(model), None
    else:
        # メモリが限られている場合は共有コピーとバックアップを使用
        print("[LoRA] 限られたメモリ環境: 共有コピー戦略とバックアップを使用")
        return create_shared_copy_with_backup(model)

def calculate_lora_memory_requirements(model, lora_path=None, lora_rank=4):
    """
    LoRA適用のためのメモリ要件を計算します
    
    Args:
        model: 対象のモデル
        lora_path: LoRAファイルパス（オプション）
        lora_rank: LoRAのランク値（デフォルト: 4）
        
    Returns:
        dict: メモリ要件の詳細情報
    """
    # モデルのパラメータ数を計算
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 2 / (1024 * 1024)  # BF16/FP16を想定して2バイト
    
    # アダプタ対象レイヤー数を概算（線形レイヤー）
    adapter_targets = 0
    linear_layer_sizes = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            adapter_targets += 1
            in_features = module.in_features
            out_features = module.out_features
            linear_layer_sizes.append((in_features, out_features))
    
    # LoRAファイルサイズ
    lora_file_size_mb = 0
    if lora_path and os.path.exists(lora_path):
        lora_file_size_mb = os.path.getsize(lora_path) / (1024 * 1024)
    
    # 理論的なLoRAアダプタサイズを計算
    lora_size_mb = 0
    for in_feat, out_feat in linear_layer_sizes:
        # down_projection + up_projection のパラメータ数
        lora_params = (in_feat * lora_rank) + (lora_rank * out_feat)
        lora_size_mb += lora_params * 2 / (1024 * 1024)  # BF16/FP16を想定して2バイト
    
    # 推定メモリ要件
    estimated_ram = model_size_mb * 2 + lora_size_mb  # モデルコピー + LoRAアダプタ
    estimated_vram = model_size_mb + lora_size_mb  # 実行時に必要なVRAM
    
    # バッファを考慮した推奨メモリ
    recommended_ram = estimated_ram + 2048  # +2GB バッファ
    recommended_vram = estimated_vram + 2048  # +2GB バッファ
    
    # 結果をログ出力
    print(f"===== LoRAメモリ要件予測 =====")
    print(f"モデルサイズ: {model_size_mb:.2f}MB ({param_count:,} パラメータ)")
    print(f"アダプタ対象レイヤー: {adapter_targets}層")
    print(f"推定LoRAアダプタサイズ: {lora_size_mb:.2f}MB (ランク {lora_rank})")
    if lora_file_size_mb > 0:
        print(f"実際のLoRAファイルサイズ: {lora_file_size_mb:.2f}MB")
    print(f"推定必要RAM: {estimated_ram:.2f}MB (推奨: {recommended_ram:.2f}MB)")
    print(f"推定必要VRAM: {estimated_vram:.2f}MB (推奨: {recommended_vram:.2f}MB)")
    print("=" * 30)
    
    return {
        "model_size_mb": model_size_mb,
        "param_count": param_count,
        "adapter_targets": adapter_targets,
        "lora_size_mb": lora_size_mb,
        "lora_file_size_mb": lora_file_size_mb,
        "estimated_ram": estimated_ram,
        "estimated_vram": estimated_vram,
        "recommended_ram": recommended_ram,
        "recommended_vram": recommended_vram
    }

def replace_deepcopy(model, high_vram=False):
    """
    copy.deepcopyの直接的な代替関数。
    endframe_ichi.pyでの最小限の変更で使用できます。
    
    Args:
        model: コピーするモデル
        high_vram: 高VRAMモードかどうか
    
    Returns:
        モデルのコピー
    """
    # メモリ使用量をログ出力
    log_memory_usage("[LoRA] コピー前")
    
    # ハイブリッドコピー戦略を使用
    result, restore_func = hybrid_copying_strategy(model, high_vram)
    
    # 復元関数がある場合はグローバル変数に保存
    if restore_func is not None:
        global _last_restore_func
        _last_restore_func = restore_func
    
    # メモリ使用量の変化をログ出力
    log_memory_usage("[LoRA] コピー後")
    
    return result

def restore_last_model():
    """
    最後に作成したモデルを元の状態に復元します。
    replace_deepcopyを使用した場合で、復元関数が利用可能な場合のみ機能します。
    """
    global _last_restore_func
    if '_last_restore_func' in globals() and _last_restore_func is not None:
        _last_restore_func()
        _last_restore_func = None
        return True
    else:
        print("[LoRA] 復元可能なモデルがありません")
        return False

# 最後の復元関数を保持するグローバル変数
_last_restore_func = None