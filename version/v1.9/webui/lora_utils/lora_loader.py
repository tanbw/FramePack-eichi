# FramePack-eichi LoRA Loader
#
# LoRAモデルの読み込みと適用のための機能を提供します。

import os
import torch
from tqdm import tqdm
from .lora_utils import merge_lora_to_state_dict

# 国際化対応
try:
    from locales import i18n
    HAS_I18N = True
except ImportError:
    HAS_I18N = False
    print("Warning: i18n module not found, using fallback translations")

# 翻訳ヘルパー関数
def _(text):
    """国際化対応のためのヘルパー関数"""
    if HAS_I18N:
        return i18n.translate(text)
    return text

def load_and_apply_lora(
    model,
    lora_path,
    lora_scale=0.8,
    device=None
):
    """
    LoRA重みをロードしてモデルに適用する
    
    Args:
        model: 適用先のモデル
        lora_path: LoRAファイルのパス
        lora_scale: LoRAの適用強度
        device: 計算に使用するデバイス

    Returns:
        LoRAが適用されたモデル
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(_("LoRAファイルが見つかりません: {0}").format(lora_path))

    if device is None:
        device = next(model.parameters()).device

    print(_("LoRAを読み込み中: {0} (スケール: {1})").format(os.path.basename(lora_path), lora_scale))
    
    # モデルの状態辞書を取得
    state_dict = model.state_dict()
    
    # LoRA重みを状態辞書にマージ
    print(_("フォーマット: HunyuanVideo"))
    
    # LoRAをマージ
    merged_state_dict = merge_lora_to_state_dict(state_dict, lora_path, lora_scale, device)

    # 状態辞書をモデルに読み込み
    model.load_state_dict(merged_state_dict, strict=True)
    
    # LoRAが適用されたことを示すフラグを設定
    model._lora_applied = True
    
    print(_("LoRAの適用が完了しました"))
    return model

def check_lora_applied(model):
    """
    モデルにLoRAが適用されているかをチェック
    
    Args:
        model: チェック対象のモデル
        
    Returns:
        (bool, str): LoRAが適用されているかどうかとその適用方法
    """
    # _lora_appliedフラグのチェック
    has_flag = hasattr(model, '_lora_applied') and model._lora_applied
    
    if has_flag:
        return True, "direct_application"
    
    return False, "none"
