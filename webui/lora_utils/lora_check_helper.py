"""
FramePack-eichi LoRA診断ヘルパーモジュール
LoRAの適用状況を診断するための機能を提供します。
最小限の診断機能のみを実装。
"""

import torch
import logging
from typing import Dict, List, Optional, Any, Union, Tuple

from locales import i18n

# ロギング設定
logger = logging.getLogger("lora_check_helper")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def check_lora_applied(model: torch.nn.Module) -> Tuple[bool, str]:
    """
    モデルにLoRAが適用されているかをチェック

    Args:
        model: チェック対象のモデル

    Returns:
        Tuple[bool, str]: 適用されているかどうかとその方法
    """
    # _lora_appliedフラグの確認
    has_flag = hasattr(model, '_lora_applied') and model._lora_applied
    source = getattr(model, '_lora_source', 'unknown') if has_flag else 'none'

    return has_flag, source

def diagnose_lora_application_failure(model: torch.nn.Module, lora_state_dict: Dict[str, torch.Tensor]) -> str:
    """
    LoRA適用に失敗した原因を診断

    Args:
        model: 診断対象のモデル
        lora_state_dict: LoRAの状態辞書

    Returns:
        str: 診断結果
    """
    # フラグチェック
    has_flag, source = check_lora_applied(model)
    if has_flag:
        return i18n.translate("LoRAフラグは正常に設定されています（適用方法: {source}）").format(source=source)

    # LoRAキーの存在チェック
    lora_keys = [k for k in lora_state_dict.keys() if '.lora_down' in k or '.lora_up' in k]
    if len(lora_keys) == 0:
        return i18n.translate("LoRA状態辞書に有効なLoRAキーが見つかりません")

    # モデルパラメータとLoRAキーの対応チェック
    model_params = dict(model.named_parameters())
    matching_count = 0

    for lora_key in lora_keys:
        if '.lora_down' in lora_key:
            param_name = lora_key.replace('.lora_down', '')
            if param_name in model_params:
                matching_count += 1

    if matching_count == 0:
        return i18n.translate("モデルパラメータとLoRAキーが一致しません。モデル構造の違いや名前空間の不一致が考えられます。")

    # その他の一般的な問題
    return i18n.translate("不明な原因でLoRAの適用に失敗しました。詳細なログを確認してください。")

def create_lora_stats_report(model: torch.nn.Module, lora_name: str = "", lora_state_dict: Optional[Dict[str, torch.Tensor]] = None) -> str:
    """
    LoRA適用状況の統計レポートを生成

    Args:
        model: 診断対象のモデル
        lora_name: LoRAの名前
        lora_state_dict: LoRAの状態辞書

    Returns:
        str: 診断レポート
    """
    # LoRA適用チェック
    has_lora, source = check_lora_applied(model)

    report = [i18n.translate("--- LoRA適用状況レポート: {lora_name} ---").format(lora_name=lora_name)]
    report.append(i18n.translate("LoRA適用フラグ: {has_lora}").format(has_lora=i18n.translate("有効") if has_lora else i18n.translate("無効")))
    report.append(i18n.translate("適用方法: {source}").format(source=source))

    # LoRA辞書のチェック
    if lora_state_dict is not None:
        total_params = sum(v.numel() for k, v in lora_state_dict.items() if isinstance(v, torch.Tensor) and ('lora_down' in k or 'lora_up' in k))
        down_keys = [k for k in lora_state_dict.keys() if 'lora_down' in k]
        up_keys = [k for k in lora_state_dict.keys() if 'lora_up' in k]

        report.append(i18n.translate("LoRA辞書: {count} エントリ").format(count=len(lora_state_dict)))
        report.append(i18n.translate("lora_downキー: {count}").format(count=len(down_keys)))
        report.append(i18n.translate("lora_upキー: {count}").format(count=len(up_keys)))
        report.append(i18n.translate("合計パラメータ数: {count}").format(count=total_params))

    # 適用に失敗した場合の診断
    if not has_lora and lora_state_dict is not None:
        diagnosis = diagnose_lora_application_failure(model, lora_state_dict)
        report.append(i18n.translate("問題診断: {diagnosis}").format(diagnosis=diagnosis))

    return "\n".join(report)
