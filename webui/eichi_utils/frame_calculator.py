"""
フレームサイズに基づくセクション数計算モジュール
0.5秒/1秒モードに対応したセクション数の計算機能を提供します
"""

import math
from eichi_utils.video_mode_settings import VIDEO_MODE_SETTINGS

from locales import i18n

def calculate_frames_per_section(latent_window_size=9):
    """1セクションあたりのフレーム数を計算"""
    return latent_window_size * 4 - 3


def calculate_sections_from_frames(total_frames, latent_window_size=9):
    """フレーム数から必要なセクション数を計算"""
    frames_per_section = calculate_frames_per_section(latent_window_size)
    return math.ceil(total_frames / frames_per_section)


def calculate_total_frame_count(sections, latent_window_size=9):
    """セクション数から総フレーム数を計算"""
    frames_per_section = calculate_frames_per_section(latent_window_size)
    return sections * frames_per_section


def calculate_total_second_length(frames, fps=30):
    """フレーム数から秒数を計算"""
    return frames / fps


def calculate_sections_for_mode_and_size(mode_key, frame_size_setting=None):
    """動画モードとフレームサイズ設定から必要なセクション数を計算"""

    if frame_size_setting is None:
        frame_size_setting = i18n.translate("1秒 (33フレーム)")

    # 動画モードから秒数を取得
    if mode_key not in VIDEO_MODE_SETTINGS:
        return 15  # デフォルト値

    # 動的計算ではなく、VIDEO_MODE_SETTINGSで定義された静的なセクション数を使用
    required_sections = VIDEO_MODE_SETTINGS[mode_key]["sections"]

    # フレームサイズ設定の情報をログ用に取得
    if frame_size_setting == i18n.translate("0.5秒 (17フレーム)"):
        latent_window_size = 5  # 0.5秒モード
    else:
        latent_window_size = 9  # 1秒モードがデフォルト

    # デバッグ情報
    total_seconds = VIDEO_MODE_SETTINGS[mode_key]["display_seconds"]
    frames_per_section = calculate_frames_per_section(latent_window_size)
    total_frames = int(total_seconds * 30)
    print(i18n.translate("静的セクション数使用: モード={mode_key}, フレームサイズ={frame_size_setting}, 秒数={total_seconds}, セクション数={required_sections}").format(mode_key=mode_key, frame_size_setting=frame_size_setting, total_seconds=total_seconds, required_sections=required_sections))

    # 結果を返す
    return required_sections
