# png_metadata.py
# FramePack-eichiのためのPNGメタデータ処理ユーティリティ

import json
import os
from PIL import Image, PngImagePlugin
import traceback

# メタデータキーの定義
PROMPT_KEY = "prompt"
SEED_KEY = "seed"
SECTION_PROMPT_KEY = "section_prompt"
SECTION_NUMBER_KEY = "section_number"
PARAMETERS_KEY = "parameters"  # SD系との互換性のため

def embed_metadata_to_png(image_path, metadata_dict):
    """PNGファイルにメタデータを埋め込む
    
    Args:
        image_path (str): PNGファイルのパス
        metadata_dict (dict): 埋め込むメタデータの辞書
    
    Returns:
        str: 処理したファイルのパス
    """
    try:
        # print(f"[DEBUG] メタデータ埋め込み開始: {image_path}")
        # print(f"[DEBUG] 埋め込むメタデータ: {metadata_dict}")
        
        img = Image.open(image_path)
        metadata = PngImagePlugin.PngInfo()
        
        # パラメータを結合したテキストも作成（SD系との互換性のため）
        parameters_text = ""
        
        for key, value in metadata_dict.items():
            if value is not None:
                if isinstance(value, (dict, list)):
                    # 辞書やリストの場合はJSON文字列に変換
                    metadata.add_text(key, json.dumps(value))
                    # print(f"[DEBUG] JSON形式でメタデータ追加: {key}={json.dumps(value)}")
                else:
                    # その他の値は文字列に変換
                    metadata.add_text(key, str(value))
                    # print(f"[DEBUG] 文字列形式でメタデータ追加: {key}={value}")
                    
                # パラメータテキストの構築
                if key == PROMPT_KEY:
                    parameters_text += f"{value}\n"
                elif key == SEED_KEY:
                    parameters_text += f"Seed: {value}\n"
                elif key == SECTION_PROMPT_KEY:
                    parameters_text += f"Section Prompt: {value}\n"
                elif key == SECTION_NUMBER_KEY:
                    parameters_text += f"Section Number: {value}\n"
        
        # パラメータテキストがあれば追加
        if parameters_text:
            metadata.add_text(PARAMETERS_KEY, parameters_text.strip())
            # print(f"[DEBUG] parameters形式でメタデータ追加: {parameters_text.strip()}")
        
        # 保存（ファイル形式は変更せず）
        if image_path.lower().endswith('.png'):
            img.save(image_path, "PNG", pnginfo=metadata)
        else:
            # PNGでなければPNGに変換して保存
            png_path = os.path.splitext(image_path)[0] + '.png'
            img.save(png_path, "PNG", pnginfo=metadata)
            image_path = png_path
            
        # print(f"[DEBUG] メタデータを埋め込みました: {image_path}")
        return image_path
    
    except Exception as e:
        print(f"[ERROR] メタデータ埋め込みエラー: {e}")
        traceback.print_exc()
        return image_path

def extract_metadata_from_png(image_path_or_object):
    """PNGファイルからメタデータを抽出する
    
    Args:
        image_path_or_object: PNGファイルのパスまたはPIL.Image.Imageオブジェクト
    
    Returns:
        dict: 抽出したメタデータの辞書
    """
    try:
        # パスが文字列ならイメージを開く、イメージオブジェクトならそのまま使う
        if isinstance(image_path_or_object, str):
            if not os.path.exists(image_path_or_object):
                print(f"[DEBUG] ファイルが存在しません: {image_path_or_object}")
                return {}
            print(f"[DEBUG] 画像ファイルを開いています: {image_path_or_object}")
            img = Image.open(image_path_or_object)
        else:
            print(f"[DEBUG] 与えられた画像オブジェクトを使用します")
            img = image_path_or_object
        
        print(f"[DEBUG] PIL Image info: {img.info}")
        
        metadata = {}
        
        # 個別のキーを処理
        for key in [PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY, SECTION_NUMBER_KEY, PARAMETERS_KEY]:
            if key in img.info:
                value = img.info[key]
                print(f"[DEBUG] メタデータ発見: {key}={value}")
                try:
                    # JSONとして解析を試みる
                    metadata[key] = json.loads(value)
                    print(f"[DEBUG] JSONとして解析: {key}={metadata[key]}")
                except (json.JSONDecodeError, TypeError):
                    # JSONでなければそのまま格納
                    metadata[key] = value
                    print(f"[DEBUG] 文字列として解析: {key}={value}")
        
        # parametersキーからの抽出処理（SD系との互換性のため）
        if PARAMETERS_KEY in metadata and isinstance(metadata[PARAMETERS_KEY], str):
            params = metadata[PARAMETERS_KEY]
            print(f"[DEBUG] parameters形式のメタデータを処理: {params}")
            
            # プロンプト抽出（最初の行がプロンプト）
            if PROMPT_KEY not in metadata and "\n" in params:
                first_line = params.split("\n")[0].strip()
                if first_line and not first_line.startswith("Seed:") and not first_line.startswith("Section"):
                    metadata[PROMPT_KEY] = first_line
                    print(f"[DEBUG] parameters形式からプロンプトを抽出: {first_line}")
            
            # Seed抽出
            if SEED_KEY not in metadata and "Seed:" in params:
                seed_parts = [p.strip() for p in params.split("Seed:")[1].split("\n")[0].split(",")]
                if seed_parts[0].isdigit():
                    metadata[SEED_KEY] = int(seed_parts[0])
                    print(f"[DEBUG] parameters形式からSEEDを抽出: {metadata[SEED_KEY]}")
        
        print(f"[DEBUG] 最終抽出メタデータ: {metadata}")
        return metadata
    
    except Exception as e:
        print(f"[ERROR] メタデータ抽出エラー: {e}")
        traceback.print_exc()
        return {}

def extract_metadata_from_numpy_array(numpy_image):
    """NumPy配列からメタデータを抽出する（PILを介して）
    
    Args:
        numpy_image: NumPy配列の画像データ
    
    Returns:
        dict: 抽出したメタデータの辞書
    """
    try:
        if numpy_image is None:
            print("[DEBUG] extract_metadata_from_numpy_array: 入力がNoneです")
            return {}
        
        print(f"[DEBUG] numpy_imageのサイズと形状: {numpy_image.shape}, データ型: {numpy_image.dtype}")
        
        # NumPy配列からPIL.Imageに変換
        img = Image.fromarray(numpy_image)
        print(f"[DEBUG] PIL Imageサイズ: {img.size}, モード: {img.mode}")
        print(f"[DEBUG] PIL Image info: {img.info}")
        
        # メタデータの抽出を試行
        metadata = extract_metadata_from_png(img)
        print(f"[DEBUG] 抽出されたメタデータ: {metadata}")
        
        return metadata
    
    except Exception as e:
        print(f"[ERROR] NumPy配列からのメタデータ抽出エラー: {e}")
        traceback.print_exc()
        return {}
