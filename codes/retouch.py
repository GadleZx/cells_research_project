import cv2
import numpy as np
import os
import argparse
from glob import glob

def lightroom_like_process_grayscale(input_path, output_path):
    # === 画像読み込み（グレースケール & 正規化）===
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # === 1. 露光量補正（+3.45 EV ≒ 2^3.45）===
    img *= 2.0 ** 3.45  # 約10.93倍

    # === 2. 黒レベル補正（-86）===
    img -= 0.6

    # === 3. 白レベル補正（明部のみ加算）===
    img[img > 0.98] += 45 / 255.0

    # === 4. ハイライト補正（明部をさらに持ち上げ）===
    highlight_mask = img > 180 / 255.0
    img[highlight_mask] += (1.0 - img[highlight_mask]) * 0.3

    # === 5. コントラスト補正（+100）===
    img = (img - 0.5) * 1.4 + 0.5

    # === 6. シャープ化（テクスチャ + 明瞭度）===
    def unsharp_mask(image, strength=1.5, blur_size=(3, 3)):
        blurred = cv2.GaussianBlur(image, blur_size, 0)
        return np.clip(image * (1 + strength) - blurred * strength, 0, 1)

    img = unsharp_mask(img, strength=1.5)

    # === 7. かすみの除去（CLAHE）===
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.2, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_uint8)

    # === 保存 ===
    cv2.imwrite(output_path, img_clahe)
    print(f"✅ 保存完了: {output_path}")

def process_directory(input_dir, output_dir):
    # 入力ディレクトリ内の画像ファイルをすべて取得（png, jpg, jpeg）
    image_paths = glob(os.path.join(input_dir, "*.*"))
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        lightroom_like_process_grayscale(img_path, output_path)

# === コマンドライン引数処理 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ディレクトリ内の画像をLightroom風に処理します。")
    parser.add_argument("--input_dir", type=str, required=True, help="入力ディレクトリのパス")
    parser.add_argument("--output_dir", type=str, required=True, help="出力ディレクトリのパス")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
