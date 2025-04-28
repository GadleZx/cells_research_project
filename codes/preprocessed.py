import cv2
import numpy as np
import os

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def correct_illumination(img, blur_kernel=101):
    illumination = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    corrected = img.astype(np.float32) / (illumination + 1e-6)
    corrected *= 128
    return np.clip(corrected, 0, 255).astype(np.uint8)

def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    return clahe.apply(img)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def preprocess_fluorescence_image(input_path, output_dir="preprocessed"):
    os.makedirs(output_dir, exist_ok=True)

    # ① 入力画像の読み込み
    img = load_image(input_path)

    # ② 照明ムラ補正
    corrected = correct_illumination(img)

    # ③ コントラスト強調（CLAHE）
    contrast = enhance_contrast(corrected)

    # ④ シャープ化
    sharpened = sharpen_image(contrast)

    # ⑤ 中間結果の保存
    cv2.imwrite(os.path.join(output_dir, "corrected.png"), corrected)
    cv2.imwrite(os.path.join(output_dir, "contrast.png"), contrast)
    cv2.imwrite(os.path.join(output_dir, "sharpened.png"), sharpened)

    print(f"保存完了: {output_dir}/sharpened.png をSAM2へ入力可能")

if __name__ == "__main__":
    preprocess_fluorescence_image("/home/umelab/workspace/Data/250204/VH+soil/3h/x60/Image_Sequence_1-FL/1-FL_Stack0000.png")
