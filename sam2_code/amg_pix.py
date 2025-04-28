import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from PIL import Image
import numpy as np
import os
import argparse
from glob import glob

# モデルの初期化
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
model = build_sam2(model_cfg, checkpoint)

mask_generator = SAM2AutomaticMaskGenerator(
    model,
    points_per_side=100,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.85,
    crop_n_layers=0,
    min_mask_region_area=160,
)

def process_image(image_path, output_dir, min_height=570, max_height=720, min_width=570, max_width=720):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # マスク生成
    with torch.inference_mode():
        masks = mask_generator.generate(image_np)
    print(f"[{os.path.basename(image_path)}] {len(masks)} masks detected.")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_output_dir = os.path.join(output_dir, f"{base_name}_masks")
    os.makedirs(mask_output_dir, exist_ok=True)

    valid_count = 0
    for i, mask in enumerate(masks):
        seg = mask["segmentation"]

        coords = np.argwhere(seg)
        if coords.size == 0:
            continue

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        bbox_height = y1 - y0
        bbox_width = x1 - x0

        # ★ 縦横サイズでフィルタリング
        if not (min_height <= bbox_height <= max_height):
            continue
        if not (min_width <= bbox_width <= max_width):
            continue

        cropped = image_np[y0:y1, x0:x1]
        cropped_image = Image.fromarray(cropped)
        cropped_image.save(os.path.join(mask_output_dir, f"segment_{valid_count:03d}.png"))
        valid_count += 1

    print(f"Saved {valid_count} segmented crops to {mask_output_dir}\\n")

def main():
    parser = argparse.ArgumentParser(description="SAM2 segmentation with height/width filtering.")
    parser.add_argument("--input_dir", required=True, help="Path to input directory with images")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--min_height", type=int, default=570, help="Minimum Bbox height")
    parser.add_argument("--max_height", type=int, default=720, help="Maximum Bbox height")
    parser.add_argument("--min_width", type=int, default=570, help="Minimum Bbox width")
    parser.add_argument("--max_width", type=int, default=720, help="Maximum Bbox width")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = glob(os.path.join(args.input_dir, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))]

    if not image_paths:
        print("No valid image files found in the input directory.")
        return

    for image_path in image_paths:
        process_image(
            image_path, args.output_dir,
            min_height=args.min_height, max_height=args.max_height,
            min_width=args.min_width, max_width=args.max_width
        )

if __name__ == "__main__":
    main()
