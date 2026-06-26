# -*- coding: utf-8 -*-
"""
bitemporal_region_modeling_batch.py

功能：
1. 批量读取两个文件夹中的双时相图像
2. 按同名文件自动配对
3. 为每对图像生成共享超像素/区域分割图
4. 对每个区域提取前时相、后时相像素集合
5. 保存每对图像的：
   - region label map
   - 可视化边界图
   - 每个区域的基本统计信息
   - 每个区域的像素集合（npz）
6. 生成总处理日志 summary.csv

适用：
- 灰度 SAR 图像
- RGB 光学遥感图像
- 已完成配准的双时相图像
"""

import os
import json
import traceback
import numpy as np
import cv2
import pandas as pd

from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.util import img_as_float


# =========================================================
# 这里直接写路径和参数
# =========================================================
PRE_DIR = r"/home/f523/disk2/nym/open-cd-main/data/WHU-CD-256/A"
POST_DIR = r"/home/f523/disk2/nym/open-cd-main/data/WHU-CD-256/B"
OUT_DIR = r"/home/f523/disk2/nym/open-cd-main/data/WHU-CD-256/segout-250"

GUIDANCE_MODE = "mean"      # 可选: "mean", "stack", "diffmean"
SCALE = 250.0              # 越大区域越大
SIGMA = 0.6                 # 高斯平滑参数
MIN_SIZE = 220              # 最小区域面积
GRAY_ONLY = True            # True: 提取灰度/强度集合；False: 提取多通道像素

IMG_SUFFIX = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# =========================================================
# 基础函数
# =========================================================
def read_image(path):
    """读取图像，支持中文路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def save_image(path, img):
    """保存图像，支持中文路径。"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"无法保存图像: {path}")
    buf.tofile(path)


def ensure_same_size(img1, img2):
    if img1.shape[:2] != img2.shape[:2]:
        raise ValueError(
            f"双时相尺寸不一致: pre={img1.shape[:2]}, post={img2.shape[:2]}"
        )


def to_gray_float(img):
    """
    转为 [0,1] 浮点灰度图。
    对 SAR/灰度图：直接归一化
    对 RGB/BGR：转灰度
    """
    if img.ndim == 2:
        g = img.astype(np.float32)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            g = img[..., 0].astype(np.float32)
        else:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        raise ValueError("不支持的图像维度")

    g -= g.min()
    if g.max() > 0:
        g /= g.max()
    return g


def to_3ch_float(img):
    """
    转为 [0,1] 的三通道图像，用于图分割。
    灰度图会复制成 3 通道。
    """
    if img.ndim == 2:
        x = img.astype(np.float32)
        x -= x.min()
        if x.max() > 0:
            x /= x.max()
        x = np.stack([x, x, x], axis=-1)
        return x

    if img.ndim == 3:
        if img.shape[2] == 1:
            x = img[..., 0].astype(np.float32)
            x -= x.min()
            if x.max() > 0:
                x /= x.max()
            x = np.stack([x, x, x], axis=-1)
            return x
        else:
            x = img.astype(np.float32)
            if x.max() > 1.0:
                x /= 255.0
            return x

    raise ValueError("不支持的图像维度")


def build_shared_guidance(pre_img, post_img, mode="mean"):
    """
    构建共享引导图，用于生成同一套超像素分割。
    """
    pre3 = to_3ch_float(pre_img)
    post3 = to_3ch_float(post_img)

    if mode == "mean":
        guide = 0.5 * pre3 + 0.5 * post3

    elif mode == "stack":
        pre_g = to_gray_float(pre_img)
        post_g = to_gray_float(post_img)
        mean_g = 0.5 * pre_g + 0.5 * post_g
        guide = np.stack([pre_g, post_g, mean_g], axis=-1)

    elif mode == "diffmean":
        pre_g = to_gray_float(pre_img)
        post_g = to_gray_float(post_img)
        mean_g = 0.5 * pre_g + 0.5 * post_g
        diff_g = np.abs(pre_g - post_g)
        guide = np.stack([mean_g, diff_g, mean_g], axis=-1)

    else:
        raise ValueError(f"未知 guidance mode: {mode}")

    guide = np.clip(guide, 0, 1)
    return guide


def generate_shared_superpixels(guide_img, scale=100, sigma=0.8, min_size=80):
    """
    基于图的分割（Felzenszwalb），属于典型 MST/图分割思路。
    """
    labels = felzenszwalb(
        img_as_float(guide_img),
        scale=scale,
        sigma=sigma,
        min_size=min_size
    )
    return labels.astype(np.int32)


def extract_region_pixels(pre_img, post_img, labels, gray_only=True):
    """
    对每个区域提取前后时相像素集合。
    """
    region_ids = np.unique(labels)

    if gray_only:
        pre_data = to_gray_float(pre_img)
        post_data = to_gray_float(post_img)
    else:
        pre_data = to_3ch_float(pre_img)
        post_data = to_3ch_float(post_img)

    region_dict = {}
    stats = []

    for rid in region_ids:
        mask = (labels == rid)
        ys, xs = np.where(mask)
        n = len(xs)
        if n == 0:
            continue

        x_m = pre_data[mask].astype(np.float32)
        y_m = post_data[mask].astype(np.float32)

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        region_dict[int(rid)] = {
            "pre_pixels": x_m,
            "post_pixels": y_m,
            "coords": np.stack([ys, xs], axis=1).astype(np.int32),
        }

        if gray_only:
            stats.append({
                "region_id": int(rid),
                "num_pixels": int(n),
                "bbox_xmin": int(x0),
                "bbox_ymin": int(y0),
                "bbox_xmax": int(x1),
                "bbox_ymax": int(y1),
                "pre_mean": float(np.mean(x_m)),
                "pre_std": float(np.std(x_m)),
                "post_mean": float(np.mean(y_m)),
                "post_std": float(np.std(y_m)),
                "abs_mean_diff": float(abs(np.mean(x_m) - np.mean(y_m))),
            })
        else:
            stats.append({
                "region_id": int(rid),
                "num_pixels": int(n),
                "bbox_xmin": int(x0),
                "bbox_ymin": int(y0),
                "bbox_xmax": int(x1),
                "bbox_ymax": int(y1),
                "pre_mean": x_m.mean(axis=0).tolist(),
                "pre_std": x_m.std(axis=0).tolist(),
                "post_mean": y_m.mean(axis=0).tolist(),
                "post_std": y_m.std(axis=0).tolist(),
            })

    stats_df = pd.DataFrame(stats)
    return region_dict, stats_df


def save_region_npz(region_dict, save_path):
    """
    保存每个区域像素集合。
    """
    save_data = {}
    for rid, item in region_dict.items():
        save_data[f"region_{rid}_pre"] = item["pre_pixels"]
        save_data[f"region_{rid}_post"] = item["post_pixels"]
        save_data[f"region_{rid}_coords"] = item["coords"]
    np.savez_compressed(save_path, **save_data)


def colorize_labels(labels):
    """
    将 label map 着色可视化。
    """
    rng = np.random.default_rng(1234)
    num_labels = labels.max() + 1
    colors = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)
    color_map = colors[labels]
    return color_map


def get_stem(filename):
    """去掉后缀后的文件名。"""
    return os.path.splitext(filename)[0]


def list_images(folder):
    """列出文件夹内所有图像文件。"""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"文件夹不存在: {folder}")

    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(IMG_SUFFIX):
            files.append(name)
    files.sort()
    return files


# =========================================================
# 单对图像处理
# =========================================================
def process_one_pair(pre_path, post_path, pair_out_dir):
    os.makedirs(pair_out_dir, exist_ok=True)

    pre_img = read_image(pre_path)
    post_img = read_image(post_path)
    ensure_same_size(pre_img, post_img)

    # 1) 共享引导图
    guide = build_shared_guidance(pre_img, post_img, mode=GUIDANCE_MODE)

    # 2) 共享超像素分割
    labels = generate_shared_superpixels(
        guide_img=guide,
        scale=SCALE,
        sigma=SIGMA,
        min_size=MIN_SIZE
    )

    # 3) 提取区域像素
    region_dict, stats_df = extract_region_pixels(
        pre_img, post_img, labels, gray_only=GRAY_ONLY
    )

    # 4) 保存标签图
    label_npy = os.path.join(pair_out_dir, "shared_labels.npy")
    np.save(label_npy, labels)

    label_png = os.path.join(pair_out_dir, "shared_labels.png")
    label_vis_16 = labels.astype(np.uint16)
    save_image(label_png, label_vis_16)

    # 5) 彩色标签图
    color_map = colorize_labels(labels)
    save_image(os.path.join(pair_out_dir, "shared_labels_color.png"), color_map)

    # 6) 边界图
    # pre_show = to_3ch_float(pre_img)
    # boundary_vis = (mark_boundaries(pre_show, labels) * 255).astype(np.uint8)
    # boundary_vis = cv2.cvtColor(boundary_vis, cv2.COLOR_RGB2BGR)
    # save_image(os.path.join(pair_out_dir, "pre_with_boundaries.png"), boundary_vis)

    # 6) A/B 双时相边界图
    pre_show = to_3ch_float(pre_img)
    post_show = to_3ch_float(post_img)

    pre_boundary_vis = (mark_boundaries(pre_show, labels) * 255).astype(np.uint8)
    post_boundary_vis = (mark_boundaries(post_show, labels) * 255).astype(np.uint8)

    pre_boundary_vis = cv2.cvtColor(pre_boundary_vis, cv2.COLOR_RGB2BGR)
    post_boundary_vis = cv2.cvtColor(post_boundary_vis, cv2.COLOR_RGB2BGR)

    save_image(os.path.join(pair_out_dir, "A_with_boundaries.png"), pre_boundary_vis)
    save_image(os.path.join(pair_out_dir, "B_with_boundaries.png"), post_boundary_vis)
    

    # 7) 区域统计
    stats_csv = os.path.join(pair_out_dir, "region_stats.csv")
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")

    # 8) 区域像素集合
    npz_path = os.path.join(pair_out_dir, "region_pixels.npz")
    save_region_npz(region_dict, npz_path)

    # 9) 保存配置
    meta = {
        "pre_path": pre_path,
        "post_path": post_path,
        "guidance": GUIDANCE_MODE,
        "scale": SCALE,
        "sigma": SIGMA,
        "min_size": MIN_SIZE,
        "gray_only": GRAY_ONLY,
        "num_regions": int(labels.max() + 1),
        "image_shape": list(labels.shape),
    }
    with open(os.path.join(pair_out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "num_regions": int(labels.max() + 1),
        "image_h": int(labels.shape[0]),
        "image_w": int(labels.shape[1]),
        "out_dir": pair_out_dir,
    }


# =========================================================
# 主流程：批量处理
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pre_files = list_images(PRE_DIR)
    post_files = list_images(POST_DIR)

    pre_map = {get_stem(f): f for f in pre_files}
    post_map = {get_stem(f): f for f in post_files}

    common_names = sorted(set(pre_map.keys()) & set(post_map.keys()))
    only_pre = sorted(set(pre_map.keys()) - set(post_map.keys()))
    only_post = sorted(set(post_map.keys()) - set(pre_map.keys()))

    print(f"前时相图像数: {len(pre_files)}")
    print(f"后时相图像数: {len(post_files)}")
    print(f"成功配对数: {len(common_names)}")
    print(f"仅前时相存在: {len(only_pre)}")
    print(f"仅后时相存在: {len(only_post)}")

    if only_pre:
        print("\n以下文件仅在 PRE_DIR 中存在：")
        for name in only_pre:
            print("  ", pre_map[name])

    if only_post:
        print("\n以下文件仅在 POST_DIR 中存在：")
        for name in only_post:
            print("  ", post_map[name])

    summary = []

    for idx, name in enumerate(common_names, 1):
        pre_name = pre_map[name]
        post_name = post_map[name]

        pre_path = os.path.join(PRE_DIR, pre_name)
        post_path = os.path.join(POST_DIR, post_name)

        pair_out_dir = os.path.join(OUT_DIR, name)

        print(f"\n[{idx}/{len(common_names)}] 正在处理: {name}")

        try:
            result = process_one_pair(pre_path, post_path, pair_out_dir)
            summary.append({
                "name": name,
                "pre_file": pre_name,
                "post_file": post_name,
                "status": result["status"],
                "num_regions": result["num_regions"],
                "image_h": result["image_h"],
                "image_w": result["image_w"],
                "out_dir": result["out_dir"],
                "error": "",
            })
            print(f"  处理成功，区域数 = {result['num_regions']}")

        except Exception as e:
            err_msg = str(e)
            summary.append({
                "name": name,
                "pre_file": pre_name,
                "post_file": post_name,
                "status": "failed",
                "num_regions": "",
                "image_h": "",
                "image_w": "",
                "out_dir": pair_out_dir,
                "error": err_msg,
            })
            print(f"  处理失败: {err_msg}")
            traceback.print_exc()

    # 保存总表
    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(OUT_DIR, "summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # 保存未配对信息
    unmatched_info = {
        "only_pre": [pre_map[k] for k in only_pre],
        "only_post": [post_map[k] for k in only_post],
    }
    with open(os.path.join(OUT_DIR, "unmatched.json"), "w", encoding="utf-8") as f:
        json.dump(unmatched_info, f, indent=2, ensure_ascii=False)

    print("\n全部处理完成")
    print(f"结果输出目录: {OUT_DIR}")
    print(f"总表: {summary_csv}")


if __name__ == "__main__":
    main()