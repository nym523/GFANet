# -*- coding: utf-8 -*-
"""
bitemporal_gev_lrt_batch.py

功能：
1. 批量读取双时相图像与共享区域标签图
2. 在每个区域内分别对 pre / post / merged 样本拟合 GEV 分布
3. 计算区域级对数似然比统计量 D_m
4. 为每个样本生成：
   - 每个区域的 GEV 参数与统计量表格 csv
   - 区域级统计量字典 npz
   - 区域级统计图（label map -> D map）
   - 可选二值变化图
   - 运行摘要 json
5. 生成整个批次的总表 batch_summary.csv

统计量定义：
    H0: theta_pre = theta_post = theta_0
    H1: theta_pre != theta_post

    D_m = 2 * [ ll(theta_pre_hat; x_m)
              + ll(theta_post_hat; y_m)
              - ll(theta_0_hat; z_m) ]

其中：
    x_m: 第 m 个区域的 pre 像素集合
    y_m: 第 m 个区域的 post 像素集合
    z_m: 合并样本 = [x_m, y_m]

注意：
1. scipy.stats.genextreme 的 shape 参数记为 c，
   与很多论文中的 GEV 形状参数 k 常有符号关系：k = -c
2. 若区域像素太少、区域全常数、或拟合失败，会跳过或做稳健处理
3. 本脚本默认适用于单通道图像；若输入是 RGB，会自动转灰度
4. 该版本支持整个文件夹批处理，按“同名文件”配对
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import cv2

from scipy.stats import genextreme
from scipy.optimize import OptimizeWarning


# =========================================================
# 参数配置区（直接修改这里）
# =========================================================

# ---------- 输入输出文件夹 ----------
PRE_DIR = r"/home/f523/disk2/nym/open-cd-main/data/JL1-CD/train/A"
POST_DIR = r"/home/f523/disk2/nym/open-cd-main/data/JL1-CD/train/B"
LABEL_DIR = r"/home/f523/disk2/nym/open-cd-main/data/JL1-CD/shared_labels_merged-500"
OUT_DIR = r"/home/f523/disk2/nym/open-cd-main/data/JL1-CD/GEV-out-250"

# ---------- 文件匹配设置 ----------
PRE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
POST_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
LABEL_EXTS = (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# 若 label 图像文件名和 pre/post 主文件名一致，则 KEEP_LABEL_SUFFIX=""
# 例如：
# pre:   001.png
# post:  001.png
# label: 001.npy
#
# 若 label 比主文件名多后缀，例如：
# pre:   001.png
# post:  001.png
# label: 001_label.npy
# 则设置为：
# LABEL_SUFFIX = "_label"
LABEL_SUFFIX = ""

# ---------- 核心参数 ----------
MIN_REGION_SIZE = 1200
IGNORE_LABEL = None
THRESHOLD = None          # None 表示每个样本自动用 Otsu
RANDOM_SEED = 42

# ---------- 可视化参数 ----------
CLIP_PERCENTILE = (1, 99)

# ---------- 批处理控制 ----------
SORT_FILES = True
SKIP_IF_MISSING = True    # True: 缺失配对文件则跳过；False: 直接报错停止


# =========================
# 工具函数
# =========================

def is_file(path):
    return os.path.isfile(path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_image(path, grayscale=True):
    """
    支持中文路径读取图像。
    grayscale=True 时读成单通道；否则按原通道读取。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    if not os.path.isfile(path):
        raise ValueError(f"输入不是文件: {path}")

    data = np.fromfile(path, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    img = cv2.imdecode(data, flag)

    if img is None:
        raise ValueError(f"图像读取失败: {path}")

    return img


def ensure_gray(img):
    """
    若输入为彩色图像，转换为灰度。
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"不支持的图像维度: {img.shape}")


def save_image(path, img):
    """
    支持中文路径保存图像。
    """
    ext = os.path.splitext(path)[1]
    ok, buffer = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"图像编码失败: {path}")
    buffer.tofile(path)


def robust_normalize_to_uint8(arr, clip_percentile=(1, 99)):
    """
    将任意实数数组稳健拉伸到 [0,255]，用于可视化。
    """
    arr = arr.astype(np.float64)
    valid = np.isfinite(arr)

    out = np.zeros_like(arr, dtype=np.uint8)
    if valid.sum() == 0:
        return out

    vals = arr[valid]
    lo = np.percentile(vals, clip_percentile[0])
    hi = np.percentile(vals, clip_percentile[1])

    if hi <= lo:
        hi = vals.max()
        lo = vals.min()

    if hi <= lo:
        out[valid] = 0
        return out

    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0, 1)
    out[valid] = (scaled[valid] * 255).astype(np.uint8)
    return out


def preprocess_region_samples(x, eps=1e-6):
    """
    区域样本预处理：
    1. 转 float64
    2. 去掉 nan / inf
    3. 若数据几乎为常数，加入极小扰动避免 GEV 拟合退化
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return x

    if np.std(x) < eps:
        x = x + np.random.normal(0.0, eps, size=x.shape)

    return x


def fit_gev_mle(samples):
    """
    对样本做 GEV MLE 拟合。
    返回：
        dict:
            success: bool
            c: scipy shape
            loc: location
            scale: scale
            k_paper: 论文常见形状参数，定义为 -c
            loglik: 对数似然
            n: 样本数
            msg: 状态信息
    """
    x = preprocess_region_samples(samples)

    if x.size < 3:
        return {
            "success": False,
            "c": np.nan,
            "loc": np.nan,
            "scale": np.nan,
            "k_paper": np.nan,
            "loglik": np.nan,
            "n": int(x.size),
            "msg": "too_few_samples"
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=OptimizeWarning)

            c, loc, scale = genextreme.fit(x)

        if not np.isfinite(scale) or scale <= 0:
            return {
                "success": False,
                "c": np.nan,
                "loc": np.nan,
                "scale": np.nan,
                "k_paper": np.nan,
                "loglik": np.nan,
                "n": int(x.size),
                "msg": "invalid_scale"
            }

        logpdf = genextreme.logpdf(x, c=c, loc=loc, scale=scale)

        if np.any(~np.isfinite(logpdf)):
            return {
                "success": False,
                "c": np.nan,
                "loc": np.nan,
                "scale": np.nan,
                "k_paper": np.nan,
                "loglik": np.nan,
                "n": int(x.size),
                "msg": "non_finite_logpdf"
            }

        loglik = float(np.sum(logpdf))

        return {
            "success": True,
            "c": float(c),
            "loc": float(loc),
            "scale": float(scale),
            "k_paper": float(-c),
            "loglik": loglik,
            "n": int(x.size),
            "msg": "ok"
        }

    except Exception as e:
        return {
            "success": False,
            "c": np.nan,
            "loc": np.nan,
            "scale": np.nan,
            "k_paper": np.nan,
            "loglik": np.nan,
            "n": int(x.size),
            "msg": f"fit_error: {str(e)}"
        }


def compute_region_lrt(pre_samples, post_samples):
    """
    对单个区域计算：
    - pre GEV 拟合
    - post GEV 拟合
    - merged GEV 拟合
    - D = 2 * (ll_pre + ll_post - ll_0)

    返回：
        result(dict)
    """
    x = preprocess_region_samples(pre_samples)
    y = preprocess_region_samples(post_samples)
    z = np.concatenate([x, y], axis=0) if x.size > 0 and y.size > 0 else np.array([], dtype=np.float64)

    fit_pre = fit_gev_mle(x)
    fit_post = fit_gev_mle(y)
    fit_0 = fit_gev_mle(z)

    valid = fit_pre["success"] and fit_post["success"] and fit_0["success"]

    if valid:
        D = 2.0 * (fit_pre["loglik"] + fit_post["loglik"] - fit_0["loglik"])
        D = max(0.0, float(D))
        status = "ok"
    else:
        D = np.nan
        status = "failed"

    result = {
        "n_pre": int(x.size),
        "n_post": int(y.size),
        "n_merged": int(z.size),

        "pre_success": fit_pre["success"],
        "pre_c": fit_pre["c"],
        "pre_k_paper": fit_pre["k_paper"],
        "pre_loc": fit_pre["loc"],
        "pre_scale": fit_pre["scale"],
        "pre_loglik": fit_pre["loglik"],
        "pre_msg": fit_pre["msg"],

        "post_success": fit_post["success"],
        "post_c": fit_post["c"],
        "post_k_paper": fit_post["k_paper"],
        "post_loc": fit_post["loc"],
        "post_scale": fit_post["scale"],
        "post_loglik": fit_post["loglik"],
        "post_msg": fit_post["msg"],

        "null_success": fit_0["success"],
        "null_c": fit_0["c"],
        "null_k_paper": fit_0["k_paper"],
        "null_loc": fit_0["loc"],
        "null_scale": fit_0["scale"],
        "null_loglik": fit_0["loglik"],
        "null_msg": fit_0["msg"],

        "D": D,
        "status": status
    }

    return result


def otsu_threshold_from_valid(values):
    """
    对一维有效统计量做 Otsu 阈值。
    返回阈值（原始量纲，不是 0-255）。
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    if values.size == 1:
        return float(values[0])

    vals_u8 = robust_normalize_to_uint8(values, clip_percentile=CLIP_PERCENTILE)
    thr_u8, _ = cv2.threshold(vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    vmin = values.min()
    vmax = values.max()
    if vmax <= vmin:
        return float(vmin)

    thr = vmin + (thr_u8 / 255.0) * (vmax - vmin)
    return float(thr)


def list_files_by_stem(folder, exts):
    """
    扫描文件夹，返回：
        stem -> full_path
    若同名不同扩展同时存在，默认保留第一次遇到的。
    """
    file_dict = {}
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"不是有效文件夹: {folder}")

    names = os.listdir(folder)
    if SORT_FILES:
        names = sorted(names)

    for name in names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in tuple(e.lower() for e in exts):
            continue

        stem = os.path.splitext(name)[0]
        if stem not in file_dict:
            file_dict[stem] = path

    return file_dict


def find_label_for_stem(stem, label_dict, label_suffix=""):
    """
    按 stem 查找对应 label。
    例如：
        stem = "001"
        label_suffix = "_label"
        则优先找 "001_label"
    """
    key = stem + label_suffix
    if key in label_dict:
        return label_dict[key]
    return None


def process_one_sample(pre_path, post_path, label_path, out_dir,
                       min_region_size=20,
                       ignore_label=None,
                       threshold=None):
    """
    处理单个样本，并把结果保存到 out_dir。
    返回 summary(dict)。
    """
    ensure_dir(out_dir)

    # 1. 读入图像
    pre_img = read_image(pre_path, grayscale=False)
    post_img = read_image(post_path, grayscale=False)
    pre_img = ensure_gray(pre_img)
    post_img = ensure_gray(post_img)

    pre_img = pre_img.astype(np.float64)
    post_img = post_img.astype(np.float64)

    # 2. 读入区域标签图
    if label_path.lower().endswith(".npy"):
        label_map = np.load(label_path)
    else:
        label_map = read_image(label_path, grayscale=True)

    label_map = np.asarray(label_map)

    if pre_img.shape != post_img.shape:
        raise ValueError(f"前后时相尺寸不一致: pre={pre_img.shape}, post={post_img.shape}")

    if label_map.shape != pre_img.shape:
        raise ValueError(f"标签图尺寸不匹配: label={label_map.shape}, img={pre_img.shape}")

    label_map = label_map.astype(np.int32)

    unique_labels = np.unique(label_map)
    if ignore_label is not None:
        unique_labels = unique_labels[unique_labels != ignore_label]

    # 3. 遍历区域拟合
    results = []
    d_map = np.full(label_map.shape, np.nan, dtype=np.float64)

    print(f"[INFO] 当前样本区域总数: {len(unique_labels)}")

    for region_id in unique_labels:
        mask = (label_map == region_id)
        area = int(mask.sum())

        if area < min_region_size:
            results.append({
                "region_id": int(region_id),
                "area": area,
                "status": "too_small",
                "D": np.nan
            })
            continue

        x_m = pre_img[mask]
        y_m = post_img[mask]

        region_result = compute_region_lrt(x_m, y_m)
        region_result["region_id"] = int(region_id)
        region_result["area"] = area
        results.append(region_result)

        if np.isfinite(region_result["D"]):
            d_map[mask] = region_result["D"]

    # 4. 保存区域表
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "region_gev_lrt_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 5. 保存 npz
    npz_path = os.path.join(out_dir, "region_gev_lrt_results.npz")
    np.savez_compressed(
        npz_path,
        label_map=label_map,
        d_map=d_map
    )

    # 6. 可视化
    d_vis = d_map.copy()
    invalid_mask = ~np.isfinite(d_vis)

    if np.all(invalid_mask):
        d_vis_u8 = np.zeros_like(label_map, dtype=np.uint8)
    else:
        d_vis_u8 = robust_normalize_to_uint8(d_vis, clip_percentile=CLIP_PERCENTILE)
        d_vis_u8[invalid_mask] = 0

    d_vis_path = os.path.join(out_dir, "region_D_map.png")
    save_image(d_vis_path, d_vis_u8)

    heatmap = cv2.applyColorMap(d_vis_u8, cv2.COLORMAP_JET)
    heatmap[invalid_mask] = 0
    heatmap_path = os.path.join(out_dir, "region_D_map_heatmap.png")
    save_image(heatmap_path, heatmap)

    # 7. 二值变化图
    valid_D = df["D"].values if "D" in df.columns else np.array([])
    valid_D = valid_D[np.isfinite(valid_D)]

    if valid_D.size > 0:
        if threshold is not None:
            used_threshold = float(threshold)
            threshold_type = "manual"
        else:
            used_threshold = otsu_threshold_from_valid(valid_D)
            threshold_type = "otsu"

        change_map = np.zeros(label_map.shape, dtype=np.uint8)
        change_map[np.isfinite(d_map) & (d_map >= used_threshold)] = 255

        change_map_path = os.path.join(out_dir, "region_change_map.png")
        save_image(change_map_path, change_map)
    else:
        used_threshold = np.nan
        threshold_type = "none"
        change_map_path = None

    # 8. 样本级摘要
    num_valid_regions = 0
    if "D" in df.columns:
        num_valid_regions = int(np.isfinite(df["D"]).sum())

    summary = {
        "pre_img": pre_path,
        "post_img": post_path,
        "label_map": label_path,
        "out_dir": out_dir,
        "num_regions_total": int(len(unique_labels)),
        "num_regions_valid_D": num_valid_regions,
        "min_region_size": int(min_region_size),
        "ignore_label": ignore_label,
        "threshold_type": threshold_type,
        "threshold_value": None if not np.isfinite(used_threshold) else float(used_threshold),
        "clip_percentile": list(CLIP_PERCENTILE)
    }

    summary_path = os.path.join(out_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] 样本结果已保存到: {out_dir}")
    return summary


def main():
    np.random.seed(RANDOM_SEED)
    ensure_dir(OUT_DIR)

    if not os.path.isdir(PRE_DIR):
        raise NotADirectoryError(f"PRE_DIR 不是有效文件夹: {PRE_DIR}")
    if not os.path.isdir(POST_DIR):
        raise NotADirectoryError(f"POST_DIR 不是有效文件夹: {POST_DIR}")
    if not os.path.isdir(LABEL_DIR):
        raise NotADirectoryError(f"LABEL_DIR 不是有效文件夹: {LABEL_DIR}")

    pre_dict = list_files_by_stem(PRE_DIR, PRE_EXTS)
    post_dict = list_files_by_stem(POST_DIR, POST_EXTS)
    label_dict = list_files_by_stem(LABEL_DIR, LABEL_EXTS)

    pre_stems = list(pre_dict.keys())
    if SORT_FILES:
        pre_stems = sorted(pre_stems)

    print(f"[INFO] pre 文件数:   {len(pre_dict)}")
    print(f"[INFO] post 文件数:  {len(post_dict)}")
    print(f"[INFO] label 文件数: {len(label_dict)}")

    batch_rows = []

    for stem in pre_stems:
        pre_path = pre_dict[stem]
        post_path = post_dict.get(stem, None)
        label_path = find_label_for_stem(stem, label_dict, label_suffix=LABEL_SUFFIX)

        if post_path is None or label_path is None:
            msg = {
                "sample_id": stem,
                "pre_path": pre_path,
                "post_path": post_path,
                "label_path": label_path,
                "status": "missing_pair"
            }
            batch_rows.append(msg)

            print(f"[WARN] 跳过样本 {stem}：缺少配对文件")
            if not SKIP_IF_MISSING:
                raise FileNotFoundError(f"样本 {stem} 缺少 post 或 label 文件")
            continue

        sample_out_dir = os.path.join(OUT_DIR, stem)
        print("=" * 60)
        print(f"[INFO] 开始处理样本: {stem}")
        print(f"[INFO] pre   : {pre_path}")
        print(f"[INFO] post  : {post_path}")
        print(f"[INFO] label : {label_path}")

        try:
            summary = process_one_sample(
                pre_path=pre_path,
                post_path=post_path,
                label_path=label_path,
                out_dir=sample_out_dir,
                min_region_size=MIN_REGION_SIZE,
                ignore_label=IGNORE_LABEL,
                threshold=THRESHOLD
            )

            batch_row = {
                "sample_id": stem,
                "pre_path": pre_path,
                "post_path": post_path,
                "label_path": label_path,
                "status": "ok",
                "num_regions_total": summary["num_regions_total"],
                "num_regions_valid_D": summary["num_regions_valid_D"],
                "threshold_type": summary["threshold_type"],
                "threshold_value": summary["threshold_value"],
                "out_dir": sample_out_dir
            }

        except Exception as e:
            batch_row = {
                "sample_id": stem,
                "pre_path": pre_path,
                "post_path": post_path,
                "label_path": label_path,
                "status": f"failed: {str(e)}",
                "num_regions_total": np.nan,
                "num_regions_valid_D": np.nan,
                "threshold_type": np.nan,
                "threshold_value": np.nan,
                "out_dir": sample_out_dir
            }
            print(f"[ERROR] 样本 {stem} 处理失败: {e}")

        batch_rows.append(batch_row)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv_path = os.path.join(OUT_DIR, "batch_summary.csv")
    batch_df.to_csv(batch_csv_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"[DONE] 批处理完成")
    print(f"[DONE] 总摘要表: {batch_csv_path}")


if __name__ == "__main__":
    main()