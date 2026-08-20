"""Preprocess CBIS-DDSM into single-view and CC/MLO paired CSV files.

Outputs:
- breast_cancer_train.csv / breast_cancer_test.csv
- breast_cancer_train_cc_mlo_pair.csv / breast_cancer_test_cc_mlo_pair.csv
- RGB PNG images and binary mask tensors at configurable resolution (default 1520x912).
"""
import argparse
import os
import re

# ====================== 全局可配置参数（训练/部署仅修改此处） ======================
# 数据集输出根目录
ROOT_OUTPUT_DIR = "data/processed/ddsm"
# 图像/掩码子目录
SUB_TRAIN_IMAGE = "breast_cancer_roi_rgb_train"
SUB_TEST_IMAGE = "breast_cancer_roi_rgb_test"
SUB_TRAIN_MASK = "breast_cancer_mask_train"
SUB_TEST_MASK = "breast_cancer_mask_test"
# 原始DDSM数据集根目录 & 标签CSV
ROOT_DCM_FOLDER = "data/raw/CBIS-DDSM"
TRAIN_CSV_LIST = []
TEST_CSV_LIST = []
# 输出CSV路径（单样本表 + 双视角配对表）
OUTPUT_TRAIN_CSV = os.path.join(ROOT_OUTPUT_DIR, "breast_cancer_train.csv")
OUTPUT_TEST_CSV = os.path.join(ROOT_OUTPUT_DIR, "breast_cancer_test.csv")
OUTPUT_TRAIN_PAIR_CSV = os.path.join(ROOT_OUTPUT_DIR, "breast_cancer_train_cc_mlo_pair.csv")
OUTPUT_TEST_PAIR_CSV = os.path.join(ROOT_OUTPUT_DIR, "breast_cancer_test_cc_mlo_pair.csv")

# 标签映射
LABEL_MAPPING = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1
}
# 医学参数：乳腺钼靶背景对比度、目标尺寸、DICOM窗宽窗位
BACKGROUND_CONTRAST = 0.2    # 提升背景亮度，保留解剖结构
TARGET_SIZE_H = 1520          # 默认 Mammo-CLIP 高度 (BiomedCLIP 用 224)
TARGET_SIZE_W = 912           # 默认 Mammo-CLIP 宽度 (BiomedCLIP 用 224)
DCM_WINDOW_CENTER = 1024     # 乳腺专用窗位
DCM_WINDOW_WIDTH = 4096      # 乳腺专用窗宽
# =================================================================================

def init_folder(folder_path):
    """初始化文件夹，不存在则创建"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[INFO] 创建文件夹：{folder_path}")

def find_target_dcm(directory, target_dcm="1-1.dcm"):
    """递归查找指定名称DICOM文件"""
    try:
        for entry in os.listdir(directory):
            entry_path = os.path.join(directory, entry)
            if os.path.isdir(entry_path):
                res = find_target_dcm(entry_path, target_dcm)
                if res:
                    return res
            elif entry == target_dcm:
                return entry_path
        return None
    except Exception as e:
        print(f"[WARN] 查找DCM失败 {directory} | 错误：{str(e)}")
        return None

def dcm2array_with_window(dcm_path):
    """原始最优版本：全局归一化，保留全部乳腺纹理细节"""
    try:
        ds = pydicom.dcmread(dcm_path)
        pixel_array = ds.pixel_array.astype(np.float32)
        # 纯全局归一化，不做任何窗宽截断、局部映射
        if pixel_array.max() != pixel_array.min():
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        return pixel_array.astype(np.uint8)
    except Exception as e:
        print(f"读取DCM失败：{dcm_path}，错误：{e}")
        return None
  

def resize_to_target(array_2d, target_size=(TARGET_SIZE_H, TARGET_SIZE_W)):
    """图像/掩码统一缩放到{TARGET_SIZE_H}x{TARGET_SIZE_W}"""
    try:
        img = Image.fromarray(array_2d)
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img_resized).astype(np.uint8)
    except Exception as e:
        print(f"[WARN] 缩放图像失败 | 错误：{str(e)}")
        return None

def resize_mask(mask_array, target_shape=(TARGET_SIZE_H, TARGET_SIZE_W)):
    """掩码缩放 + 二值化（仅保留0/1）"""
    try:
        mask_resized = resize_to_target(mask_array, target_shape)
        mask_resized = (mask_resized > 0).astype(np.uint8)
        return mask_resized
    except Exception as e:
        print(f"[WARN] 缩放掩码失败 | 错误：{str(e)}")
        return None
        
def mask_fusion(image_array, mask_array):
    """原始最优融合逻辑：病灶高亮 + 背景压暗 + 保留纹理"""
    # 统一缩放到{TARGET_SIZE_H}x{TARGET_SIZE_W}
    img_resized = resize_to_target(image_array)
    if img_resized is None:
        return None
    mask_resized = resize_mask(mask_array, (TARGET_SIZE_H, TARGET_SIZE_W))
    if mask_resized is None:
        return None

    mask_binary = (mask_resized > 0).astype(np.uint8)
    # 病灶完全保留原图，背景降低对比度
    fused_array = img_resized * mask_binary + img_resized * (1 - mask_binary) * BACKGROUND_CONTRAST

    # 融合后再次全局归一化，保证动态范围正常
    if fused_array.max() != fused_array.min():
        fused_array = (fused_array - fused_array.min()) / (fused_array.max() - fused_array.min()) * 255

    return fused_array.astype(np.uint8)
    

def save_rgb_image(array_2d, save_folder, image_name, is_relative=True):
    """保存{TARGET_SIZE_H}x{TARGET_SIZE_W} RGB图像，返回相对路径"""
    try:
        array_rgb = np.stack([array_2d, array_2d, array_2d], axis=-1)
        abs_save_path = os.path.join(save_folder, f"{image_name}.png").replace("\\", "/")
        Image.fromarray(array_rgb).save(abs_save_path)
        if is_relative:
            rel_path = os.path.relpath(abs_save_path, ROOT_OUTPUT_DIR).replace("\\", "/")
            return rel_path
        return abs_save_path
    except Exception as e:
        print(f"[WARN] 保存图像失败 {image_name} | 错误：{str(e)}")
        return None

def save_aligned_mask(mask_array, save_folder, image_name, is_relative=True):
    """保存{TARGET_SIZE_H}x{TARGET_SIZE_W}二值掩码张量(.pt)，返回相对路径"""
    try:
        mask_resized = resize_mask(mask_array, (TARGET_SIZE_H, TARGET_SIZE_W))
        mask_tensor = torch.from_numpy(mask_resized).to(torch.uint8)
        abs_save_path = os.path.join(save_folder, f"{image_name}_mask.pt").replace("\\", "/")
        torch.save(mask_tensor, abs_save_path)
        if is_relative:
            rel_path = os.path.relpath(abs_save_path, ROOT_OUTPUT_DIR).replace("\\", "/")
            return rel_path
        return abs_save_path
    except Exception as e:
        print(f"[WARN] 保存掩码失败 {image_name} | 错误：{str(e)}")
        return None

def clean_path_str(path_str):
    """清洗路径字符串：去除换行、空格、引号"""
    return re.sub(r'[\n\s"]', '', path_str).strip()

def generate_cc_mlo_pair(df, output_pair_csv):
    """
    基于单样本CSV，生成【同一乳房CC+MLO配对表】
    用于双视角特征融合、投影一致性损失计算
    """
    pair_dict = {
        "side_id": [],
        "cc_image_path": [],
        "cc_mask_path": [],
        "cc_description": [],
        "mlo_image_path": [],
        "mlo_mask_path": [],
        "mlo_description": [],
        "label": []
    }
    # 按side_id分组（同一个乳房）
    side_groups = df.groupby("side_id")
    for side_id, group in side_groups:
        cc_data = group[group["view"] == "CC"]
        mlo_data = group[group["view"] == "MLO"]
        # 仅保留同时存在CC和MLO视图的样本（有效配对）
        if len(cc_data) == 0 or len(mlo_data) == 0:
            continue
        # 取第一条病灶配对（多病灶可按需扩展）
        cc_row = cc_data.iloc[0]
        mlo_row = mlo_data.iloc[0]
        # 标签一致性校验
        if cc_row["label"] != mlo_row["label"]:
            continue

        pair_dict["side_id"].append(side_id)
        pair_dict["cc_image_path"].append(cc_row["image_path"])
        pair_dict["cc_mask_path"].append(cc_row["mask_path"])
        pair_dict["cc_description"].append(cc_row["description"])
        pair_dict["mlo_image_path"].append(mlo_row["image_path"])
        pair_dict["mlo_mask_path"].append(mlo_row["mask_path"])
        pair_dict["mlo_description"].append(mlo_row["description"])
        pair_dict["label"].append(cc_row["label"])

    pair_df = pd.DataFrame(pair_dict)
    pair_df.to_csv(output_pair_csv, index=False, encoding="utf-8")
    print(f"[INFO] 双视角配对表保存完成：{output_pair_csv}，有效配对数：{pair_df.shape[0]}")
    return pair_df

def process_dataset(csv_list, output_image_subdir, output_mask_subdir, output_csv_path, output_pair_csv):
    """
    主处理函数：单样本处理 + 自动生成CC/MLO配对表
    """
    output_image_folder = os.path.join(ROOT_OUTPUT_DIR, output_image_subdir)
    output_mask_folder = os.path.join(ROOT_OUTPUT_DIR, output_mask_subdir)
    init_folder(output_image_folder)
    init_folder(output_mask_folder)

    # 读取原始CSV
    df_list = []
    for csv in csv_list:
        if os.path.exists(csv):
            df = pd.read_csv(csv, encoding="utf-8")
            df_list.append(df)
    if not df_list:
        raise ValueError("未读取到有效原始CSV文件")
    df = pd.concat(df_list, ignore_index=True)

    # 结果存储字典（单样本表）
    result_dict = {
        "image_name": [],
        "side_id": [],
        "view": [],
        "image_path": [],
        "mask_path": [],
        "description": [],
        "pathology_org": [],
        "label": []
    }
    processed_names = set()  # 按唯一文件名去重（修复原逻辑Bug）

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"处理{output_image_subdir}"):
        # 解析基础字段
        core_fields = {
            "patient_id": row.get("patient_id", "unknown"),
            "breast_density": row.get("breast density", "unknown"),
            "left_right": row.get("left or right breast", "unknown").lower(),
            "image_view": row.get("image view", "unknown"),
            "abn_id": row.get("abnormality id", "unknown"),
            "abn_type": row.get("abnormality type", "unknown"),
            "pathology": row.get("pathology", "unknown").upper(),
            "subtlety": row.get("subtlety", "unknown"),
            "assessment": row.get("assessment", "unknown"),
            "calc_type": row.get("calc type", ""),
            "calc_dist": row.get("calc distribution", ""),
            "mass_shape": row.get("mass shape", ""),
            "mass_marg": row.get("mass margins", ""),
            "mass_dist": row.get("mass distribution", "")
        }

        # 过滤无效标签
        if core_fields["pathology"] not in LABEL_MAPPING:
            continue
        label = LABEL_MAPPING[core_fields["pathology"]]

        # 拼接精细化医学描述文本（用于文本提示学习）
        desc_parts = [
            f"breast density is {core_fields['breast_density']}",
            f"{core_fields['left_right']} breast",
            f"{core_fields['image_view']} view",
            f"abnormality id {core_fields['abn_id']}",
            f"abnormality type {core_fields['abn_type']}"
        ]
        if core_fields["abn_type"].lower() == "calcification" and core_fields["calc_type"] and core_fields["calc_dist"]:
            desc_parts.append(f"{core_fields['calc_type']} type")
            desc_parts.append(f"{core_fields['calc_dist']} distribution")
        elif core_fields["abn_type"].lower() == "mass":
            if core_fields["mass_shape"]: desc_parts.append(f"{core_fields['mass_shape']} shape")
            if core_fields["mass_marg"]: desc_parts.append(f"{core_fields['mass_marg']} margins")
            if core_fields["mass_dist"]: desc_parts.append(f"{core_fields['mass_dist']} distribution")
        desc_parts.extend([f"assessment {core_fields['assessment']}", f"subtlety {core_fields['subtlety']}"])
        final_desc = ", ".join([p for p in desc_parts if p not in ["unknown", ""]])

        # 清洗路径
        image_path_str = clean_path_str(row.get("image file path", ""))
        mask_path_str = clean_path_str(row.get("ROI mask file path", ""))
        if not image_path_str or not mask_path_str:
            continue

        # 生成唯一文件名
        abn_prefix = "Calc" if core_fields["abn_type"].lower() == "calcification" else "Mass"
        image_name = f"{abn_prefix}-{core_fields['patient_id']}-{core_fields['left_right']}-{core_fields['image_view']}-abn{core_fields['abn_id']}"
        side_id = f"{core_fields['patient_id']}-{core_fields['left_right']}"
        view = core_fields["image_view"].upper()

        # 按文件名去重（修复原Bug：允许同一视角多个病灶）
        if image_name in processed_names:
            continue
        processed_names.add(image_name)

        # 查找DICOM文件
        full_image_dir = os.path.join(ROOT_DCM_FOLDER, image_path_str.split("/")[0])
        full_mask_dir = os.path.join(ROOT_DCM_FOLDER, mask_path_str.split("/")[0])
        dcm_image = find_target_dcm(full_image_dir, "1-1.dcm")
        dcm_mask = find_target_dcm(full_mask_dir, "1-2.dcm")
        if not dcm_image or not dcm_mask:
            continue

        # 读取并融合图像
        img_array = dcm2array_with_window(dcm_image)
        mask_array = dcm2array_with_window(dcm_mask)
        if img_array is None or mask_array is None:
            continue
        fused_array = mask_fusion(img_array, mask_array)
        if fused_array is None:
            continue

        # 保存图像与掩码
        rel_img_path = save_rgb_image(fused_array, output_image_folder, image_name)
        rel_mask_path = save_aligned_mask(mask_array, output_mask_folder, image_name)
        if not rel_img_path or not rel_mask_path:
            continue

        # 写入单样本数据
        result_dict["image_name"].append(image_name)
        result_dict["side_id"].append(side_id)
        result_dict["view"].append(view)
        result_dict["image_path"].append(rel_img_path)
        result_dict["mask_path"].append(rel_mask_path)
        result_dict["description"].append(final_desc)
        result_dict["pathology_org"].append(core_fields["pathology"])
        result_dict["label"].append(label)

    # 保存单样本CSV
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    benign_num = sum(result_df["label"] == 0)
    mal_num = sum(result_df["label"] == 1)
    print(f"[INFO] 单样本表保存完成：{output_csv_path}")
    print(f"[INFO] 有效样本总数：{result_df.shape[0]} | 良性：{benign_num} | 恶性：{mal_num}\n")

    # 生成CC/MLO双视角配对表
    generate_cc_mlo_pair(result_df, output_pair_csv)
    return result_df

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CBIS-DDSM for CDF-VLM DDSM experiments.")
    parser.add_argument("--root-dcm-folder", required=True, help="Root folder containing CBIS-DDSM DICOM subfolders.")
    parser.add_argument("--output-dir", default=ROOT_OUTPUT_DIR)
    parser.add_argument("--train-csv", nargs="+", required=True, help="Train calc/mass description CSV files.")
    parser.add_argument("--test-csv", nargs="+", required=True, help="Test calc/mass description CSV files.")
    parser.add_argument("--target-size-h", type=int, default=1520, help="Output image height (Mammo-CLIP: 1520, BiomedCLIP: 224)")
    parser.add_argument("--target-size-w", type=int, default=912, help="Output image width (Mammo-CLIP: 912, BiomedCLIP: 224)")
    parser.add_argument("--background-contrast", type=float, default=BACKGROUND_CONTRAST)
    return parser.parse_args()


def configure_from_args(args):
    global ROOT_OUTPUT_DIR, ROOT_DCM_FOLDER, TRAIN_CSV_LIST, TEST_CSV_LIST
    global OUTPUT_TRAIN_CSV, OUTPUT_TEST_CSV, OUTPUT_TRAIN_PAIR_CSV, OUTPUT_TEST_PAIR_CSV
    global TARGET_SIZE_H, TARGET_SIZE_W, BACKGROUND_CONTRAST

    ROOT_OUTPUT_DIR = args.output_dir
    ROOT_DCM_FOLDER = args.root_dcm_folder
    TRAIN_CSV_LIST = args.train_csv
    TEST_CSV_LIST = args.test_csv
    TARGET_SIZE_H = args.target_size_h
    TARGET_SIZE_W = args.target_size_w
    BACKGROUND_CONTRAST = args.background_contrast
    OUTPUT_TRAIN_CSV = os.path.join(ROOT_OUTPUT_DIR, "breast_cancer_train.csv")
    OUTPUT_TEST_CSV = os.path.join(ROOT_OUTPUT_DIR, "breast_cancer_test.csv")
    OUTPUT_TRAIN_PAIR_CSV = os.path.join(ROOT_OUTPUT_DIR, "breast_cancer_train_cc_mlo_pair.csv")
    OUTPUT_TEST_PAIR_CSV = os.path.join(ROOT_OUTPUT_DIR, "breast_cancer_test_cc_mlo_pair.csv")


def main():
    args = parse_args()
    configure_from_args(args)
    global pd, np, pydicom, torch, Image, tqdm
    import numpy as np
    import pandas as pd
    import pydicom
    import torch
    from PIL import Image
    from tqdm import tqdm

    try:
        init_folder(ROOT_OUTPUT_DIR)
        # 处理训练集
        process_dataset(
            TRAIN_CSV_LIST,
            SUB_TRAIN_IMAGE,
            SUB_TRAIN_MASK,
            OUTPUT_TRAIN_CSV,
            OUTPUT_TRAIN_PAIR_CSV
        )
        # 处理测试集
        process_dataset(
            TEST_CSV_LIST,
            SUB_TEST_IMAGE,
            SUB_TEST_MASK,
            OUTPUT_TEST_CSV,
            OUTPUT_TEST_PAIR_CSV
        )

        print("=" * 70)
        print("✅ DDSM数据集预处理全部完成！")
        print(f"数据集根目录：{ROOT_OUTPUT_DIR}")
        print("输出文件说明：")
        print("1. breast_cancer_train/test.csv  → 单视角样本表（通用分类/提示学习）")
        print("2. breast_cancer_*_cc_mlo_pair.csv → CC/MLO双视角配对表（几何融合/投影损失专用）")
        print(f"3. 图像：{TARGET_SIZE_H}x{TARGET_SIZE_W} RGB | 掩码：{TARGET_SIZE_H}x{TARGET_SIZE_W} 二值.pt张量")
        print("4. 所有路径为相对路径，服务器仅需修改ROOT_OUTPUT_DIR")
        print("=" * 70)

    except Exception as e:
        print(f"[ERROR] 程序运行失败：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
