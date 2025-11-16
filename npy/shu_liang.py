import numpy as np

feats = np.load("D:/IQA/da_fen_zhangjm/quality/npy/npy.npy")
print(f"特征数量: {feats.shape[0]}")  # 目前是 14098

# 原始 list 文件
with open("D:/IQA/da_fen_zhangjm/quality/generate_pseudo_labels/DATA.label", "r") as f:
    label_list = [line.strip().split()[0] for line in f]

# 特征文件
feats = np.load("D:/IQA/da_fen_zhangjm/quality/npy/npy.npy")
print("label 文件数:", len(label_list))
print("特征数:", feats.shape[0])

# 比对
if len(label_list) != feats.shape[0]:
    print("开始检查丢失的图像")
    for i in range(len(feats), len(label_list)):
        print(f"丢失：{label_list[i]}")
