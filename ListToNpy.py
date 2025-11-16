import numpy as np


def convert_list_to_npy(list_file_path, npy_file_path):
    print("Loading feature list from:", list_file_path)

    # 加载 feature_list
    feature_list = np.load(list_file_path, allow_pickle=True)
    print("Feature list loaded. Length:", len(feature_list))

    # 检查是否为空
    if len(feature_list) == 0:
        print("Feature list is empty! Exiting...")
        return

    # 拼接数组
    print("Concatenating feature list...")
    feature_array = np.concatenate([f for f in feature_list], axis=0)
    print("Feature array shape:", feature_array.shape)

    # 保存为 .npy 文件
    print("Saving .npy file to:", npy_file_path)
    np.save(npy_file_path, feature_array)
    print(".npy file successfully saved!")


if __name__ == "__main__":
    # list_file_path = r'D:\IQA\da_fen_zhangjm\quality\feature_list_n'  # 替换为你的 list 文件路径
    list_file_path = r"D:\IQA\da_fen_zhangjm\TexDirNet\feature\feature_list"  # 替换为你的 list 文件路径
    npy_file_path = (
        r"D:\IQA\da_fen_zhangjm\quality\npy\npy"  # 替换为你想保存的 npy 文件路径
    )

    convert_list_to_npy(list_file_path, npy_file_path)
