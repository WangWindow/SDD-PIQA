import os

# 指定根目录
root_dir = r'D:\IQA\da_fen_zhangjm\newPdata4_jm'

# 遍历所有子文件夹
for subdir in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subfolder_path):
        # 获取子文件夹中的所有图片文件
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 按文件名排序
        image_files.sort()

        # 逐个重命名
        for idx, filename in enumerate(image_files, start=1):
            old_path = os.path.join(subfolder_path, filename)
            ext = os.path.splitext(filename)[1]  # 获取原始扩展名
            new_name = f"{idx}{ext}"
            new_path = os.path.join(subfolder_path, new_name)

            # 如果新文件名已存在，先暂时重命名以避免冲突
            if os.path.exists(new_path):
                temp_path = os.path.join(subfolder_path, f"temp_{idx}{ext}")
                os.rename(old_path, temp_path)
            else:
                os.rename(old_path, new_path)

        # 第二轮处理因冲突被临时改名的文件
        temp_files = [f for f in os.listdir(subfolder_path) if f.startswith("temp_")]
        temp_files.sort()
        for idx, filename in enumerate(temp_files, start=1 + len(image_files) - len(temp_files)):
            temp_path = os.path.join(subfolder_path, filename)
            ext = os.path.splitext(filename)[1]
            final_path = os.path.join(subfolder_path, f"{idx}{ext}")
            os.rename(temp_path, final_path)

        print(f"完成重命名：{subfolder_path}")
