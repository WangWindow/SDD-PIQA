# SDD-PIQA: Unsupervised Palmprint Image Quality Assessment with Similarity Distribution Distance
# SDD-PIQA: 基于相似度分布距离的无监督掌纹图像质量评估

This repository contains the implementation of SDD-PIQA, adapted for Palmprint Image Quality Assessment.
本项目包含 SDD-PIQA 的实现代码，适用于掌纹图像质量评估。

## 🛠️ Prerequisites / 准备工作
*   Python >= 3.10
*   PyTorch >= 2.*
*   Torchvision (match PyTorch version)
*   Numpy, Scipy, Tqdm, PIL ...

根据 `pyproject.toml` 安装项目依赖。

## 🚀 Usage / 使用方法

> 注意，为了避免路径格式不统一导致无法运行，本项目中所有路径使用绝对路径。

> 代码中的项目路径为 `/root/workspace/SDD-PIQA`。

> 可以使用文件搜索工具（如 `VS Code` 的搜索功能），将项目文件中的 `/root/workspace/SDD-PIQA` 全部替换为 **`{自己的项目绝对路径}`**

### 1. Get a Palmprint Recognition Model / 获取掌纹识别模型
**(Optional / 可选)**
If you already have a pre-trained recognition model, you can skip this step. Otherwise, you can train a simple ResNet50 model using the provided script.
如果您已有预训练的识别模型，可跳过此步。否则，您可以使用提供的脚本训练一个简单的 ResNet50 模型。

```bash
# Train the recognition model / 训练识别模型
python utils/train_recognition/train_recognition.py
```
*   The model will be saved at: `checkpoints/recognition_model/palmprint_R50_backbone_best.pth`
*   模型将保存于上述路径。

### 2. Generation of Quality Pseudo-Labels / 生成质量伪标签

#### Step 1: Generate Data List / 生成数据列表
Generate the image list and label files from your dataset.
从您的数据集生成图像列表和标签文件。
```bash
python gen_pseudo_labels/gen_datalist.py
```
*   **Input**: `data/ROI_Data` (Configure in script / 在脚本中配置)
*   **Output**: `gen_pseudo_labels/features/DATA.label`, `gen_pseudo_labels/features/DATA.labelpath`

#### Step 2: Extract Embeddings / 提取特征
Extract palmprint features using the recognition model.
使用识别模型提取掌纹特征。
```bash
# Ensure configuration is correct in the script
# 确保脚本中的配置正确
python gen_pseudo_labels/extract_feats.py
```
*   **Output**: `gen_pseudo_labels/features/features.npy`

#### Step 3: Calculate Pseudo-Labels / 计算伪标签
Calculate quality scores based on the distribution distance of intra-class and inter-class similarities.
基于类内和类间相似度的分布距离计算质量分数。
```bash
python gen_pseudo_labels/gen_pseudo_labels.py
```
*   **Output**: `gen_pseudo_labels/annotations/quality_pseudo_labels.txt`

### 3. Training of Quality Regression Model / 训练质量回归模型

1.  **Configure / 配置**: Modify `train.py` to set your data paths (e.g., `img_list`, `data_root`).
    修改 `train.py` 设置数据路径。
2.  **Train / 训练**:

```bash
# Run directly / 直接运行
python train.py

# Or run in background / 后台运行
bash scripts/train.sh
```
*   **Checkpoints**: Saved in `checkpoints/quality_model`

### 4. Prediction / 预测
Use the trained model to predict quality scores for new images.
使用训练好的模型预测新图像的质量分数。

```bash
python eval.py
```

## 📂 Project Structure / 项目结构
*   `data/`: Dataset folder / 数据集目录
*   `gen_pseudo_labels/`: Code for pseudo-label generation / 伪标签生成的代码
*   `train.py`: Main training script / 主训练脚本
*   `eval.py`: Evaluation script / 评估脚本
*   `checkpoints/`: Saved models / 保存的模型
*   `scripts/`: Shell scripts run in silence / 静默运行的 Shell 脚本
