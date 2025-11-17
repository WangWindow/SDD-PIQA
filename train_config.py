import torch
import torchvision.transforms as T


class Config:
    # dataset
    img_list = "/root/workplace/SDD-FIQA/generate_pseudo_labels/annotations/quality_pseudo_labels.txt"
    data_root = "/root/workplace/SDD-FIQA/data/ROI_Data"  # 新增 data_root 属性
    # 使用掌纹识别阶段训练得到的 R50 backbone 作为初始化(可设为 None 从头训练)
    finetuning_model = "/root/workplace/SDD-FIQA/generate_pseudo_labels/extract_embedding/model/palmprint_R50_backbone.pth"
    # save settings
    checkpoints = "/root/workplace/SDD-FIQA/checkpoints/quality_model"
    checkpoints_name = "R50"
    # data preprocess
    transform = T.Compose(
        [
            T.Resize((112, 112)),
            # 掌纹左右手不对称，去除水平翻转以免引入跨手噪声
            T.Grayscale(num_output_channels=3),  # 转为 3 通道以兼容预训练卷积
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # training settings
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 0
    # multi_GPUs = [0,1,2,3,4,5,6,7]
    multi_GPUs = [0]
    backbone = "R_50"  # 使用与识别一致的 IR50 结构
    pin_memory = True
    # num_workers = 12
    num_workers = 4
    # 依据显存调整 batch_size，1 过低影响 BN；64 是较常见起点
    batch_size = 64
    epoch = 20
    lr = 0.0001
    stepLR = [5, 10]
    weight_decay = 0.0005
    display = 100
    saveModel_epoch = 1
    loss = "SmoothL1"  # ['L1', 'L2', 'SmoothL1']


config = Config()
