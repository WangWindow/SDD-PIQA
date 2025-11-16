import torch
import torchvision.transforms as T


class Config:
    # dataset
    img_list = "/root/workplace/SDD-FIQA/generate_pseudo_labels/annotations/quality_pseudo_labels.txt"
    data_root = "/root/workplace/SDD-FIQA/data/ROI_Data"  # 新增 data_root 属性
    finetuning_model = "/root/workplace/SDD-FIQA/generate_pseudo_labels/extract_embedding/model/20500backbone.pth"
    # save settings
    checkpoints = "/root/workplace/SDD-FIQA/checkpoints/MS1M_Quality_Regression/S1"
    checkpoints_name = "MFN"
    # data preprocess
    transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # training settings
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 0
    # multi_GPUs = [0,1,2,3,4,5,6,7]
    multi_GPUs = [0]
    backbone = "MFN"  # [MFN, R_50]
    pin_memory = True
    # num_workers = 12
    num_workers = 0
    # batch_size = 5000
    batch_size = 1
    epoch = 20
    lr = 0.0001
    stepLR = [5, 10]
    weight_decay = 0.0005
    display = 100
    saveModel_epoch = 1
    loss = "SmoothL1"  # ['L1', 'L2', 'SmoothL1']


config = Config()
