# import torch
# import torchvision.transforms as T


# class Config:
#     # dataset
#     # img_list = './generate_pseudo_labels/labels/pseudo_quality_score.txt'
#     img_list = './generate_pseudo_labels/labels/quality_pseudo_labels.txt'
#     # finetuning_model = './generate_pseudo_labels/extract_embedding/model/MobileFaceNet_MS1M.pth'
#     finetuning_model = './generate_pseudo_labels/extract_embedding/model/20500backbone.pth'
#     # save settings
#     checkpoints = "./checkpoints/MS1M_Quality_Regression/S1"
#     checkpoints_name = "MFN"
#     # data preprocess
#     transform = T.Compose([
#         T.RandomHorizontalFlip(),
#         T.ToTensor(),
#         T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ])
#     # training settings
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     seed = 0
#     # multi_GPUs = [0,1,2,3,4,5,6,7]
#     multi_GPUs = [0]
#     backbone = 'MFN'  # [MFN, R_50]
#     pin_memory = True
#     # num_workers = 12
#     num_workers = 8
#     # batch_size = 5000
#     batch_size = 64
#     epoch = 20
#     lr = 0.0001
#     stepLR = [5, 10]
#     weight_decay = 0.0005
#     display = 100
#     saveModel_epoch = 1
#     loss = 'SmoothL1'   # ['L1', 'L2', 'SmoothL1']

# config = Config()


import torch
import torchvision.transforms as T


class Config:
    # dataset
    # img_list = './generate_pseudo_labels/labels/pseudo_quality_score.txt'
    img_list = './generate_pseudo_labels/labels/quality_pseudo_labels.txt'
    data_root = 'D:\\IQA\\da_fen_zhangjm\\newPdata4_jm'  # 新增 data_root 属性
    # finetuning_model = './generate_pseudo_labels/extract_embedding/model/MobileFaceNet_MS1M.pth'
    finetuning_model = './generate_pseudo_labels/extract_embedding/model/20500backbone.pth'
    # save settings
    checkpoints = "./checkpoints/MS1M_Quality_Regression/S1"
    checkpoints_name = "MFN"
    # data preprocess
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    # training settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 0
    # multi_GPUs = [0,1,2,3,4,5,6,7]
    multi_GPUs = [0]
    backbone = 'MFN'  # [MFN, R_50]
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
    loss = 'SmoothL1'   # ['L1', 'L2', 'SmoothL1']

config = Config()
