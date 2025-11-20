import torch
import torchvision.transforms as T


class Config:
    # dataset
    data_root = "/root/workspace/SDD-PIQA/data/ROI_Data"
    img_list = "/root/workspace/SDD-PIQA/generate_pseudo_labels/DATA.labelpath"
    # 使用我们训练得到的掌纹识别 backbone 权重
    eval_model = "/root/workspace/SDD-PIQA/generate_pseudo_labels/extract_embedding/model/palmprint_R50_backbone.pth"
    outfile = "/root/workspace/SDD-PIQA/generate_pseudo_labels/features.npy"
    # data preprocess
    transform = T.Compose(
        [
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # network settings
    # backbone = "R_50"  # [MFN, R_50]
    backbone = "R_50"  # [R_50]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_GPUs = [0]
    embedding_size = 512
    batch_size = 512
    pin_memory = True
    num_workers = 2


config = Config()
