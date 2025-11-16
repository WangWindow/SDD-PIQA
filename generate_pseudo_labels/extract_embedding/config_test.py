import torch
import torchvision.transforms as T


class Config:
    # dataset
    data_root = "/root/workplace/SDD-FIQA/data/ROI_Data"
    img_list = "/root/workplace/SDD-FIQA/generate_pseudo_labels/DATA.labelpath"
    # img_list = "feature_list"
    eval_model = "/root/workplace/SDD-FIQA/generate_pseudo_labels/extract_embedding/model/MobileFaceNet_MS1M.pth"
    outfile = "/root/workplace/SDD-FIQA/generate_pseudo_labels/npy/npy.npy"
    # data preprocess
    transform = T.Compose(
        [
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # network settings
    backbone = "MFN"  # [MFN, R_50]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_GPUs = [0]
    embedding_size = 512
    batch_size = 2000
    pin_memory = True
    num_workers = 4


config = Config()
