import torch
import torchvision.transforms as T
from pathlib import Path


class RecConfig:
    # dataset
    data_root = (
        "/root/workspace/SDD-PIQA/data/ROI_Data"  # 目录结构: data_root/<id>/<img>
    )
    # training
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_GPUs = [0]
    seed = 0
    batch_size = 256
    num_workers = 8
    epoch = 40
    lr = 1e-3
    weight_decay = 5e-4
    stepLR = [10, 15]

    # backbone
    backbone = "R_50"  # [R_50]
    embedding_size = 512

    # transforms
    train_transform = T.Compose(
        [
            T.Resize((112, 112)),
            # 掌纹左右手不对称, 默认不做水平翻转; 如需可手动开启
            # T.RandomHorizontalFlip(p=0.0),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # checkpoint
    out_dir = Path(
        "/root/workspace/SDD-PIQA/generate_pseudo_labels/extract_embedding/model"
    )
    out_backbone = out_dir / "palmprint_R50_backbone.pth"
    out_classifier = out_dir / "palmprint_R50_classifier.pth"


rec_conf = RecConfig()
