import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from tqdm import tqdm

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from utils import model  # noqa: E402


class Config:
    # dataset
    data_root = (
        "/root/workspace/SDD-PIQA/data/ROI_Data"  # 目录结构: data_root/<id>/<img>
    )
    # training
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_GPUs = [0]
    seed = 0
    batch_size = 64
    num_workers = 8
    epoch = 30
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
    out_dir = Path("/root/workspace/SDD-PIQA/checkpoints/recognition_model")
    out_backbone = out_dir / "palmprint_R50_backbone.pth"
    out_classifier = out_dir / "palmprint_R50_classifier.pth"
    log_file = out_dir / "log"


conf = Config()


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_size: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits, feats


def build_backbone(device: str):
    backbone = model.IR50([112, 112], use_type="Rec").to(device)
    return backbone


def main():
    os.makedirs(conf.out_dir, exist_ok=True)
    set_seed(conf.seed)
    device = conf.device

    # dataset
    dataset = ImageFolder(conf.data_root, transform=conf.train_transform)
    num_classes = len(dataset.classes)
    loader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=True,
    )

    # model
    backbone = build_backbone(device)
    model = Classifier(backbone, conf.embedding_size, num_classes).to(device)

    if device != "cpu" and len(conf.multi_GPUs) > 1:
        model = nn.DataParallel(model, device_ids=conf.multi_GPUs)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(  # pyright: ignore[reportPrivateImportUsage]
        model.parameters(),
        lr=conf.lr,
        weight_decay=conf.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=conf.stepLR, gamma=0.1
    )

    model.train()
    # Open log file
    with open(conf.log_file, "w") as log_f:
        for epoch in range(conf.epoch):
            running_loss = 0.0
            correct = 0
            total = 0
            pbar = tqdm(loader, desc=f"Rec-Train {epoch + 1}/{conf.epoch}")
            for imgs, targets in pbar:
                imgs = imgs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                logits, _ = model(imgs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(logits, 1)
                correct += (preds == targets).sum().item()
                total += imgs.size(0)
                pbar.set_postfix(
                    {
                        "loss": f"{running_loss / total:.4f}",
                        "acc": f"{correct / total:.4f}",
                    }
                )
            scheduler.step()

            # save checkpoint each epoch
            # 仅保存 backbone 的权重，方便后续 extract_feats 直接加载
            if isinstance(model, nn.DataParallel):
                backbone_state = model.module.backbone.state_dict()
                full_state = model.module.state_dict()
            else:
                backbone_state = model.backbone.state_dict()
                full_state = model.state_dict()
            torch.save(backbone_state, conf.out_backbone)
            torch.save(full_state, conf.out_classifier)

            log_msg = f"Epoch {epoch + 1}: loss={running_loss / total:.4f}, acc={correct / total:.4f}"
            print(log_msg)
            log_f.write(log_msg + "\n")
            log_f.flush()  # Ensure writing to disk
            print(f"Saved backbone -> {conf.out_backbone}")


if __name__ == "__main__":
    main()
