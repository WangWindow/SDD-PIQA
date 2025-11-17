import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from train_recognition_config import rec_conf as conf
from generate_pseudo_labels.extract_embedding.model import model as ir_model
from generate_pseudo_labels.extract_embedding.model import (
    model_mobilefaceNet as mfn_model,
)


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
    if conf.backbone == "MFN":
        backbone = mfn_model.MobileFaceNet(
            [112, 112], conf.embedding_size, output_name="GDC", use_type="Rec"
        ).to(device)
    else:
        backbone = ir_model.R50([112, 112], use_type="Rec").to(device)
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
                {"loss": f"{running_loss / total:.4f}", "acc": f"{correct / total:.4f}"}
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

        print(
            f"Epoch {epoch + 1}: loss={running_loss / total:.4f}, acc={correct / total:.4f}"
        )
        print(f"Saved backbone -> {conf.out_backbone}")


if __name__ == "__main__":
    main()
