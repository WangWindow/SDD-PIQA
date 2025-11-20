import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm

from utils import model
from utils.dataset_txt import load_data as load_data_txt


class Config:
    # dataset
    img_list = "/root/workspace/SDD-PIQA/generate_pseudo_labels/annotations/quality_pseudo_labels.txt"
    data_root = "/root/workspace/SDD-PIQA/data/ROI_Data"  # 新增 data_root 属性
    # 使用掌纹识别阶段训练得到的 R50 backbone 作为初始化(可设为 None 从头训练)
    finetuning_model = "/root/workspace/SDD-PIQA/checkpoints/recognition_model/palmprint_R50_backbone.pth"
    # save settings
    checkpoints = "/root/workspace/SDD-PIQA/checkpoints/quality_model"
    checkpoints_name = "SDD-PIQA_quality_model"
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
    backbone = "IR_50"  # 使用与识别一致的 IR50 结构
    pin_memory = True
    # num_workers = 12
    num_workers = 8
    # 依据显存调整 batch_size，1 过低影响 BN；64 是较常见起点
    batch_size = 64
    epoch = 30
    lr = 1e-4
    stepLR = [5, 10]
    weight_decay = 0.0005
    display = 100
    saveModel_epoch = 1
    loss = "SmoothL1"  # ['L1', 'L2', 'SmoothL1']


conf = Config()


class TrainQualityTask:
    """TrainTask of quality model"""

    def __init__(self, config):
        super(TrainQualityTask, self).__init__()
        self.config = config

    def dataSet(self):
        # Data Setup
        trainloader, class_num = load_data_txt(self.config, label=True, train=True)
        return trainloader

    def backboneSet(self):
        # Network Setup
        device = self.config.device
        multi_GPUs = self.config.multi_GPUs
        net = model.IR50([112, 112], use_type="Qua").to(device)
        # Transfer learning from recognition model
        if self.config.finetuning_model is not None:
            print("=" * 20 + "FINE-TUNING" + "=" * 20)
            net_dict = net.state_dict()
            print("=" * 20 + "LOADING NETWROK PARAMETERS" + "=" * 20)
            pretrained_dict = torch.load(
                conf.finetuning_model, map_location=device, weights_only=True
            )
            pretrained_dict = {
                k.replace("module.", ""): v for k, v in pretrained_dict.items()
            }
            same_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            diff_dict = {k: v for k, v in net_dict.items() if k not in pretrained_dict}
            net_dict.update(same_dict)
            net.load_state_dict(net_dict)
            print(
                "=" * 20
                + f"LOADING DONE {len(same_dict)}/{len(pretrained_dict)} LAYERS"
                + "=" * 20
            )
            ignore_dictName = list(diff_dict.keys())
            print("=" * 20 + "INGNORING LAYERS:" + "=" * 20)
            print(ignore_dictName)
        if device != "cpu" and len(multi_GPUs) > 1:
            net = nn.DataParallel(net, device_ids=multi_GPUs)
        return net

    def trainSet(self, net):
        # Different regression loss including L1, L2, and Smooth L1
        if self.config.loss == "L1":
            print("LOSS TYPE = L1")
            criterion = nn.L1Loss()
        elif self.config.loss == "SmoothL1":
            print("LOSS TYPE = Smooth L1")
            criterion = nn.SmoothL1Loss()
        else:
            print("LOSS TYPE = L2")
            criterion = nn.MSELoss(reduction="mean")
        # Optimizer
        optimizer = optim.Adam(  # pyright: ignore[reportPrivateImportUsage]
            net.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.99),
            eps=1e-06,
            weight_decay=self.config.weight_decay,
        )
        # Scheduler
        scheduler_gamma = 0.1
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.config.stepLR, gamma=scheduler_gamma
        )
        return criterion, optimizer, scheduler

    def train(self, trainloader, net, epoch, criterion, optimizer, scheduler):
        # Train quality regression model
        net.train()
        itersNum = 1
        os.makedirs(self.config.checkpoints, exist_ok=True)
        logfile = open(os.path.join(self.config.checkpoints, "log"), "w")
        for e in range(epoch):
            loss_sum = 0
            for _, data, labels in tqdm(
                trainloader, desc=f"Epoch {e + 1}/{epoch}", total=len(trainloader)
            ):
                data = data.to(self.config.device)
                labels = labels.to(self.config.device).float()
                preds = net(data).squeeze()
                loss = criterion(preds, labels)
                loss_sum += np.mean(loss.cpu().detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itersNum % self.config.display == 0:
                    logfile = open(os.path.join(self.config.checkpoints, "log"), "a")
                    logfile.write(
                        f"Epoch {e + 1} / {self.config.epoch} | {itersNum} Loss="
                        + "\t"
                        + f"{loss}"
                        + "\n"
                    )
                itersNum += 1
            mean_loss = loss_sum / len(trainloader)
            print(f"LR = {optimizer.param_groups[0]['lr']} | Mean_Loss = {mean_loss}")
            logfile.write(
                f"LR = {optimizer.param_groups[0]['lr']} | Mean_Loss = {mean_loss}"
                + "\n"
            )

            # Save only the latest and the best model
            if e == 0:
                best_loss = float("inf")
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_model_path = os.path.join(
                    self.config.checkpoints,
                    f"{self.config.checkpoints_name}_best.pth",
                )
                torch.save(net.state_dict(), best_model_path)
                print(f"SAVE BEST MODEL: {best_model_path}")
            latest_model_path = os.path.join(
                self.config.checkpoints,
                f"{self.config.checkpoints_name}_latest.pth",
            )
            torch.save(net.state_dict(), latest_model_path)
            print(f"SAVE LATEST MODEL: {latest_model_path}")
            scheduler.step()
        return net


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(conf.seed)
    train_task = TrainQualityTask(conf)
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)
    net = train_task.backboneSet()
    trainloader = train_task.dataSet()
    criterion, optimizer, scheduler = train_task.trainSet(net)
    net = train_task.train(
        trainloader,
        net,
        epoch=conf.epoch,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
