from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils import model  # noqa: E402
from utils.dataset_txt import load_data as load_data_txt  # noqa: E402


class Config:
    # dataset
    data_root = "/root/workspace/SDD-PIQA/data/ROI_Data"
    img_list = "/root/workspace/SDD-PIQA/generate_pseudo_labels/features/DATA.labelpath"
    # 使用我们训练得到的掌纹识别 backbone 权重
    eval_model = "/root/workspace/SDD-PIQA/checkpoints/recognition_model/palmprint_R50_backbone.pth"
    outfile = "/root/workspace/SDD-PIQA/generate_pseudo_labels/features/features.npy"
    # data preprocess
    transform = T.Compose(
        [
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # network settings
    backbone = "R_50"  # [R_50]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_GPUs = [0]
    embedding_size = 512
    batch_size = 512
    pin_memory = True
    num_workers = 2


conf = Config()


def dataSet():  # Dataset setup
    """
    Dataset setup
    Bulid a dataloader for training
    """
    dataloader, class_num = load_data_txt(conf, label=False, train=False)
    return dataloader, class_num


def backboneSet():  # Network setup
    """
    Backbone setup
    Load a Backbone for training, support MobileFaceNet(MFN) and ResNet50(R50)
    """
    net = model.IR50([112, 112], use_type="Rec").to(device)
    # load trained model weights
    if conf.eval_model is not None:
        net_dict = net.state_dict()
        eval_dict = torch.load(conf.eval_model, map_location=device)
        eval_dict = {k.replace("module.", ""): v for k, v in eval_dict.items()}
        same_dict = {k: v for k, v in eval_dict.items() if k in net_dict}
        net_dict.update(same_dict)
        net.load_state_dict(net_dict)
    # if use multi-GPUs
    if device != "cpu" and len(multi_GPUs) > 1:
        net = nn.DataParallel(net, device_ids=multi_GPUs)
    return net


def compcos(feats1, feats2):  # Computing cosine distance
    """
    Computing cosine distance
    For similarity
    """
    cos = np.dot(feats1, feats2) / (np.linalg.norm(feats1) * np.linalg.norm(feats2))
    return cos


def npy2txt(img_list, feats_nplist, outfile):  # npy to txt for embedding save
    """
    For save embeddings to txt file
    """
    allFeats = np.load(feats_nplist)
    print(np.shape(allFeats))
    with open(img_list, "r") as f:
        for index, value in tqdm(enumerate(f)):
            imgPath = value.split()[0]
            feats = allFeats[index]
            feats = " ".join(map(str, feats))
            # ouput to the txt
            outfile.write(imgPath + " " + feats + "\n")


if __name__ == "__main__":
    """
    This method is to extract features from face dataset
    and save to numpy file
    """
    device = conf.device
    multi_GPUs = conf.multi_GPUs
    net = backboneSet()
    outfile = conf.outfile
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    dataloader, class_num = dataSet()
    count = 0
    net.eval()
    with open(conf.img_list, "r") as f:
        txtContent = f.readlines()
    # computer the number of sampes
    sample_num = len(txtContent)
    print(f"Sample_num = {sample_num}")
    feats = np.zeros(
        [sample_num, conf.embedding_size]
    )  # initnte features of all samples
    with torch.no_grad():
        for datapath, data in tqdm(dataloader, total=len(dataloader)):
            data = data.to(device)
            embeddings = F.normalize(net(data), p=2, dim=1).cpu().numpy().squeeze()
            start_idx = count * conf.batch_size
            end_idx = (count + 1) * conf.batch_size
            # save embeddings of one iteration
            try:
                feats[start_idx:end_idx, :] = embeddings
            # save embeddings of the final iteration
            except Exception:
                feats[start_idx:, :] = embeddings
            count += 1
        np.save(outfile, feats)
        checkfeats = np.load(outfile)
        print(np.shape(checkfeats))
        print(outfile)
