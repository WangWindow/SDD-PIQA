# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from torch.utils import data
# from PIL import Image
# import numpy as np
# import os
# import pdb

# class Dataset(data.Dataset):
#     '''
#     Build dataset via data list file
#     '''
#     def __init__(self, conf, label=True):
#         super().__init__()
#         # self.data_root = conf.data_root
#         self.img_list = conf.img_list
#         self.transform = conf.transform
#         self.batch_size = conf.batch_size
#         self.label = label
#         with open(self.img_list, 'r') as f:
#             self.imgPath = []
#             self.target = []
#             self.classes = set()
#             for index, value in enumerate(f):
#                 value = value.split()
#                 if self.label:
#                     if value and len(value) < 2:                     # check data file
#                         print(f"ERROR, {value}({index}-th) is missing, please check it")
#                     else:
#                         self.imgPath.append(value[0])
#                         self.target.append(float(value[1]))
#                         self.classes.add(float(value[1]))
#                 else:
#                     self.imgPath.append(value[0])
#             self.target = np.asarray(self.target)
#         print(f"Number of samples: {len(self.imgPath)}")

#     def __getitem__(self, index):
#         '''
#         This method is used during the visiting of dataloader
#         Data processing and output
#         '''
#         imgPath = self.imgPath[index]
#         img = Image.open(imgPath).convert("RGB")
#         if self.transform is not None: img = self.transform(img)    # data processing
#         if self.label:
#             target = self.target[index]
#             return imgPath, img, target
#         else:
#             return imgPath, img

#     def __len__(self):
#         return(len(self.imgPath))

# def load_data(conf, label=True, train=False):                                     # build dataloder
#     '''
#     Build dataloader
#     Two parameters including "label" and "train" are used for the output of dataloader
#     '''
#     dataset = Dataset(conf, label)
#     if train:
#         loader = DataLoader(dataset,
#                         batch_size=conf.batch_size,
#                         shuffle=True,
#                         pin_memory=conf.pin_memory,
#                         num_workers=conf.num_workers)
#     else:
#         loader = DataLoader(dataset,
#                         batch_size=conf.batch_size,
#                         shuffle=False,
#                         pin_memory=conf.pin_memory,
#                         num_workers=conf.num_workers)
#     class_num = len(dataset.classes)
#     return loader, class_num


from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import numpy as np
import os


class Dataset(data.Dataset):
    """
    Build dataset via data list file
    """

    def __init__(self, conf, label=True):
        super().__init__()
        self.data_root = conf.data_root
        self.img_list = conf.img_list
        self.transform = conf.transform
        self.batch_size = conf.batch_size
        self.label = label
        with open(self.img_list, "r") as f:
            self.imgPath = []
            self.target = []
            self.classes = set()
            for index, value in enumerate(f):
                value = value.split()
                if self.label:
                    if value and len(value) < 2:  # check data file
                        print(f"ERROR, {value}({index}-th) is missing, please check it")
                    else:
                        # 去除可能包含的 data_root 部分
                        img_path = value[0].replace(self.data_root, "").lstrip("\\/")
                        self.imgPath.append(img_path)
                        self.target.append(float(value[1]))
                        self.classes.add(float(value[1]))
                else:
                    img_path = value[0].replace(self.data_root, "").lstrip("\\/")
                    self.imgPath.append(img_path)
            self.target = np.asarray(self.target)
        print(f"Number of samples: {len(self.imgPath)}")

    def __getitem__(self, index):
        """
        This method is used during the visiting of dataloader
        Data processing and output
        """
        imgPath = self.imgPath[index]
        full_img_path = os.path.join(self.data_root, imgPath)
        img = Image.open(full_img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)  # data processing
        if self.label:
            target = self.target[index]
            return imgPath, img, target
        else:
            return imgPath, img

    def __len__(self):
        return len(self.imgPath)


def load_data(conf, label=True, train=False):  # build dataloder
    """
    Build dataloader
    Two parameters including "label" and "train" are used for the output of dataloader
    """
    dataset = Dataset(conf, label)
    # 修改 num_workers 为 0
    if train:
        loader = DataLoader(
            dataset,
            batch_size=conf.batch_size,
            shuffle=True,
            pin_memory=conf.pin_memory,
            num_workers=0,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=conf.batch_size,
            shuffle=False,
            pin_memory=conf.pin_memory,
            num_workers=0,
        )
    class_num = len(dataset.classes)
    return loader, class_num
