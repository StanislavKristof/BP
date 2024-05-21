import torch
import torchvision.transforms as transforms
import pandas as pd
from omegaconf.errors import ConfigAttributeError
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations
from pathlib import Path
import numpy as np

from hydra.utils import instantiate

"""
def index_to_one_hot(y):
    y = torch.LongTensor([y])[0]
    y = TNF.one_hot(y, num_classes=1000) # 120
    y = y.float()
    return y
"""

def y_transform(y):
    y = np.array(y)
    y = torch.FloatTensor(y)
    y = y.float()
    return y


class FOOTBALL(Dataset):
    def __init__(self, dataset_path, split, download, transform=None, target_transform=None):
        if split == None: # if TRAVERSE
            labels_path = Path(dataset_path) / f"labels.csv"
            self.isFinal = True
            self.idxs = []
        else: # if TRAIN or VAL
            labels_path = Path(dataset_path) / f"labels_{split}.csv"
            self.isFinal = False
        self.img_labels = pd.read_csv(labels_path.as_posix())
        self.keypoints = np.load(Path(dataset_path) / f"keypoints.npy")

        self.root_dir = Path(dataset_path)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.root_dir / self.img_labels.iloc[idx]["img"]
        
        # tensor of keypoints
        label = self.keypoints[self.img_labels.iloc[idx]["Unnamed: 0"]]
        
        image = cv2.imread(img_name.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.isFinal: # TRAVERSE does not have augmentations
            transform = albumentations.HorizontalFlip(p=0.5)
            augImg = transform(image=image)['image']
        
            areImagesEqual = np.all(image == augImg)
            # if the image was flipped
            if not areImagesEqual:

                # Change of x coordinate
                label[:, 0] = 1 - label[:, 0]
                
                # Swap of keypoints
                npOrder = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
                label = label[npOrder]
                image = augImg

            transform = albumentations.Compose(
                [
                albumentations.ImageCompression(p=0.3, quality_lower=30, quality_upper=100),
                albumentations.ColorJitter(p=0.4, hue=0),
                albumentations.FancyPCA(p=0.2),
                albumentations.GaussNoise(p=0.1),
                albumentations.ISONoise(p=0.2),
                albumentations.MultiplicativeNoise(p=0.3, multiplier=(0.7, 1.3)),
                albumentations.RandomBrightnessContrast(p=0.2),
                albumentations.RingingOvershoot(p=0.1),
                albumentations.Sharpen(p=0.1),
                albumentations.ToSepia(p=0.05),
                albumentations.UnsharpMask(p=0.2)
                ]
            )
            augImg = transform(image=image)['image']

            image = augImg

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class FOOTBALL_HEATMAP(Dataset):
    def __init__(self, dataset_path, split, download, transform=None, target_transform=None):
        if split == None: # if TRAVERSE
            labels_path = Path(dataset_path) / f"labels.csv"
            self.isFinal = True
        else: # if TRAIN or VAL
            labels_path = Path(dataset_path) / f"labels_{split}.csv"
            self.isFinal = False
        self.img_labels = pd.read_csv(labels_path.as_posix())
        self.keypoints = np.load(Path(dataset_path) / f"keypoints.npy")

        self.root_dir = Path(dataset_path)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.root_dir / self.img_labels.iloc[idx]["img"]
        
        # tensor of keypoints
        label = self.keypoints[self.img_labels.iloc[idx]["Unnamed: 0"]]

        kernelSize = 10
        keypoint_heatmaps = [] # collects heatmaps of a pose
        for keypoint in range(14):
            # Convert into heatmap
            # source for constructing Gaussian Kernel: https://stackoverflow.com/a/57459583
            xdir_gauss = cv2.getGaussianKernel(kernelSize, 1)
            kernel = np.multiply(xdir_gauss.T, xdir_gauss)
            middle = kernel[kernelSize//2, kernelSize//2] # middle of the kernel has the highest probability
            highest_num = 1 # keypoint has the probability of 1
            multiplier = highest_num / middle
            kernel = kernel * multiplier

            # if visible then this
            if label[keypoint][2] == 1:
                lower_x = int(128*label[keypoint][0]//1)
                lower_y = int(128*label[keypoint][1]//1)
                keypoint_x = round(128*label[keypoint][0])
                keypoint_y = round(128*label[keypoint][1])


                begin_y_offset = 0 # how much to cut from the top
                if lower_y < 5: # kernel will have to be cut
                    begin_y_offset = 4-lower_y
                    if lower_y == 0: # minor adjustment
                        begin_y_offset = 5
                    kernel = kernel[begin_y_offset:, :] # cutting of kernel

                begin_x_offset = 0 # how much to cut from the left
                if lower_x < 5: # kernel will have to be cut
                    begin_x_offset = 4-lower_x
                    if lower_x == 0: # minor adjustment
                        begin_x_offset = 5
                    kernel = kernel[:, begin_x_offset:] # cutting of kernel
        
            else:  # if not visible
                kernel = np.zeros(128*128)
                kernel = kernel.reshape(128, 128) # empty kernel
                keypoint_heatmaps.append(kernel)
                continue

            # number of rows before the kernel
            n_upper_y = keypoint_y-(kernelSize//2)+1-(keypoint_y-lower_y)
            # if Y somewhere between lower_y and rounded keypoint_y, we subtract 1, kernelMiddle will be in both lower_y and keypoint_y

            try:
                row = np.zeros((n_upper_y, kernel.shape[1])) # pad kernel from the top
                kernel = np.append(row, kernel, axis=0)
            except ValueError: # n_upper_y < 0
                pass # we dont need to add any rows before the kernel
            try:
                row = np.zeros((128-kernel.shape[0], kernel.shape[1])) # pad kernel from the bottom
                kernel = np.append(kernel, row, axis=0)
            except ValueError: # no need to add new rows, the kernel is too big
                kernel = kernel[:128,:] # cutting the kernel (in case it is too big)
            
            # equivalent to the code above, but for x
            # number of columns before the kernel
            n_left_x = keypoint_x - (kernelSize//2) +1 - (keypoint_x-lower_x)
            try:
                col = np.zeros((kernel.shape[0], n_left_x)) # pad kernel from the left
                kernel = np.append(col, kernel, axis=1)
            except ValueError:
                pass

            try:
                col = np.zeros((kernel.shape[0], 128 - kernel.shape[1])) # pad kernel from the right
                kernel = np.append(kernel, col,axis=1)
            except ValueError:
                kernel = kernel[:, :128] # cutting the kernel
            
            keypoint_heatmaps.append(kernel)

        # Open image
        image = cv2.imread(img_name.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.isFinal: # TRAVERSE does not have augmentations
            transform = albumentations.HorizontalFlip(p=0.5)
            augImg = transform(image=image)['image']
            
            areImagesEqual = np.all(image == augImg)
            # if the image was flipped
            if not areImagesEqual:
                for i, kernel in enumerate(keypoint_heatmaps):
                    
                    # Flip the kernel
                    kernel = kernel[:, ::-1]
                    keypoint_heatmaps[i] = kernel

            keypoint_heatmaps = np.array(keypoint_heatmaps)
        
            # if the image was flipped
            if not areImagesEqual:

                # Swap of keypoints
                npOrder = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13] # vymena keypointov
                keypoint_heatmaps = keypoint_heatmaps[npOrder]
                image = augImg
        
            transform = albumentations.Compose(
                [
                albumentations.ImageCompression(p=0.3, quality_lower=30, quality_upper=100),
                albumentations.ColorJitter(p=0.4, hue=0),
                albumentations.FancyPCA(p=0.2),
                albumentations.GaussNoise(p=0.1),
                albumentations.ISONoise(p=0.2),
                albumentations.MultiplicativeNoise(p=0.3, multiplier=(0.7, 1.3)),
                albumentations.RandomBrightnessContrast(p=0.2),
                albumentations.RingingOvershoot(p=0.1),
                albumentations.Sharpen(p=0.1),
                albumentations.ToSepia(p=0.05),
                albumentations.UnsharpMask(p=0.2)
                ]
            )
            augImg = transform(image=image)['image']

            image = augImg
        else:
            keypoint_heatmaps = np.array(keypoint_heatmaps)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, keypoint_heatmaps

class DataModule:
    def __init__(self, cfg):
        self.hparams = cfg
        self.dataset_path = "/mnt/nfs-data/public/xkristof"#".scratch/datasets" # iny adresar

    def setup(self):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        try:
            self.ds_train = instantiate(
                self.hparams.dataset.train, 
                transform=transform,
                target_transform=y_transform
            )

            self.ds_val = instantiate(
                self.hparams.dataset.val,
                transform=transform,
                target_transform=y_transform
            )
        except ConfigAttributeError: # FINAL
            pass

        # ADDED WHILE CREATING TRAVERSER
        try:
            self.ds_final = instantiate(
                self.hparams.dataset.final,
                transform=transform,
                target_transform=y_transform
            )
        except ConfigAttributeError:
            pass

        # Setup loaders
        try:
            self.loader_train = DataLoader(
                self.ds_train,
                batch_size = self.hparams.train.batch_size,
                shuffle=True,
                num_workers=self.hparams.train.num_workers
            )
        except AttributeError:
            pass
        try:
            self.loader_val = DataLoader(
                self.ds_val,
                batch_size=self.hparams.train.batch_size,
                shuffle=False,
                num_workers=self.hparams.train.num_workers
            )
        except AttributeError:
            pass

        # ADDED WHILE CREATING TRAVERSER
        try:
            self.loader_final = DataLoader(
                self.ds_final,
                batch_size=self.hparams.train.batch_size,
                shuffle = False,
                num_workers=self.hparams.train.num_workers
            )
        except AttributeError:
            pass
