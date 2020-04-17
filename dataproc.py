# dataproc.py: Dataset loader classes for BSDS
# Author: Nishanth Koganti
# Date: 2017/10/11

# Source: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# Issues:
# Merge TrainDataset and TestDataset classes

# import libraries
import os
import numpy as np
import pandas as pd
import nibabel as nb

# import torch modules
from torch.utils.data import Dataset


# BSDS dataset class for training data
class TrainDataset(Dataset):
    def __init__(self, fileNames,
                 transform=None, target_transform=None):
        self.transform = transform
        self.targetTransform = target_transform
        self.frame = pd.read_csv(
            fileNames, dtype=str, delimiter=' ', header=None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        inputName = os.path.join( self.frame.iloc[idx, 0])
        targetName = os.path.join( self.frame.iloc[idx, 1])
        Name= os.path.basename(self.frame.iloc[idx,0])
        # process the images
        inputImage = nb.load(inputName).get_fdata()
        if self.transform is not None:
            inputImage = self.transform(inputImage)

        targetImage = nb.load(targetName).get_fdata()
        if self.targetTransform is not None:
            targetImage = self.targetTransform(targetImage)

        return Name, inputImage, targetImage


# dataset class for test dataset
class TestDataset(Dataset):
    def __init__(self, fileNames, transform=None):
        self.transform = transform
        self.frame = pd.read_csv(
            fileNames, dtype=str, delimiter=' ', header=None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        fname = self.frame.iloc[idx, 0]
        inputName = os.path.join( fname)
        Namet = os.path.basename(fname)
        # process the images
        inputImage = nb.load(inputName).get_fdata()
        if self.transform is not None:
            inputImage = self.transform(inputImage)

        return inputImage, Namet
