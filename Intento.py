#Intento de Dataloader
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataproc import TrainDataset

valPath='val_pair.lst'
trainPath='train_pair.lst'
valDataset = TrainDataset(valPath)
trainDataset = TrainDataset(trainPath)

valDataloader = DataLoader(valDataset, shuffle=False)
trainDataloader = DataLoader(trainDataset, shuffle=True)
import pdb; pdb.set_trace()