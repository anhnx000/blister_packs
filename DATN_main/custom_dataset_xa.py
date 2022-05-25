import torch
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

class CustomDataset_xa(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_path_list = []
        self.label_list = []

        folder_list = os.listdir(root_dir)
        for folder_name in folder_list:
            rel_path = os.path.join(root_dir, folder_name)
            for file in os.listdir(rel_path):
                self.image_path_list.append(os.path.join(rel_path, file))
                self.label_list.append(folder_name)
    
    
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        img_path = self.image_path_list[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.label_list[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(label)
        
    


