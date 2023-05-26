from ctypes import addressof
from re import A
from matplotlib import transforms
from torchvision import transforms as T
from PIL import Image
import numpy as np
from time import time
import torch
from torch.utils import data
import torch.nn.functional as F
import h5py
from torch.utils.data import TensorDataset
import pickle
import matplotlib.pyplot as plt
import random
import torch.nn as nn
from tqdm import tqdm
from sudoku_data import process_inputs
import sys 

data_path = 'data/'
sys.path.append('../data/')


class SudokuDataset_Perception(data.Dataset):
    def __init__(self, filename, data_type, transform=None, t_param=None):
        assert filename in ['big_kaggle', 'minimal_17', 'multiple_sol', 'satnet'], 'error in the dataset name. Available datasets are big_kaggle, minimal_17, multiple_sol, and satnet'
        assert data_type in ['-test','-train','-valid'], 'error, types allowed are -test, -train, -valid.'
        self.images = h5py.File(data_path+filename+'/'+filename+'_imgs'+ data_type +'.hdf5','r')
        self.labels = np.load(data_path+filename+'/'+filename+ data_type+'.npy',allow_pickle=True).item()
        self.is_satnet_data = False
        if filename == "satnet":
            self.is_satnet_data = True
        if transform:
            if transform == 'rotation':
                angle = t_param
                self.transform = T.RandomRotation((-angle,angle),fill=255)
                print(f"Using noise RandomRotation, with angle {angle}")
            else:
                assert transform == "blur", f'Invalid noise parameter: {transform}'
                blur = t_param
                self.transform = T.GaussianBlur(kernel_size=(5,9),sigma=(blur,blur+0.1))
                print(f"Using noise GaussianBlur, in [{blur},{blur+0.1}]")
        else:
            self.transform = None

    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx): # 81 28 28
        if self.is_satnet_data:
            x = torch.from_numpy(np.array(self.images[str(idx)]))
        else:
            x = np.array(self.images[str(idx)])
            blocks = []
            def sliding(x):
                for i in range (0,252,28):
                    for j in range (0,252,28):
                        yield(i,j,x[i:i+28,j:j+28])
            
            bb = sliding(x)
            if self.transform is None:
                for b in bb:
                    blocks.append(b[2])
            else:
                transf1 = T.ToPILImage(mode = 'L')
                transf2 = T.PILToTensor()
                for b in bb:
                    if self.data_type == '-test':
                        image = torch.from_numpy(b[2])
                        rot_image = self.transform(transf1(image.to(torch.uint8)))                
                        rot_image = transf2(rot_image)
                        rot_image = rot_image.reshape(28,28)
                        blocks.append(rot_image)
                    else:
                        blocks.append(b[2])
            blocks = np.stack(blocks)    
            x = torch.from_numpy(blocks)
            x = x.unsqueeze(1).float()

        yt = torch.from_numpy(self.labels[idx].reshape(81))        
        yt = F.one_hot(yt,num_classes=10)
        return x.to(torch.float32), yt.to(torch.float32)


class SudokuDataset_RL(data.Dataset):
    def __init__(self, filename, data_type, transform=None, t_param=None):
        assert filename in ['big_kaggle', 'minimal_17', 'multiple_sol', 'satnet'], 'error in the dataset name. Available datasets are big_kaggle, minimal_17, multiple_sol, and satnet'
        assert data_type in ['-test','-train','-valid'], 'error, types allowed are -test, -train, -valid.'
        if filename=='satnet' and transform: 
            print('------>> Data tranform not implemented for satnet dataset') 
            quit()
        self.images = h5py.File(data_path+filename+'/'+filename+'_imgs'+ data_type +'.hdf5','r')
        self.labels = np.load(data_path+filename+'/'+filename+'_sol'+ data_type+'.npy',allow_pickle=True).item()
        self.data_type = data_type
        self.is_satnet_data = False
        if filename == "satnet":
            self.is_satnet_data = True
        if transform:
            if transform == 'rotation':
                angle = t_param
                self.transform = T.RandomRotation((-angle,angle),fill=255)
                print(f"Using noise RandomRotation, with angle {angle}")
            else:
                assert transform == "blur", f'Invalid noise parameter: {transform}'
                blur = t_param
                self.transform = T.GaussianBlur(kernel_size=(5,9),sigma=(blur,blur+0.1))
                print(f"Using noise GaussianBlur, in [{blur},{blur+0.1}]")
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        if self.is_satnet_data:
            x = torch.from_numpy(np.array(self.images[str(idx)]))
        else:
            x = np.array(self.images[str(idx)])
            blocks = []
            def sliding(x):
                for i in range (0,252,28):
                    for j in range (0,252,28):
                        yield(i,j,x[i:i+28,j:j+28])
            bb = sliding(x)
            if self.transform is None:
                for b in bb:
                    blocks.append(b[2])
            else:
                transf1 = T.ToPILImage(mode = 'L')
                transf2 = T.PILToTensor()
                for b in bb:
                    if self.data_type == '-test':
                        image = torch.from_numpy(b[2])
                        rot_image = self.transform(transf1(image.to(torch.uint8)))                
                        rot_image = transf2(rot_image)
                        rot_image = rot_image.reshape(28,28)
                        blocks.append(rot_image)
                    else:
                        blocks.append(b[2])
            blocks = np.stack(blocks)    
            
            x = torch.from_numpy(blocks)
            x = x.unsqueeze(1).float()
        yt = torch.from_numpy(self.labels[idx].reshape(81))        
        yt = F.one_hot(yt,num_classes=10)
        return x.to(torch.float32), yt.to(torch.float32)



class SudokuDataset_Solver(data.Dataset):
    # x 9x9x10
    # y 9x9x9
    def __init__(self, filename, data_type):
        assert filename in ['big_kaggle', 'minimal_17', 'multiple_sol', 'satnet'], 'error in the dataset name. Available datasets are big_kaggle, minimal_17, multiple_sol, and satnet'
        assert data_type in ['-test','-train','-valid'], 'error, types allowed are -test, -train, -valid.'
        self.data = np.load(data_path+filename+'/'+filename + data_type +'.npy',allow_pickle=True).item()
        self.labels = np.load(data_path+filename+'/'+filename+'_sol'+ data_type +'.npy',allow_pickle=True).item()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xt = torch.from_numpy(self.data[idx].reshape(81))
        yt = torch.from_numpy(self.labels[idx].reshape(81))
        xt = F.one_hot(xt,num_classes=10)
        yt = F.one_hot(yt,num_classes=10)
        yt = yt[:,1:]
        
        return xt.to(torch.float32), yt.to(torch.float32)


class SudokuDataset_Mask(data.Dataset):
    # x 9x9x9
    # y 9x9x1
    def __init__(self, filename, data_type):
        assert filename in ['big_kaggle', 'minimal_17', 'multiple_sol', 'satnet'], 'error in the dataset name. Available datasets are big_kaggle, minimal_17, multiple_sol, and satnet'
        assert data_type in ['-test','-train','-valid'], 'error, types allowed are -test, -train, -valid.'
        self.data = np.load(data_path+filename+'/'+filename+'_noise'+ data_type +'.npy',allow_pickle=True).item()
        self.labels = np.load(data_path+filename+'/'+filename+'_mask'+ data_type +'.npy',allow_pickle=True).item()
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xt = torch.from_numpy(self.data[idx].reshape(81))
        xt = F.one_hot(xt,num_classes = 10)
        xt = xt[:,1:]
        
        y = self.labels[idx].reshape(81)
        y = torch.tensor(y)
        
        return xt.to(torch.float32), y.to(torch.float32)



