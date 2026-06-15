import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')
        

class Dataset_AddPure_Extract4types(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
                 asi=0, aei=18, val_ratio=0.1, add_noise=False, noise_std=1e-4,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        if flag == 'pred':
            flag = 'test'
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.asi = asi
        self.aei = aei
        self.val_ratio = val_ratio
        self.add_noise = add_noise
        self.noise_std = noise_std
         
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        
        self.__read_data__()

    def __read_data__(self):
        stat_raw = np.load(os.path.join(self.root_path, self.stat_path))
        self.x_mean = stat_raw['x_mean']
        self.x_std = stat_raw['x_std']
        self.y_mean = stat_raw['y_mean']
        self.y_std = stat_raw['y_std']
        
        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        # x = (N, Y, 12, 136)
        # y = (age, N, Y+1, 12, 10)
        pft_raw = np.load(os.path.join(self.root_path, self.pft_path))
        # (N, 18, Y+1, 12)

        if self.set_type == 2: 
            data_x = data_raw[f'x_test_MIX'] # (2825, Y, 12, 136)
            data_y = data_raw[f'y_test_MIX'] # (2825, Y+1, 12, 10)
            data_x_BL = data_raw[f'x_test_BL'] # (254, Y, 12, 136)
            data_x_NL = data_raw[f'x_test_NL'] # (582, Y, 12, 136)
            data_x_GS = data_raw[f'x_test_GS'] # (615, Y, 12, 136)
            data_y_BL = data_raw[f'y_test_BL'] # (254, Y+1, 12, 10)
            data_y_NL = data_raw[f'y_test_NL'] # (582, Y+1, 12, 10)
            data_y_GS = data_raw[f'y_test_GS'] # (615, Y+1, 12, 10)

            # pft_BL = pft_raw['test_BL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] # (N, 18, Y+1, 12) to (N, age, Y+1, 12, 1) 
            # pft_NL = pft_raw['test_NL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            # pft_GS = pft_raw['test_GS'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            pft_BL_BL = pft_raw['test_pft_BL_BL'][:, 12:][..., np.newaxis] # (254, 492) to (N, 336, 1) 
            pft_BL_NL = pft_raw['test_pft_BL_NL'][:, 12:][..., np.newaxis]
            pft_BL_GS = pft_raw['test_pft_BL_GS'][:, 12:][..., np.newaxis]
            pft_NL_BL = pft_raw['test_pft_NL_BL'][:, 12:][..., np.newaxis]
            pft_NL_NL = pft_raw['test_pft_NL_NL'][:, 12:][..., np.newaxis]
            pft_NL_GS = pft_raw['test_pft_NL_GS'][:, 12:][..., np.newaxis]
            pft_GS_BL = pft_raw['test_pft_GS_BL'][:, 12:][..., np.newaxis]
            pft_GS_NL = pft_raw['test_pft_GS_NL'][:, 12:][..., np.newaxis]
            pft_GS_GS = pft_raw['test_pft_GS_GS'][:, 12:][..., np.newaxis]
            pft_MIX_BL = pft_raw['test_pft_MIX_BL'][:, 12:][..., np.newaxis]
            pft_MIX_NL = pft_raw['test_pft_MIX_NL'][:, 12:][..., np.newaxis]
            pft_MIX_GS = pft_raw['test_pft_MIX_GS'][:, 12:][..., np.newaxis]

        else: 
            data_x = data_raw[f'x_train_MIX'] # (11415, Y, 12, 136)
            data_y = data_raw[f'y_train_MIX'] # (11415, Y+1, 12, 10)
            data_x_BL = data_raw[f'x_train_BL'] # (958, Y, 12, 136)
            data_x_NL = data_raw[f'x_train_NL'] # (2180, Y, 12, 136)
            data_x_GS = data_raw[f'x_train_GS'] # (2556, Y, 12, 136)
            data_y_BL = data_raw[f'y_train_BL'] # (958, Y+1, 12, 10)
            data_y_NL = data_raw[f'y_train_NL'] # (2180, Y+1, 12, 10)
            data_y_GS = data_raw[f'y_train_GS'] # (2556, Y+1, 12, 10)
            # pft_BL = pft_raw['train_BL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] # (N, age, Y, 12, 1)
            # pft_NL = pft_raw['train_NL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            # pft_GS = pft_raw['train_GS'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            pft_BL_BL = pft_raw['train_pft_BL_BL'][:, 12:][..., np.newaxis] 
            pft_BL_NL = pft_raw['train_pft_BL_NL'][:, 12:][..., np.newaxis]
            pft_BL_GS = pft_raw['train_pft_BL_GS'][:, 12:][..., np.newaxis]
            pft_NL_BL = pft_raw['train_pft_NL_BL'][:, 12:][..., np.newaxis]
            pft_NL_NL = pft_raw['train_pft_NL_NL'][:, 12:][..., np.newaxis]
            pft_NL_GS = pft_raw['train_pft_NL_GS'][:, 12:][..., np.newaxis]
            pft_GS_BL = pft_raw['train_pft_GS_BL'][:, 12:][..., np.newaxis]
            pft_GS_NL = pft_raw['train_pft_GS_NL'][:, 12:][..., np.newaxis]
            pft_GS_GS = pft_raw['train_pft_GS_GS'][:, 12:][..., np.newaxis]
            pft_MIX_BL = pft_raw['train_pft_MIX_BL'][:, 12:][..., np.newaxis]
            pft_MIX_NL = pft_raw['train_pft_MIX_NL'][:, 12:][..., np.newaxis]
            pft_MIX_GS = pft_raw['train_pft_MIX_GS'][:, 12:][..., np.newaxis]


        mask = np.ones_like(data_y, dtype=bool) 
        mask[..., 7] = False                    
        data_y[mask & (data_y < 0)] = 0

        data_y_arr = data_y.copy() # (11415, Y+1, 12, 10)
        data_y = data_y.reshape(data_y_arr.shape[0], data_y_arr.shape[1]*data_y_arr.shape[2], data_y_arr.shape[3]) # (N, 492, 10)
        data_y = data_y[:, 11:, :] # (N, 337, 10)
        data_y = data_y[:, np.newaxis, :, :]         
        # data_y = np.transpose(data_y, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
        # add random-walk noise to target
        if self.add_noise:
            data_y =  self.__add_random_walk_noise_to_batch__(data_y, self.noise_std)


        # data_y_BL[data_y_BL < 0] = 0
        mask = np.ones_like(data_y_BL, dtype=bool) 
        mask[..., 7] = False                    
        data_y_BL[mask & (data_y_BL < 0)] = 0

        data_y_BL_arr = data_y_BL.copy() # (11415, Y+1, 12, 10)
        data_y_BL = data_y_BL.reshape(data_y_BL_arr.shape[0], data_y_BL_arr.shape[1]*data_y_BL_arr.shape[2], data_y_BL_arr.shape[3]) # (N, 492, 10)
        data_y_BL = data_y_BL[:, 11:, :] 
        data_y_BL = data_y_BL[:, np.newaxis, :, :]         
        # data_y_BL = np.transpose(data_y_BL, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
        # add random-walk noise to target
        if self.add_noise:
            data_y_BL =  self.__add_random_walk_noise_to_batch__(data_y_BL, self.noise_std)


        # data_y_NL[data_y_NL < 0] = 0
        mask = np.ones_like(data_y_NL, dtype=bool) 
        mask[..., 7] = False                    
        data_y_NL[mask & (data_y_NL < 0)] = 0

        data_y_NL_arr = data_y_NL.copy() # (11415, Y+1, 12, 10)
        data_y_NL = data_y_NL.reshape(data_y_NL_arr.shape[0], data_y_NL_arr.shape[1]*data_y_NL_arr.shape[2], data_y_NL_arr.shape[3]) # (N, 492, 10)
        data_y_NL = data_y_NL[:, 11:, :] 
        data_y_NL = data_y_NL[:, np.newaxis, :, :]         
        # data_y_NL = np.transpose(data_y_NL, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
        # add random-walk noise to target
        if self.add_noise:
            data_y_NL =  self.__add_random_walk_noise_to_batch__(data_y_NL, self.noise_std)


        # data_y_GS[data_y_GS < 0] = 0
        mask = np.ones_like(data_y_GS, dtype=bool) 
        mask[..., 7] = False                    
        data_y_GS[mask & (data_y_GS < 0)] = 0

        data_y_GS_arr = data_y_GS.copy() # (11415, Y+1, 12, 10)
        data_y_GS = data_y_GS.reshape(data_y_GS_arr.shape[0], data_y_GS_arr.shape[1]*data_y_GS_arr.shape[2], data_y_GS_arr.shape[3]) # (N, 492, 10)
        data_y_GS = data_y_GS[:, 11:, :] 
        data_y_GS = data_y_GS[:, np.newaxis, :, :]         
        # data_y_GS = np.transpose(data_y_GS, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
        # add random-walk noise to target
        if self.add_noise:
            data_y_GS =  self.__add_random_walk_noise_to_batch__(data_y_GS, self.noise_std)
                 

        # dup x for matching ages 
        # data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1)  # (2825, Y, 12, 136)
        data_x_arr = data_x.copy()
        data_x = data_x.reshape(data_x_arr.shape[0], data_x_arr.shape[1]*data_x_arr.shape[2], data_x_arr.shape[3]) # (N, Y *12, 136)
        data_x = data_x[:, np.newaxis, :, :] # (N, 1, Y *12, 136)
        # normalize data
        data_x = self.__norm_data__(data_x, self.x_mean, self.x_std)

        data_x_BL_arr = data_x_BL.copy()
        data_x_BL = data_x_BL.reshape(data_x_BL_arr.shape[0], data_x_BL_arr.shape[1]*data_x_BL_arr.shape[2], data_x_BL_arr.shape[3]) # (N, Y *12, 136)
        data_x_BL = data_x_BL[:, np.newaxis, :, :] # (N, 1, Y *12, 136)
        # normalize data
        data_x_BL = self.__norm_data__(data_x_BL, self.x_mean, self.x_std)

        data_x_NL_arr = data_x_NL.copy()
        data_x_NL = data_x_NL.reshape(data_x_NL_arr.shape[0], data_x_NL_arr.shape[1]*data_x_NL_arr.shape[2], data_x_NL_arr.shape[3]) # (N, Y *12, 136)
        data_x_NL = data_x_NL[:, np.newaxis, :, :] # (N, 1, Y *12, 136)
        # normalize data
        data_x_NL = self.__norm_data__(data_x_NL, self.x_mean, self.x_std)

        data_x_GS_arr = data_x_GS.copy()
        data_x_GS = data_x_GS.reshape(data_x_GS_arr.shape[0], data_x_GS_arr.shape[1]*data_x_GS_arr.shape[2], data_x_GS_arr.shape[3]) # (N, Y *12, 136)
        data_x_GS = data_x_GS[:, np.newaxis, :, :] # (N, 1, Y *12, 136)
        # normalize data
        data_x_GS = self.__norm_data__(data_x_GS, self.x_mean, self.x_std)


        data_y = self.__norm_data__(data_y, self.y_mean, self.y_std)
        data_y_BL = self.__norm_data__(data_y_BL, self.y_mean, self.y_std)
        data_y_NL = self.__norm_data__(data_y_NL, self.y_mean, self.y_std)
        data_y_GS = self.__norm_data__(data_y_GS, self.y_mean, self.y_std)
                
        pft_BL_BL = pft_BL_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_BL_NL = pft_BL_NL[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_BL_GS = pft_BL_GS[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_NL_BL = pft_NL_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)   
        pft_NL_NL = pft_NL_NL[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_NL_GS = pft_NL_GS[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_GS_BL = pft_GS_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)   
        pft_GS_NL = pft_GS_NL[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_GS_GS = pft_GS_GS[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_MIX_BL = pft_MIX_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)    
        pft_MIX_NL = pft_MIX_NL[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_MIX_GS = pft_MIX_GS[:,np.newaxis, :, :] # (N, 1, 480, 1)


        # split train and val
        if self.set_type != 2:
            idx = np.arange(data_x.shape[0])
            np.random.shuffle(idx)
            idx_val = idx[:int(len(idx)*self.val_ratio)]
            idx_train = idx[int(len(idx)*self.val_ratio):]

            idx_BL = np.arange(data_x_BL.shape[0])
            np.random.shuffle(idx_BL)
            idx_val_BL = idx_BL[:int(len(idx_BL)*self.val_ratio)]
            idx_train_BL = idx_BL[int(len(idx_BL)*self.val_ratio):]

            idx_NL = np.arange(data_x_NL.shape[0])
            np.random.shuffle(idx_NL)
            idx_val_NL = idx_NL[:int(len(idx_NL)*self.val_ratio)]
            idx_train_NL = idx_NL[int(len(idx_NL)*self.val_ratio):]

            idx_GS = np.arange(data_x_GS.shape[0])
            np.random.shuffle(idx_GS)
            idx_val_GS = idx_GS[:int(len(idx_GS)*self.val_ratio)]
            idx_train_GS = idx_GS[int(len(idx_GS)*self.val_ratio):]
            if self.set_type == 0:
                data_x = data_x[idx_train]
                data_x_BL = data_x_BL[idx_train_BL]
                data_x_NL = data_x_NL[idx_train_NL]
                data_x_GS = data_x_GS[idx_train_GS]
            
                data_y = data_y[idx_train]
                data_y_BL = data_y_BL[idx_train_BL]
                data_y_NL = data_y_NL[idx_train_NL]
                data_y_GS = data_y_GS[idx_train_GS]

                pft_BL_BL = pft_BL_BL[idx_train_BL]
                pft_BL_NL = pft_BL_NL[idx_train_BL]
                pft_BL_GS = pft_BL_GS[idx_train_BL]
                pft_NL_BL = pft_NL_BL[idx_train_NL]
                pft_NL_NL = pft_NL_NL[idx_train_NL]
                pft_NL_GS = pft_NL_GS[idx_train_NL]
                pft_GS_BL = pft_GS_BL[idx_train_GS]
                pft_GS_NL = pft_GS_NL[idx_train_GS]
                pft_GS_GS = pft_GS_GS[idx_train_GS]
                pft_MIX_BL = pft_MIX_BL[idx_train]
                pft_MIX_NL = pft_MIX_NL[idx_train]
                pft_MIX_GS = pft_MIX_GS[idx_train]
            else:
                data_x = data_x[idx_val]
                data_x_BL = data_x_BL[idx_val_BL]
                data_x_NL = data_x_NL[idx_val_NL]
                data_x_GS = data_x_GS[idx_val_GS]
            
                data_y = data_y[idx_val]
                data_y_BL = data_y_BL[idx_val_BL]
                data_y_NL = data_y_NL[idx_val_NL]
                data_y_GS = data_y_GS[idx_val_GS]

                pft_BL_BL = pft_BL_BL[idx_val_BL]
                pft_BL_NL = pft_BL_NL[idx_val_BL]
                pft_BL_GS = pft_BL_GS[idx_val_BL]
                pft_NL_BL = pft_NL_BL[idx_val_NL]
                pft_NL_NL = pft_NL_NL[idx_val_NL]
                pft_NL_GS = pft_NL_GS[idx_val_NL]
                pft_GS_BL = pft_GS_BL[idx_val_GS]
                pft_GS_NL = pft_GS_NL[idx_val_GS]
                pft_GS_GS = pft_GS_GS[idx_val_GS]
                pft_MIX_BL = pft_MIX_BL[idx_val]
                pft_MIX_NL = pft_MIX_NL[idx_val]
                pft_MIX_GS = pft_MIX_GS[idx_val]    

        self.data_x = data_x.reshape(-1, data_x.shape[2], data_x.shape[3]) # (-1, 380, 126)
        self.data_x_BL = data_x_BL.reshape(-1, data_x_BL.shape[2], data_x_BL.shape[3]) # (-1, 381, 10)
        self.data_x_NL = data_x_NL.reshape(-1, data_x_NL.shape[2], data_x_NL.shape[3]) # (-1, 381, 10)
        self.data_x_GS = data_x_GS.reshape(-1, data_x_GS.shape[2], data_x_GS.shape[3]) # (-1, 381, 10)
        self.data_y = data_y.reshape(-1, data_y.shape[2], data_y.shape[3]) # (-1, 381, 10)
        self.data_y_BL = data_y_BL.reshape(-1, data_y_BL.shape[2], data_y_BL.shape[3]) # (-1, 381, 10)
        self.data_y_NL = data_y_NL.reshape(-1, data_y_NL.shape[2], data_y_NL.shape[3]) # (-1, 381, 10)
        self.data_y_GS = data_y_GS.reshape(-1, data_y_GS.shape[2], data_y_GS.shape[3]) # (-1, 381, 10)

        # self.pft_BL = pft_BL.reshape(-1, pft_BL.shape[2], pft_BL.shape[3]) # (N, age, 1, 380, 1) to (-1, 380, 1)
        # self.pft_NL = pft_NL.reshape(-1, pft_NL.shape[2], pft_NL.shape[3]) 
        # self.pft_GS = pft_GS.reshape(-1, pft_GS.shape[2], pft_GS.shape[3]) 
        self.pft_BL_BL = pft_BL_BL.reshape(-1, pft_BL_BL.shape[2], pft_BL_BL.shape[3]) # (N, 1, 380, 1) to (-1, 380, 1)
        self.pft_BL_NL = pft_BL_NL.reshape(-1, pft_BL_NL.shape[2], pft_BL_NL.shape[3]) 
        self.pft_BL_GS = pft_BL_GS.reshape(-1, pft_BL_GS.shape[2], pft_BL_GS.shape[3]) 
        self.pft_NL_BL = pft_NL_BL.reshape(-1, pft_NL_BL.shape[2], pft_NL_BL.shape[3])
        self.pft_NL_NL = pft_NL_NL.reshape(-1, pft_NL_NL.shape[2], pft_NL_NL.shape[3]) 
        self.pft_NL_GS = pft_NL_GS.reshape(-1, pft_NL_GS.shape[2], pft_NL_GS.shape[3]) 
        self.pft_GS_BL = pft_GS_BL.reshape(-1, pft_GS_BL.shape[2], pft_GS_BL.shape[3])
        self.pft_GS_NL = pft_GS_NL.reshape(-1, pft_GS_NL.shape[2], pft_GS_NL.shape[3]) 
        self.pft_GS_GS = pft_GS_GS.reshape(-1, pft_GS_GS.shape[2], pft_GS_GS.shape[3]) 
        self.pft_MIX_BL = pft_MIX_BL.reshape(-1, pft_MIX_BL.shape[2], pft_MIX_BL.shape[3])
        self.pft_MIX_NL = pft_MIX_NL.reshape(-1, pft_MIX_NL.shape[2], pft_MIX_NL.shape[3]) 
        self.pft_MIX_GS = pft_MIX_GS.reshape(-1, pft_MIX_GS.shape[2], pft_MIX_GS.shape[3]) 
        print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, pft={self.pft_BL_BL.shape}')
        

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_x_BL = self.data_x_BL[index]
        seq_x_NL = self.data_x_NL[index]
        seq_x_GS = self.data_x_GS[index]
        seq_y = self.data_y[index]
        seq_y_BL = self.data_y_BL[index]
        seq_y_NL = self.data_y_NL[index]
        seq_y_GS = self.data_y_GS[index]
        
        # pft_bl = self.pft_BL[index]
        # pft_nl = self.pft_NL[index]
        # pft_gs = self.pft_GS[index]
        pft_BL_bl = self.pft_BL_BL[index]
        pft_BL_nl = self.pft_BL_NL[index]
        pft_BL_gs = self.pft_BL_GS[index]
        pft_NL_bl = self.pft_NL_BL[index]
        pft_NL_nl = self.pft_NL_NL[index]
        pft_NL_gs = self.pft_NL_GS[index]      
        pft_GS_bl = self.pft_GS_BL[index]
        pft_GS_nl = self.pft_GS_NL[index]
        pft_GS_gs = self.pft_GS_GS[index]
        pft_MIX_bl = self.pft_MIX_BL[index]
        pft_MIX_nl = self.pft_MIX_NL[index]
        pft_MIX_gs = self.pft_MIX_GS[index]


        return seq_x, seq_x_BL, seq_x_NL, seq_x_GS, seq_y, seq_y_BL, seq_y_NL, seq_y_GS, pft_BL_bl, pft_BL_nl, pft_BL_gs, pft_NL_bl, pft_NL_nl, pft_NL_gs, pft_GS_bl, pft_GS_nl, pft_GS_gs, pft_MIX_bl, pft_MIX_nl, pft_MIX_gs
    
    def __len__(self):
        return self.data_x.shape[0]
    
    def __sliding_window__(self, arr, window_size, stride, axis):
        # Ensure the axis is positive and within the valid range
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("Axis out of bounds")

        # Calculate the shape and strides for the sliding window
        new_shape = list(arr.shape)
        new_shape[axis] = (arr.shape[axis] - window_size) // stride + 1
        new_shape.insert(axis + 1, window_size)
        
        new_strides = list(arr.strides)
        new_strides.insert(axis + 1, new_strides[axis])
        new_strides[axis] = new_strides[axis] * stride

        # Create the sliding window view
        return np.lib.stride_tricks.as_strided(arr, shape=tuple(new_shape), strides=tuple(new_strides))


    def __sliding_window_monthly__(self, arr, window_size=2):
        if arr.ndim != 5:
            raise ValueError("Input array must have shape (age, N, years, months, features)")

        age, N, Y, M, F = arr.shape
        if M != 12:
            raise ValueError("Monthly dimension (axis=3) must be 12")
        
        # Step 1: reshape to merge years and months → (age, N, (Y+1)*12, 10)       
        arr_flat = arr.reshape(age, N, Y * M, F)

        # Step 2: sliding window on axis=2 (flattened time), window=24 (2 years)
        new_shape = (age, N, Y - window_size + 1, window_size * M, F)
        new_strides = (
            arr_flat.strides[0],
            arr_flat.strides[1],
            arr_flat.strides[2] * M,      # step forward by 12 months = 1 year
            arr_flat.strides[2],          # step inside 24 months
            arr_flat.strides[3]
        )

        arr_window = np.lib.stride_tricks.as_strided(
            arr_flat,
            shape=new_shape,
            strides=new_strides
        )
        
        return arr_window
    

    def __generate_random_walk_noise__(self, shape, std):
        random_steps = np.random.normal(loc=0.0, scale=std, size=shape)
        random_walk_noise = np.cumsum(random_steps, axis=2)  # Axis 2 corresponds to the "Y" dimension
        return random_walk_noise
    
    # Function to add random walk noise to the specified feature channels
    def __add_random_walk_noise_to_batch__(self, batch, std):
        N, years, channels, features = batch.shape  # (N, 1, 337, 10)  

        # Generate random walk noise for the specified channels
        noise_shape = (N, years, features)
        random_walk_noise = self.__generate_random_walk_noise__(noise_shape, std)

        # Create a zero tensor with the same shape as the batch
        noise_tensor = np.zeros_like(batch)
        
        # Add the random walk noise to the [:, :, :, 0, :] slice
        noise_tensor[:, :, 0, :] = random_walk_noise
        
        # Add the noise tensor to the batch
        batch_with_noise = batch + noise_tensor
        
        return batch_with_noise
    
    
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_AddPure_Extract4types_AgeIndependentCom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
                 asi=0, aei=18, val_ratio=0.1, add_noise=False, noise_std=1e-4,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        if flag == 'pred':
            flag = 'test'
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.asi = asi
        self.aei = aei
        self.val_ratio = val_ratio
        self.add_noise = add_noise
        self.noise_std = noise_std
         
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        
        self.__read_data__()

    def __read_data__(self):
        stat_raw = np.load(os.path.join(self.root_path, self.stat_path))
        self.x_mean = stat_raw['x_mean']
        self.x_std = stat_raw['x_std']
        self.y_mean = stat_raw['y_mean']
        self.y_std = stat_raw['y_std']
        
        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        # x = (N, Y, 12, 136)
        # y = (age, N, Y+1, 12, 10)
        pft_raw = np.load(os.path.join(self.root_path, self.pft_path))
        # (N, 18, Y+1, 12)

        if self.set_type == 2: 
            data_x = data_raw[f'x_test'] # (N, 28, 12, 136)
            data_y = data_raw[f'y_test'] # (18, N, 29, 12, 10)
            data_y_agesum = data_raw[f'y_test_agesum'] # (N, 29, 12, 10)

            pft_MIX_BL = pft_raw['test_pft_BL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] # (N, 18, 29, 12) to (N, age, 28, 12, 1)
            pft_MIX_NL = pft_raw['test_pft_NL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis]
            pft_MIX_GS = pft_raw['test_pft_GS'][:, self.asi:self.aei, 1:, ...][..., np.newaxis]

            pft_MIX_BL_agesum = pft_raw['test_pft_BL_agesum'][:, 1:, ...][..., np.newaxis] # (N, 29, 12) to (N, 28, 12, 1)
            pft_MIX_NL_agesum = pft_raw['test_pft_NL_agesum'][:, 1:, ...][..., np.newaxis]
            pft_MIX_GS_agesum = pft_raw['test_pft_GS_agesum'][:, 1:, ...][..., np.newaxis]

            AgeWeight = pft_raw['AgeWeight_test'].swapaxes(0, 1) # (N, 18)


        else: 
            data_x = data_raw[f'x_train'] # (N, 28, 12, 136)
            data_y = data_raw[f'y_train'] # (18, N, 29, 12, 10)
            data_y_agesum = data_raw[f'y_train_agesum'] # (N, 29, 12, 10)

            pft_MIX_BL = pft_raw['train_pft_BL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] # (N, 18, Y+1, 12) to (N, age, 28, 12, 1)
            pft_MIX_NL = pft_raw['train_pft_NL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis]
            pft_MIX_GS = pft_raw['train_pft_GS'][:, self.asi:self.aei, 1:, ...][..., np.newaxis]

            pft_MIX_BL_agesum = pft_raw['train_pft_BL_agesum'][:, 1:, ...][..., np.newaxis] # (N, 29, 12) to (N, 28, 12, 1)
            pft_MIX_NL_agesum = pft_raw['train_pft_NL_agesum'][:, 1:, ...][..., np.newaxis]
            pft_MIX_GS_agesum = pft_raw['train_pft_GS_agesum'][:, 1:, ...][..., np.newaxis]

            AgeWeight = pft_raw['AgeWeight_train'].swapaxes(0, 1) # (N, 18)


        mask = np.ones_like(data_y, dtype=bool) 
        mask[..., 7] = False                    
        data_y[mask & (data_y < 0)] = 0


        data_y = data_y[self.asi:self.aei, ...] # (age, N, 29, 12, 10)        
        # # prepare inital and target pair 
        # data_y = self.__sliding_window_monthly__(data_y, 2) # (age, N, Y, 24, 10)
        data_y_arr = data_y.copy() # (age, N, 29, 12, 10) 
        data_y = data_y.reshape(data_y_arr.shape[0], data_y_arr.shape[1], data_y_arr.shape[2]*data_y_arr.shape[3], data_y_arr.shape[4]) # (age, N, 29*12, 10)
        data_y = data_y[:, :, 11:, :] # (age, N, 337, 10)
        data_y = data_y[:, :, np.newaxis, :, :] # (age, N, 1, 337, 10)   
        data_y = np.transpose(data_y, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
        # add random-walk noise to target
        if self.add_noise:
            data_y =  self.__add_random_walk_noise_to_batch__(data_y, self.noise_std)

        # data_y_agesum[data_y_agesum < 0] = 0 # (N, 29, 12, 10)
        mask = np.ones_like(data_y_agesum, dtype=bool) 
        mask[..., 7] = False                    
        data_y_agesum[mask & (data_y_agesum < 0)] = 0
        data_y_agesum_arr = data_y_agesum.copy() # (N, 29, 12, 10)
        data_y_agesum = data_y_agesum.reshape(data_y_agesum_arr.shape[0], data_y_agesum_arr.shape[1]*data_y_agesum_arr.shape[2], data_y_agesum_arr.shape[3]) # (N, 29*12, 10)
        data_y_agesum = data_y_agesum[:, 11:, :] # (N, 337, 10)
        data_y_agesum = data_y_agesum[:, np.newaxis, :, :] # (N, 1, 337, 10)        
        # data_y_agesum = np.transpose(data_y_agesum, (1,0,2,3,4))
        # add random-walk noise to target
        if self.add_noise:
            data_y_agesum =  self.__add_random_walk_noise_to_batch__(data_y_agesum, self.noise_std)
                 
        # dup x for matching ages 
        data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1) # (N, age, 28, 12, 136)
        data_x_arr = data_x.copy()
        data_x = data_x.reshape(data_x_arr.shape[0], data_x_arr.shape[1], data_x_arr.shape[2]*data_x_arr.shape[3], data_x_arr.shape[4]) # (N, age, 28*12, 136)
        data_x = data_x[:, :, np.newaxis, :, :] # (N, age, 1, 28*12, 136)            
        # normalize data
        data_x = self.__norm_data__(data_x, self.x_mean, self.x_std)

        data_y = self.__norm_data__(data_y, self.y_mean, self.y_std)
        data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean, self.y_std)

        # (N, age, 28, 12, 1)
        pft_MIX_BL = pft_MIX_BL.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1) # (N, age, 336, 1)
        pft_MIX_BL = pft_MIX_BL[:, :, np.newaxis, :, :] # (N, age, 1, 336, 1)
        pft_MIX_NL = pft_MIX_NL.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1)
        pft_MIX_NL = pft_MIX_NL[:, :, np.newaxis, :, :]
        pft_MIX_GS = pft_MIX_GS.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1)
        pft_MIX_GS = pft_MIX_GS[:, :, np.newaxis, :, :]

        # (N, 28, 12, 1)
        pft_MIX_BL_agesum = pft_MIX_BL_agesum.reshape(data_y_arr.shape[1], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1) # (N, 336, 1)
        pft_MIX_BL_agesum = pft_MIX_BL_agesum[:, np.newaxis, :, :] # (N, 1, 336, 1)
        pft_MIX_NL_agesum = pft_MIX_NL_agesum.reshape(data_y_arr.shape[1], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1)
        pft_MIX_NL_agesum = pft_MIX_NL_agesum[:, np.newaxis, :, :]
        pft_MIX_GS_agesum = pft_MIX_GS_agesum.reshape(data_y_arr.shape[1], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1)
        pft_MIX_GS_agesum = pft_MIX_GS_agesum[:, np.newaxis, :, :]

        # split train and val
        if self.set_type != 2:
            idx = np.arange(data_x.shape[0])
            np.random.shuffle(idx)
            idx_val = idx[:int(len(idx)*self.val_ratio)]
            idx_train = idx[int(len(idx)*self.val_ratio):]

            if self.set_type == 0:
                data_x = data_x[idx_train] # (N, age, 1, 336, 136)        
                data_y = data_y[idx_train] # (N, age, 1, 337, 10)   
                data_y_agesum = data_y_agesum[idx_train] # (N, 1, 337, 10)

                pft_MIX_BL = pft_MIX_BL[idx_train] # (N, age, 1, 336, 1)
                pft_MIX_NL = pft_MIX_NL[idx_train]
                pft_MIX_GS = pft_MIX_GS[idx_train]
                pft_MIX_BL_agesum = pft_MIX_BL_agesum[idx_train] # (N, 1, 336, 1)
                pft_MIX_NL_agesum = pft_MIX_NL_agesum[idx_train]
                pft_MIX_GS_agesum = pft_MIX_GS_agesum[idx_train]

                AgeWeight = AgeWeight[idx_train] # (N, 18)
            else:
                data_x = data_x[idx_val]            
                data_y = data_y[idx_val]
                data_y_agesum = data_y_agesum[idx_val]

                pft_MIX_BL = pft_MIX_BL[idx_val]
                pft_MIX_NL = pft_MIX_NL[idx_val]
                pft_MIX_GS = pft_MIX_GS[idx_val]
                pft_MIX_BL_agesum = pft_MIX_BL_agesum[idx_val]
                pft_MIX_NL_agesum = pft_MIX_NL_agesum[idx_val]
                pft_MIX_GS_agesum = pft_MIX_GS_agesum[idx_val]

                AgeWeight = AgeWeight[idx_val]
  
        # self.data_x = data_x.reshape(-1, data_x.shape[3], data_x.shape[4]) # (N, age, 1, 336, 136) to (-1, 336, 136)
        # self.data_y = data_y.reshape(-1, data_y.shape[3], data_y.shape[4]) # (N, age, 1, 337, 10) to (-1, 337, 10)
        # self.pft_MIX_BL = pft_MIX_BL.reshape(-1, pft_MIX_BL.shape[3], pft_MIX_BL.shape[4]) # (N, age, 1, 336, 1) to (-1, 336, 1)
        # self.pft_MIX_NL = pft_MIX_NL.reshape(-1, pft_MIX_NL.shape[3], pft_MIX_NL.shape[4]) 
        # self.pft_MIX_GS = pft_MIX_GS.reshape(-1, pft_MIX_GS.shape[3], pft_MIX_GS.shape[4]) 
        self.data_x = np.squeeze(data_x) # (N, age, 1, 336, 136) to (N, age, 336, 136)
        self.data_y = np.squeeze(data_y) # (N, age, 1, 337, 10) to (N, age, 337, 10)
        self.pft_MIX_BL = pft_MIX_BL.squeeze(2) # (N, age, 1, 336, 1) to (N, age, 336, 1)
        self.pft_MIX_NL = pft_MIX_NL.squeeze(2)
        self.pft_MIX_GS = pft_MIX_GS.squeeze(2)
       

        self.data_y_agesum = data_y_agesum.reshape(-1, data_y_agesum.shape[2], data_y_agesum.shape[3]) # (N, 1, 337, 10) to (-1, 337, 10)
        self.pft_MIX_BL_agesum = pft_MIX_BL_agesum.reshape(-1, pft_MIX_BL_agesum.shape[2], pft_MIX_BL_agesum.shape[3]) # (N, 1, 336, 1) to (-1, 336, 1)
        self.pft_MIX_NL_agesum = pft_MIX_NL_agesum.reshape(-1, pft_MIX_NL_agesum.shape[2], pft_MIX_NL_agesum.shape[3])
        self.pft_MIX_GS_agesum = pft_MIX_GS_agesum.reshape(-1, pft_MIX_GS_agesum.shape[2], pft_MIX_GS_agesum.shape[3])
        
        self.AgeWeight = AgeWeight # (N, 18)

        print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, pft={self.pft_MIX_BL.shape}, data_y_agesum={self.data_y_agesum.shape}, pft_agesum={self.pft_MIX_BL_agesum.shape},  AgeWeight={self.AgeWeight.shape}')
        

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_y_agesum = self.data_y_agesum[index]

        seq_pft_BL = self.pft_MIX_BL[index]
        seq_pft_NL = self.pft_MIX_NL[index]
        seq_pft_GS = self.pft_MIX_GS[index]
        
        seq_pft_BL_agesum = self.pft_MIX_BL_agesum[index]
        seq_pft_NL_agesum = self.pft_MIX_NL_agesum[index]
        seq_pft_GS_agesum = self.pft_MIX_GS_agesum[index]
        seq_aw = self.AgeWeight[index]

        return seq_x, seq_y, seq_y_agesum, seq_pft_BL, seq_pft_NL, seq_pft_GS, seq_pft_BL_agesum, seq_pft_NL_agesum, seq_pft_GS_agesum, seq_aw
    
    def __len__(self):
        return self.data_x.shape[0]
    
    def __sliding_window__(self, arr, window_size, stride, axis):
        # Ensure the axis is positive and within the valid range
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("Axis out of bounds")

        # Calculate the shape and strides for the sliding window
        new_shape = list(arr.shape)
        new_shape[axis] = (arr.shape[axis] - window_size) // stride + 1
        new_shape.insert(axis + 1, window_size)
        
        new_strides = list(arr.strides)
        new_strides.insert(axis + 1, new_strides[axis])
        new_strides[axis] = new_strides[axis] * stride

        # Create the sliding window view
        return np.lib.stride_tricks.as_strided(arr, shape=tuple(new_shape), strides=tuple(new_strides))


    def __sliding_window_monthly__(self, arr, window_size=2):
        if arr.ndim != 5:
            raise ValueError("Input array must have shape (age, N, years, months, features)")

        age, N, Y, M, F = arr.shape
        if M != 12:
            raise ValueError("Monthly dimension (axis=3) must be 12")
        
        # Step 1: reshape to merge years and months → (age, N, (Y+1)*12, 10)       
        arr_flat = arr.reshape(age, N, Y * M, F)

        # Step 2: sliding window on axis=2 (flattened time), window=24 (2 years)
        new_shape = (age, N, Y - window_size + 1, window_size * M, F)
        new_strides = (
            arr_flat.strides[0],
            arr_flat.strides[1],
            arr_flat.strides[2] * M,      # step forward by 12 months = 1 year
            arr_flat.strides[2],          # step inside 24 months
            arr_flat.strides[3]
        )

        arr_window = np.lib.stride_tricks.as_strided(
            arr_flat,
            shape=new_shape,
            strides=new_strides
        )
        
        return arr_window
    

    def __generate_random_walk_noise__(self, shape, std):
        random_steps = np.random.normal(loc=0.0, scale=std, size=shape)
        random_walk_noise = np.cumsum(random_steps, axis=2)  # Axis 2 corresponds to the "Y" dimension
        return random_walk_noise
    
    # Function to add random walk noise to the specified feature channels
    def __add_random_walk_noise_to_batch__(self, batch, std):
        N, years, channels, features = batch.shape  # (N, 1, 337, 10)  

        # Generate random walk noise for the specified channels
        noise_shape = (N, years, features)
        random_walk_noise = self.__generate_random_walk_noise__(noise_shape, std)

        # Create a zero tensor with the same shape as the batch
        noise_tensor = np.zeros_like(batch)
        
        # Add the random walk noise to the [:, :, :, 0, :] slice
        noise_tensor[:, :, 0, :] = random_walk_noise
        
        # Add the noise tensor to the batch
        batch_with_noise = batch + noise_tensor
        
        return batch_with_noise
    
    
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
class Dataset_ED_ALLAGE_PFT_Prediction(Dataset):
    def __init__(self, root_path, flag='train', stage='step1', size=None, 
                 features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
                 asi=0, aei=18, val_ratio=0.1, add_noise=False, noise_std=1e-4,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        if flag == 'pred':
            flag = 'test'
        self.flag = flag
        self.stage = stage
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.asi = asi
        self.aei = aei
        self.val_ratio = val_ratio
        self.add_noise = add_noise
        self.noise_std = noise_std
         
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        
        self.__read_data__()

    def __read_data__(self):
        stat_raw = np.load(os.path.join(self.root_path, self.stat_path))
        self.x_mean = stat_raw['x_mean']
        self.x_std = stat_raw['x_std']
        # self.y_mean = stat_raw['y_mean']
        # self.y_std = stat_raw['y_std']
        
        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        # x = (N, Y, 12, 136)
        # y = (age, N, Y+1, 12, 10)
        pft_raw = np.load(os.path.join(self.root_path, self.pft_path))
        # (N, 18, Y+1, 12)

        if self.stage == 'step1': 
            if self.set_type == 2: 
                data_x = data_raw[f'x_test_all'] # (852, 28, 136*12)
                pft_MIX_BL = pft_raw['test_pft_BL'][..., np.newaxis] # (18, 852, 29) to (18, N, 29, 1) 
                pft_MIX_NL = pft_raw['test_pft_NL'][..., np.newaxis]
                pft_MIX_GS = pft_raw['test_pft_GS'][..., np.newaxis]

            else: 
                data_x = data_raw[f'x_train_all'] # (3373, 28, 136*12)

                pft_MIX_BL = pft_raw['train_pft_BL'][..., np.newaxis]
                pft_MIX_NL = pft_raw['train_pft_NL'][..., np.newaxis]
                pft_MIX_GS = pft_raw['train_pft_GS'][..., np.newaxis]


            # dup x for matching ages 
            # data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1)  # (2825, Y, 12, 136)
            data_x_arr = data_x.copy()  # (852, 28, 136*12)
            # data_x = data_x.reshape(data_x_arr.shape[0], data_x_arr.shape[1]*data_x_arr.shape[2], data_x_arr.shape[3]) # (N, Y *12, 136)
            data_x = data_x[:, np.newaxis, :, :] # (N, 1, 28, 136*12)
            # normalize data
            # data_x = self.__norm_data__(data_x, self.x_mean, self.x_std) # (N, 1, 28, 136*12)
            data_x = self.__norm_data__(data_x, np.repeat(self.x_mean, 12), np.repeat(self.x_std, 12))
                    
            pft_MIX_BL = pft_MIX_BL[:,:,np.newaxis, :, :] # (18, N, 1, 29, 1)        pft_BL = pft_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
            pft_MIX_NL = pft_MIX_NL[:,:,np.newaxis, :, :] # (18, N, 1, 29, 1)
            pft_MIX_GS = pft_MIX_GS[:,:,np.newaxis, :, :] # (18, N, 1, 29, 1)

            # data_y_pft = torch.cat([pft_MIX_BL, pft_MIX_NL, pft_MIX_GS], dim=3) # (N, 1, 29, 3)
            data_y_pft = torch.cat([
                torch.from_numpy(pft_MIX_BL),
                torch.from_numpy(pft_MIX_NL),
                torch.from_numpy(pft_MIX_GS)
            ], dim=4) # (18, N, 1, 29, 3)

            # split train and val
            if self.set_type != 2:
                idx = np.arange(data_x.shape[0])
                np.random.shuffle(idx)
                idx_val = idx[:int(len(idx)*self.val_ratio)]
                idx_train = idx[int(len(idx)*self.val_ratio):]

                if self.set_type == 0:
                    data_x = data_x[idx_train]  # (N, 1, 28, 136*12)           
                    data_y = data_y_pft[:, idx_train, :, :, :]  # (18, N, 1, 29, 3)
                else:
                    data_x = data_x[idx_val]
                    data_y = data_y_pft[:, idx_val, :, :, :]
            else:
                data_y = data_y_pft

            self.data_x = data_x.reshape(-1, data_x.shape[2], data_x.shape[3]) # (N, 28, 136*12)
            self.data_y = data_y.reshape(data_y.shape[0], -1, data_y.shape[3], data_y.shape[4]) # (18, N, 29, 3)
            print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}')

        elif self.stage == 'step2': 
            if self.set_type == 2: 
                data_x = data_raw[f'x_test_all'] # (852, 28, 136*12)
                pft_MIX_BL = pft_raw['test_pft_BL'][..., np.newaxis] # (18, 852, 29) to (18, N, 29, 1) 
                pft_MIX_NL = pft_raw['test_pft_NL'][..., np.newaxis]
                pft_MIX_GS = pft_raw['test_pft_GS'][..., np.newaxis]
                pft_ESACCI_BL = data_raw['ESA_BL_all_test'][..., np.newaxis] # (852, 29) to (N, 29, 1) 
                pft_ESACCI_NL = data_raw['ESA_NL_all_test'][..., np.newaxis]
                pft_ESACCI_GS = data_raw['ESA_GS_all_test'][..., np.newaxis]
                AgeWeight= data_raw['AgeWeight_test']# (18, N)

            else: 
                data_x = data_raw[f'x_train_all'] # (3373, 28, 136*12)

                pft_MIX_BL = pft_raw['train_pft_BL'][..., np.newaxis]
                pft_MIX_NL = pft_raw['train_pft_NL'][..., np.newaxis]
                pft_MIX_GS = pft_raw['train_pft_GS'][..., np.newaxis]
                pft_ESACCI_BL = data_raw['ESA_BL_all_train'][..., np.newaxis] # (3373, 29) to (N, 29, 1) 
                pft_ESACCI_NL = data_raw['ESA_NL_all_train'][..., np.newaxis]
                pft_ESACCI_GS = data_raw['ESA_GS_all_train'][..., np.newaxis]
                AgeWeight = data_raw['AgeWeight_train']

            # dup x for matching ages 
            # data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1)  # (2825, Y, 12, 136)
            data_x_arr = data_x.copy()  # (852, 28, 136*12)
            # data_x = data_x.reshape(data_x_arr.shape[0], data_x_arr.shape[1]*data_x_arr.shape[2], data_x_arr.shape[3]) # (N, Y *12, 136)
            data_x = data_x[:, np.newaxis, :, :] # (N, 1, 28, 136*12)
            # normalize data
            # data_x = self.__norm_data__(data_x, self.x_mean, self.x_std) # (N, 1, 28, 136*12)
            data_x = self.__norm_data__(data_x, np.repeat(self.x_mean, 12), np.repeat(self.x_std, 12))
                    
            pft_MIX_BL = pft_MIX_BL[:,:,np.newaxis, :, :] # (18, N, 1, 29, 1)        pft_BL = pft_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
            pft_MIX_NL = pft_MIX_NL[:,:,np.newaxis, :, :] # (18, N, 1, 29, 1)
            pft_MIX_GS = pft_MIX_GS[:,:,np.newaxis, :, :] # (18, N, 1, 29, 1)
            # data_y_pft = torch.cat([pft_MIX_BL, pft_MIX_NL, pft_MIX_GS], dim=3) # (N, 1, 29, 3)
            data_y_pft = torch.cat([
                torch.from_numpy(pft_MIX_BL),
                torch.from_numpy(pft_MIX_NL),
                torch.from_numpy(pft_MIX_GS)
            ], dim=4) # (18, N, 1, 29, 3)

            pft_ESACCI_BL = pft_ESACCI_BL[:,np.newaxis, :, :] # (N, 1, 29, 1)        pft_BL = pft_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
            pft_ESACCI_NL = pft_ESACCI_NL[:,np.newaxis, :, :] 
            pft_ESACCI_GS = pft_ESACCI_GS[:,np.newaxis, :, :] 
            # data_y_pft = torch.cat([pft_MIX_BL, pft_MIX_NL, pft_MIX_GS], dim=3) # (N, 1, 29, 3)
            data_y_ESACCI_pft = torch.cat([
                torch.from_numpy(pft_ESACCI_BL),
                torch.from_numpy(pft_ESACCI_NL),
                torch.from_numpy(pft_ESACCI_GS)
            ], dim=3) # (N, 1, 29, 3)


            # split train and val
            if self.set_type != 2:
                idx = np.arange(data_x.shape[0])
                np.random.shuffle(idx)
                idx_val = idx[:int(len(idx)*self.val_ratio)]
                idx_train = idx[int(len(idx)*self.val_ratio):]

                if self.set_type == 0:
                    data_x = data_x[idx_train]  # (N, 1, 28, 136*12)           
                    data_y = data_y_pft[:, idx_train, :, :, :]  # (18, N, 1, 29, 3)
                    data_y_ESACCI = data_y_ESACCI_pft[idx_train, :, :, :]  # (N, 1, 29, 3)
                    data_aw = AgeWeight[:, idx_train] # (18, N)
                else:
                    data_x = data_x[idx_val]
                    data_y = data_y_pft[:, idx_val, :, :, :]
                    data_y_ESACCI = data_y_ESACCI_pft[idx_val, :, :, :]  # (N, 1, 29, 3)
                    data_aw = AgeWeight[:, idx_val] # (18, N)
            else:
                data_y = data_y_pft
                data_y_ESACCI = data_y_ESACCI_pft
                data_aw = AgeWeight

            self.data_x = data_x.reshape(-1, data_x.shape[2], data_x.shape[3]) # (N, 28, 136*12)
            self.data_y = data_y.reshape(data_y.shape[0], -1, data_y.shape[3], data_y.shape[4]) # (18, N, 29, 3)
            self.data_y_ESACCI = data_y_ESACCI.reshape(-1, data_y_ESACCI.shape[2], data_y_ESACCI.shape[3]) # (N, 29, 3)
            self.data_aw = data_aw
            print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, y_esacci={self.data_y_ESACCI.shape}, age_weight={self.data_aw.shape}') 
        
    def __getitem__(self, index):
        if self.stage == 'step1':
            seq_x = self.data_x[index]
            seq_y = self.data_y[:, index,...]
            return seq_x, seq_y
        elif self.stage == 'step2':    
            seq_x = self.data_x[index]
            seq_y = self.data_y[:, index,...]
            seq_y_ESACCI = self.data_y_ESACCI[index]
            seq_aw = self.data_aw[:, index]
            return seq_x, seq_y, seq_y_ESACCI, seq_aw
    
    def __len__(self):
        return self.data_x.shape[0]
    
    def __sliding_window__(self, arr, window_size, stride, axis):
        # Ensure the axis is positive and within the valid range
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("Axis out of bounds")

        # Calculate the shape and strides for the sliding window
        new_shape = list(arr.shape)
        new_shape[axis] = (arr.shape[axis] - window_size) // stride + 1
        new_shape.insert(axis + 1, window_size)
        
        new_strides = list(arr.strides)
        new_strides.insert(axis + 1, new_strides[axis])
        new_strides[axis] = new_strides[axis] * stride

        # Create the sliding window view
        return np.lib.stride_tricks.as_strided(arr, shape=tuple(new_shape), strides=tuple(new_strides))


    def __sliding_window_monthly__(self, arr, window_size=2):
        if arr.ndim != 5:
            raise ValueError("Input array must have shape (age, N, years, months, features)")

        age, N, Y, M, F = arr.shape
        if M != 12:
            raise ValueError("Monthly dimension (axis=3) must be 12")
        
        # Step 1: reshape to merge years and months → (age, N, (Y+1)*12, 10)       
        arr_flat = arr.reshape(age, N, Y * M, F)

        # Step 2: sliding window on axis=2 (flattened time), window=24 (2 years)
        new_shape = (age, N, Y - window_size + 1, window_size * M, F)
        new_strides = (
            arr_flat.strides[0],
            arr_flat.strides[1],
            arr_flat.strides[2] * M,      # step forward by 12 months = 1 year
            arr_flat.strides[2],          # step inside 24 months
            arr_flat.strides[3]
        )

        arr_window = np.lib.stride_tricks.as_strided(
            arr_flat,
            shape=new_shape,
            strides=new_strides
        )
        
        return arr_window
    

    def __generate_random_walk_noise__(self, shape, std):
        random_steps = np.random.normal(loc=0.0, scale=std, size=shape)
        random_walk_noise = np.cumsum(random_steps, axis=2)  # Axis 2 corresponds to the "Y" dimension
        return random_walk_noise
    
    # Function to add random walk noise to the specified feature channels
    def __add_random_walk_noise_to_batch__(self, batch, std):
        N, years, channels, features = batch.shape  # (N, 1, 337, 10)  

        # Generate random walk noise for the specified channels
        noise_shape = (N, years, features)
        random_walk_noise = self.__generate_random_walk_noise__(noise_shape, std)

        # Create a zero tensor with the same shape as the batch
        noise_tensor = np.zeros_like(batch)
        
        # Add the random walk noise to the [:, :, :, 0, :] slice
        noise_tensor[:, :, 0, :] = random_walk_noise
        
        # Add the noise tensor to the batch
        batch_with_noise = batch + noise_tensor
        
        return batch_with_noise
    
    
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
  

class Dataset_AddPure_Extract4types_CombineWith_PFT_Prediction(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
                 asi=0, aei=18, val_ratio=0.1, add_noise=False, noise_std=1e-4,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        if flag == 'pred':
            flag = 'test'
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.asi = asi
        self.aei = aei
        self.val_ratio = val_ratio
        self.add_noise = add_noise
        self.noise_std = noise_std
         
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        
        self.__read_data__()

    def __read_data__(self):
        stat_raw = np.load(os.path.join(self.root_path, self.stat_path))
        self.x_mean = stat_raw['x_mean']
        self.x_std = stat_raw['x_std']
        self.y_mean = stat_raw['y_mean']
        self.y_std = stat_raw['y_std']
        
        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        # x = (N, Y, 12, 136)
        # y = (age, N, Y+1, 12, 10)
        pft_raw = np.load(os.path.join(self.root_path, self.pft_path))
        # (N, 18, Y+1, 12)

        if self.set_type == 2: 
            data_x = data_raw[f'x_test'] # (N, 28, 12, 136)
            data_y = data_raw[f'y_test'] # (18, N, 29, 12, 10)
            # pft_BL = pft_raw['test_BL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] # (N, 18, Y+1, 12) to (N, age, Y+1, 12, 1) 
            # pft_NL = pft_raw['test_NL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            # pft_GS = pft_raw['test_GS'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            pft_MIX_BL = pft_raw['ESA_BL_test'][..., np.newaxis] # (N, 29) to (N, 29, 1)
            pft_MIX_NL = pft_raw['ESA_NL_test'][..., np.newaxis]
            pft_MIX_GS = pft_raw['ESA_GS_test'][..., np.newaxis]
            pft_MIX_VG = pft_raw['Tree_Cover_test'][..., np.newaxis]
            AgeWeight = pft_raw['AgeWeight_test'] # (18, N)
            InSitu = pft_raw['InSitu_test'] # (N, 29, 12, 3)
            pft_ED_BL = pft_raw['test_pft_BL'][..., np.newaxis] # (N, 18, 29) to (N, 18, 29, 1)
            pft_ED_NL = pft_raw['test_pft_NL'][..., np.newaxis]
            pft_ED_GS = pft_raw['test_pft_GS'][..., np.newaxis]
        else: 
            data_x = data_raw[f'x_train'] # (46, 28, 12, 136)
            data_y = data_raw[f'y_train'] # (18, 46, 29, 12, 10)
            # pft_BL = pft_raw['train_BL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] # (N, age, Y, 12, 1)
            # pft_NL = pft_raw['train_NL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            # pft_GS = pft_raw['train_GS'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            pft_MIX_BL = pft_raw['ESA_BL_train'][..., np.newaxis]
            pft_MIX_NL = pft_raw['ESA_NL_train'][..., np.newaxis]
            pft_MIX_GS = pft_raw['ESA_GS_train'][..., np.newaxis]
            pft_MIX_VG = pft_raw['Tree_Cover_train'][..., np.newaxis]
            AgeWeight = pft_raw['AgeWeight_train']
            InSitu = pft_raw['InSitu_train']
            pft_ED_BL = pft_raw['train_pft_BL'][..., np.newaxis] # (N, 18, 29) to (N, 18, 29, 1)
            pft_ED_NL = pft_raw['train_pft_NL'][..., np.newaxis]
            pft_ED_GS = pft_raw['train_pft_GS'][..., np.newaxis]


        data_x_12months = data_x.copy() # (N, 28, 12, 136)
        data_x_12months = self.__norm_data__(data_x_12months, self.x_mean, self.x_std)
        data_x_12months = np.transpose(data_x_12months, (0, 1, 3, 2)) # (N, 28, 136, 12)
        data_x_12months_arr = data_x_12months.copy() 
        data_x_12months = data_x_12months.reshape(data_x_12months_arr.shape[0], data_x_12months_arr.shape[1], data_x_12months_arr.shape[2]*data_x_12months_arr.shape[3]) # (N, 28, 136*12)
        
        # dup x for matching ages 
        data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1) # (N, age, 28, 12, 136)
        data_x_arr = data_x.copy() # (N, age, 28, 12, 136)
        data_x = data_x.reshape(data_x_arr.shape[0], data_x_arr.shape[1], data_x_arr.shape[2]*data_x_arr.shape[3], data_x_arr.shape[4]) # (N, age, 336, 136)
        data_x = data_x[:, :, np.newaxis, :, :] # (N, age, 1, 28 *12, 136)
        # normalize data
        data_x = self.__norm_data__(data_x, self.x_mean, self.x_std)  # (N, age, 1, 336, 136)      

        mask = np.ones_like(data_y, dtype=bool) 
        mask[..., 7] = False                    
        data_y[mask & (data_y < 0)] = 0

        data_y = data_y[self.asi:self.aei, ...] # (age, N, 29, 12, 10)       
        # # prepare inital and target pair 
        # data_y = self.__sliding_window_monthly__(data_y, 2) # (age, N, Y, 24, 10)
        data_y_arr = data_y.copy() # (age, N, 29, 12, 10)   
        data_y = data_y.reshape(data_y_arr.shape[0], data_y_arr.shape[1], data_y_arr.shape[2]*data_y_arr.shape[3], data_y_arr.shape[4]) # (age, N, 348, 10)
        data_y = data_y[:, :, 11:, :] # (age, N, 337, 10)
        data_y = data_y[:, :, np.newaxis, :, :] # (age, N, 1, 337, 10)        
        data_y = np.transpose(data_y, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
        # add random-walk noise to target
        if self.add_noise:
            data_y =  self.__add_random_walk_noise_to_batch__(data_y, self.noise_std)
                 
        data_y = self.__norm_data__(data_y, self.y_mean, self.y_std)

        # InSitu_arr = InSitu[...,0].copy() # (N, 29, 12)
        mask = np.ones_like(InSitu, dtype=bool) 
        mask[..., 2] = False                    
        InSitu[mask & (InSitu < 0)] = 0
        InSitu_arr = InSitu

        InSitu = InSitu.reshape(InSitu_arr.shape[0], InSitu_arr.shape[1]*InSitu_arr.shape[2],InSitu_arr.shape[3])  # (46, 29*12) (46, 348)
        InSitu = InSitu[:, 11:,:] # (N, 337,3)
        InSitu = self.__norm_data__(InSitu, self.y_mean[[4, 9, 7]], self.y_std[[4, 9, 7]])
        
        pft_MIX_BL = pft_MIX_BL[:,np.newaxis, :, :] # (N, 1, 29, 1)        pft_BL = pft_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_MIX_NL = pft_MIX_NL[:,np.newaxis, :, :]
        pft_MIX_GS = pft_MIX_GS[:,np.newaxis, :, :]
        # pft_MIX_VG = pft_MIX_VG[:,np.newaxis, :, :]
        # pft_NL = pft_NL.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1) # (N, age, 480, 1)
        # pft_NL = pft_NL[:, :, np.newaxis, :, :] # (N, age, 1, 480, 1)
        # pft_GS = pft_GS.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1) # (N, age, 480, 1)
        # pft_GS = pft_GS[:, :, np.newaxis, :, :] # (N, age, 1, 480, 1)

        data_y_pft = torch.cat([
            torch.from_numpy(pft_MIX_BL),
            torch.from_numpy(pft_MIX_NL),
            torch.from_numpy(pft_MIX_GS)
        ], dim=3) # (N, 1, 29, 3)  

        data_y_ed_pft = torch.cat([
            torch.from_numpy(pft_ED_BL),
            torch.from_numpy(pft_ED_NL),
            torch.from_numpy(pft_ED_GS)
        ], dim=3) # (N, 18, 29, 3)

        AgeWeight = AgeWeight[self.asi:self.aei, ...].swapaxes(0, 1) # (N, 18)

        # split train and val
        if self.set_type != 2:
            idx = np.arange(data_x.shape[0])
            np.random.shuffle(idx)

            val_count = int(len(idx) * self.val_ratio)
            if val_count < 1:
                val_count = 1
            idx_val = idx[:val_count]
            idx_train = idx[val_count:]

            if self.set_type == 0:
                data_x = data_x[idx_train]           
                data_y = data_y[idx_train]
                data_x_12months = data_x_12months[idx_train]
                # pft_MIX_BL = pft_MIX_BL[idx_train] # (N, 1, 29, 1)        pft_BL = pft_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
                # pft_MIX_NL = pft_MIX_NL[idx_train]
                # pft_MIX_GS = pft_MIX_GS[idx_train]                
                data_y_pft = data_y_pft[idx_train]
                pft_MIX_VG = pft_MIX_VG[idx_train]
                AgeWeight = AgeWeight[idx_train]
                InSitu = InSitu[idx_train]
                data_y_ed_pft = data_y_ed_pft[idx_train]
            else:
                data_x = data_x[idx_val] # (N, age, 1, 336, 136)         
                data_y = data_y[idx_val] # (N, age, 1, 337, 10)  
                data_x_12months = data_x_12months[idx_val] # (N, 28, 136*12)
                # pft_MIX_BL = pft_MIX_BL[idx_val] # (N, 1, 29, 1)        pft_BL = pft_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
                # pft_MIX_NL = pft_MIX_NL[idx_val]
                # pft_MIX_GS = pft_MIX_GS[idx_val]     
                data_y_pft = data_y_pft[idx_val] # (N, 1, 29, 3)
                pft_MIX_VG = pft_MIX_VG [idx_val]
                AgeWeight = AgeWeight[idx_val] # (N, age)
                InSitu = InSitu[idx_val] # (N, 337) 
                data_y_ed_pft = data_y_ed_pft[idx_val]  # (N, 18, 29, 3) 

        self.data_x = np.squeeze(data_x, axis=2) # (N, age, 336, 136)   
        self.data_y = np.squeeze(data_y, axis=2) # (N, age, 337, 10)
        # self.pft_MIX_BL = pft_MIX_BL.reshape(-1, pft_MIX_BL.shape[2], pft_MIX_BL.shape[3]) # (N, 29, 1)
        # self.pft_MIX_BL = 
        # pft_MIX_BL.squeeze(2) # (N, age, 1, 336, 1) to (N, age, 336, 1)
        # self.pft_MIX_NL = pft_MIX_NL.squeeze(2)
        # self.pft_MIX_GS = pft_MIX_GS.squeeze(2)

        self.data_x_12months = data_x_12months # (N, 28, 136*12)   
        self.data_y_pft = np.squeeze(data_y_pft) # (N, 29, 3) 
        self.pft_MIX_VG = pft_MIX_VG # (N, 29, 1) 
        self.AgeWeight = AgeWeight # (N, age)
        self.InSitu = InSitu[:,1:, :]  # (N, 336, 3) 
        self.data_y_ed_pft = data_y_ed_pft
        print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, data_x_12months={self.data_x_12months.shape}, ESA CCI pft={self.data_y_pft.shape}, AgeWeight={self.AgeWeight.shape}, InSitu={self.InSitu.shape}, data_y_ed_pft={self.data_y_ed_pft.shape}, pft_MIX_VG={self.pft_MIX_VG.shape}')
        

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_12months = self.data_x_12months[index]
        seq_y_pft = self.data_y_pft[index]
        seq_AgeWeight = self.AgeWeight[index]        
        seq_InSitu = self.InSitu[index]
        seq_y_ed_pft = self.data_y_ed_pft[index]
        seq_pft_MIX_VG = self.pft_MIX_VG[index]

        return seq_x, seq_y, seq_x_12months, seq_y_pft, seq_AgeWeight, seq_InSitu, seq_y_ed_pft, seq_pft_MIX_VG
    
    def __len__(self):
        return self.data_x.shape[0]
    
    def __sliding_window__(self, arr, window_size, stride, axis):
        # Ensure the axis is positive and within the valid range
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("Axis out of bounds")

        # Calculate the shape and strides for the sliding window
        new_shape = list(arr.shape)
        new_shape[axis] = (arr.shape[axis] - window_size) // stride + 1
        new_shape.insert(axis + 1, window_size)
        
        new_strides = list(arr.strides)
        new_strides.insert(axis + 1, new_strides[axis])
        new_strides[axis] = new_strides[axis] * stride

        # Create the sliding window view
        return np.lib.stride_tricks.as_strided(arr, shape=tuple(new_shape), strides=tuple(new_strides))


    def __sliding_window_monthly__(self, arr, window_size=2):
        if arr.ndim != 5:
            raise ValueError("Input array must have shape (age, N, years, months, features)")

        age, N, Y, M, F = arr.shape
        if M != 12:
            raise ValueError("Monthly dimension (axis=3) must be 12")
        
        # Step 1: reshape to merge years and months → (age, N, (Y+1)*12, 10)       
        arr_flat = arr.reshape(age, N, Y * M, F)

        # Step 2: sliding window on axis=2 (flattened time), window=24 (2 years)
        new_shape = (age, N, Y - window_size + 1, window_size * M, F)
        new_strides = (
            arr_flat.strides[0],
            arr_flat.strides[1],
            arr_flat.strides[2] * M,      # step forward by 12 months = 1 year
            arr_flat.strides[2],          # step inside 24 months
            arr_flat.strides[3]
        )

        arr_window = np.lib.stride_tricks.as_strided(
            arr_flat,
            shape=new_shape,
            strides=new_strides
        )
        
        return arr_window
    

    def __generate_random_walk_noise__(self, shape, std):
        random_steps = np.random.normal(loc=0.0, scale=std, size=shape)
        random_walk_noise = np.cumsum(random_steps, axis=2)  # Axis 2 corresponds to the "Y" dimension
        return random_walk_noise
    
    # Function to add random walk noise to the specified feature channels
    def __add_random_walk_noise_to_batch__(self, batch, std):
        N, years, channels, features = batch.shape  # (N, 1, 337, 10)  

        # Generate random walk noise for the specified channels
        noise_shape = (N, years, features)
        random_walk_noise = self.__generate_random_walk_noise__(noise_shape, std)

        # Create a zero tensor with the same shape as the batch
        noise_tensor = np.zeros_like(batch)
        
        # Add the random walk noise to the [:, :, :, 0, :] slice
        noise_tensor[:, :, 0, :] = random_walk_noise
        
        # Add the noise tensor to the batch
        batch_with_noise = batch + noise_tensor
        
        return batch_with_noise
    
    
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_AddPure_Extract4types_CombineWith_PFT_Prediction_imputation(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
                 asi=0, aei=18, val_ratio=0.1, add_noise=False, noise_std=1e-4,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        if flag == 'pred':
            flag = 'test'
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.asi = asi
        self.aei = aei
        self.val_ratio = val_ratio
        self.add_noise = add_noise
        self.noise_std = noise_std
         
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        
        self.__read_data__()

    def __read_data__(self):
        stat_raw = np.load(os.path.join(self.root_path, self.stat_path))
        self.x_mean = stat_raw['x_mean']
        self.x_std = stat_raw['x_std']
        self.y_mean = stat_raw['y_mean']
        self.y_std = stat_raw['y_std']
        
        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        # x = (N, Y, 12, 136)
        # y = (age, N, Y+1, 12, 10)
        pft_raw = np.load(os.path.join(self.root_path, self.pft_path))
        # (N, 18, Y+1, 12)

        if self.set_type == 2: 
            data_x = data_raw[f'x_test'] # (N, 28, 12, 136)
            data_y = data_raw[f'y_test'] # (18, N, 29, 12, 10)
            # pft_BL = pft_raw['test_BL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] # (N, 18, Y+1, 12) to (N, age, Y+1, 12, 1) 
            # pft_NL = pft_raw['test_NL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            # pft_GS = pft_raw['test_GS'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            pft_MIX_BL = pft_raw['ESA_BL_test'][..., np.newaxis] # (N, 29) to (N, 29, 1)
            pft_MIX_NL = pft_raw['ESA_NL_test'][..., np.newaxis]
            pft_MIX_GS = pft_raw['ESA_GS_test'][..., np.newaxis]
            pft_MIX_VG = pft_raw['Tree_Cover_test'][..., np.newaxis]
            AgeWeight = pft_raw['AgeWeight_test'] # (18, N)
            InSitu = pft_raw['InSitu_test_imputation'] # (N, 29, 12, 3)
            pft_ED_BL = pft_raw['test_pft_BL'][..., np.newaxis] # (N, 18, 29) to (N, 18, 29, 1)
            pft_ED_NL = pft_raw['test_pft_NL'][..., np.newaxis]
            pft_ED_GS = pft_raw['test_pft_GS'][..., np.newaxis]
            STD = pft_raw['InSitu_test_std']
        else: 
            data_x = data_raw[f'x_train'] # (46, 28, 12, 136)
            data_y = data_raw[f'y_train'] # (18, 46, 29, 12, 10)
            # pft_BL = pft_raw['train_BL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] # (N, age, Y, 12, 1)
            # pft_NL = pft_raw['train_NL'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            # pft_GS = pft_raw['train_GS'][:, self.asi:self.aei, 1:, ...][..., np.newaxis] 
            pft_MIX_BL = pft_raw['ESA_BL_train'][..., np.newaxis]
            pft_MIX_NL = pft_raw['ESA_NL_train'][..., np.newaxis]
            pft_MIX_GS = pft_raw['ESA_GS_train'][..., np.newaxis]
            pft_MIX_VG = pft_raw['Tree_Cover_train'][..., np.newaxis]
            AgeWeight = pft_raw['AgeWeight_train']
            InSitu = pft_raw['InSitu_train_imputation']
            pft_ED_BL = pft_raw['train_pft_BL'][..., np.newaxis] # (N, 18, 29) to (N, 18, 29, 1)
            pft_ED_NL = pft_raw['train_pft_NL'][..., np.newaxis]
            pft_ED_GS = pft_raw['train_pft_GS'][..., np.newaxis]
            STD = pft_raw['InSitu_train_std']


        data_x_12months = data_x.copy() # (N, 28, 12, 136)
        data_x_12months = self.__norm_data__(data_x_12months, self.x_mean, self.x_std)
        data_x_12months = np.transpose(data_x_12months, (0, 1, 3, 2)) # (N, 28, 136, 12)
        data_x_12months_arr = data_x_12months.copy() 
        data_x_12months = data_x_12months.reshape(data_x_12months_arr.shape[0], data_x_12months_arr.shape[1], data_x_12months_arr.shape[2]*data_x_12months_arr.shape[3]) # (N, 28, 136*12)
        
        # dup x for matching ages 
        data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1) # (N, age, 28, 12, 136)
        data_x_arr = data_x.copy() # (N, age, 28, 12, 136)
        data_x = data_x.reshape(data_x_arr.shape[0], data_x_arr.shape[1], data_x_arr.shape[2]*data_x_arr.shape[3], data_x_arr.shape[4]) # (N, age, 336, 136)
        data_x = data_x[:, :, np.newaxis, :, :] # (N, age, 1, 28 *12, 136)
        # normalize data
        data_x = self.__norm_data__(data_x, self.x_mean, self.x_std)  # (N, age, 1, 336, 136)      

        # data_y[data_y < 0] = 0 # (age, N, 29, 12, 10)
        mask = np.ones_like(data_y, dtype=bool) 
        mask[..., 7] = False                    
        data_y[mask & (data_y < 0)] = 0

        data_y = data_y[self.asi:self.aei, ...] # (age, N, 29, 12, 10)       
        # # prepare inital and target pair 
        # data_y = self.__sliding_window_monthly__(data_y, 2) # (age, N, Y, 24, 10)
        data_y_arr = data_y.copy() # (age, N, 29, 12, 10)   
        data_y = data_y.reshape(data_y_arr.shape[0], data_y_arr.shape[1], data_y_arr.shape[2]*data_y_arr.shape[3], data_y_arr.shape[4]) # (age, N, 348, 10)
        data_y = data_y[:, :, 11:, :] # (age, N, 337, 10)
        data_y = data_y[:, :, np.newaxis, :, :] # (age, N, 1, 337, 10)        
        data_y = np.transpose(data_y, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
        # add random-walk noise to target
        if self.add_noise:
            data_y =  self.__add_random_walk_noise_to_batch__(data_y, self.noise_std)
                 
        data_y = self.__norm_data__(data_y, self.y_mean, self.y_std)

        # InSitu_arr = InSitu[...,0].copy() # (N, 29, 12)
        mask = np.ones_like(InSitu, dtype=bool) 
        mask[..., 2] = False                    
        InSitu[mask & (InSitu < 0)] = 0
        InSitu_arr = InSitu

        InSitu = InSitu.reshape(InSitu_arr.shape[0], InSitu_arr.shape[1]*InSitu_arr.shape[2],InSitu_arr.shape[3])  # (46, 29*12) (46, 348)
        # InSitu = InSitu[:, 11:,:] # (N, 337,3)
        InSitu = self.__norm_data__(InSitu, self.y_mean[[4, 9, 7]], self.y_std[[4, 9, 7]])


        STD = STD.reshape(InSitu_arr.shape[0], InSitu_arr.shape[1]*InSitu_arr.shape[2],InSitu_arr.shape[3]) 
        
        pft_MIX_BL = pft_MIX_BL[:,np.newaxis, :, :] # (N, 1, 29, 1)        pft_BL = pft_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
        pft_MIX_NL = pft_MIX_NL[:,np.newaxis, :, :]
        pft_MIX_GS = pft_MIX_GS[:,np.newaxis, :, :]
        # pft_MIX_VG = pft_MIX_VG[:,np.newaxis, :, :]
        # pft_NL = pft_NL.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1) # (N, age, 480, 1)
        # pft_NL = pft_NL[:, :, np.newaxis, :, :] # (N, age, 1, 480, 1)
        # pft_GS = pft_GS.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1) # (N, age, 480, 1)
        # pft_GS = pft_GS[:, :, np.newaxis, :, :] # (N, age, 1, 480, 1)

        data_y_pft = torch.cat([
            torch.from_numpy(pft_MIX_BL),
            torch.from_numpy(pft_MIX_NL),
            torch.from_numpy(pft_MIX_GS)
        ], dim=3) # (N, 1, 29, 3)  

        data_y_ed_pft = torch.cat([
            torch.from_numpy(pft_ED_BL),
            torch.from_numpy(pft_ED_NL),
            torch.from_numpy(pft_ED_GS)
        ], dim=3) # (N, 18, 29, 3)

        AgeWeight = AgeWeight[self.asi:self.aei, ...].swapaxes(0, 1) # (N, 18)

        # split train and val
        if self.set_type != 2:
            idx = np.arange(data_x.shape[0])
            np.random.shuffle(idx)

            val_count = int(len(idx) * self.val_ratio)
            if val_count < 1:
                val_count = 1
            idx_val = idx[:val_count]
            idx_train = idx[val_count:]

            if self.set_type == 0:
                data_x = data_x[idx_train]           
                data_y = data_y[idx_train]
                data_x_12months = data_x_12months[idx_train]
                # pft_MIX_BL = pft_MIX_BL[idx_train] # (N, 1, 29, 1)        pft_BL = pft_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
                # pft_MIX_NL = pft_MIX_NL[idx_train]
                # pft_MIX_GS = pft_MIX_GS[idx_train]                
                data_y_pft = data_y_pft[idx_train]
                pft_MIX_VG = pft_MIX_VG[idx_train]
                AgeWeight = AgeWeight[idx_train]
                InSitu = InSitu[idx_train]
                STD = STD[idx_train]
                data_y_ed_pft = data_y_ed_pft[idx_train]
            else:
                data_x = data_x[idx_val] # (N, age, 1, 336, 136)         
                data_y = data_y[idx_val] # (N, age, 1, 337, 10)  
                data_x_12months = data_x_12months[idx_val] # (N, 28, 136*12)
                # pft_MIX_BL = pft_MIX_BL[idx_val] # (N, 1, 29, 1)        pft_BL = pft_BL[:,np.newaxis, :, :] # (N, 1, 480, 1)
                # pft_MIX_NL = pft_MIX_NL[idx_val]
                # pft_MIX_GS = pft_MIX_GS[idx_val]     
                data_y_pft = data_y_pft[idx_val] # (N, 1, 29, 3)
                pft_MIX_VG = pft_MIX_VG [idx_val]
                AgeWeight = AgeWeight[idx_val] # (N, age)
                InSitu = InSitu[idx_val] # (N, 337) 
                STD = STD[idx_val]
                data_y_ed_pft = data_y_ed_pft[idx_val]  # (N, 18, 29, 3) 

        self.data_x = np.squeeze(data_x, axis=2) # (N, age, 336, 136)   
        self.data_y = np.squeeze(data_y, axis=2) # (N, age, 337, 10)


        self.data_x_12months = data_x_12months # (N, 28, 136*12)   
        self.data_y_pft = np.squeeze(data_y_pft) # (N, 29, 3) 
        self.pft_MIX_VG = pft_MIX_VG # (N, 29, 1) 
        self.AgeWeight = AgeWeight # (N, age)
        self.InSitu = InSitu  # (N, 336, 3) 
        self.STD = STD
        self.data_y_ed_pft = data_y_ed_pft
        print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, data_x_12months={self.data_x_12months.shape}, ESA CCI pft={self.data_y_pft.shape}, AgeWeight={self.AgeWeight.shape}, InSitu={self.InSitu.shape}, data_y_ed_pft={self.data_y_ed_pft.shape}, pft_MIX_VG={self.pft_MIX_VG.shape}, STD={self.STD.shape}')
        

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_12months = self.data_x_12months[index]
        seq_y_pft = self.data_y_pft[index]
        seq_AgeWeight = self.AgeWeight[index]        
        seq_InSitu = self.InSitu[index]
        seq_y_ed_pft = self.data_y_ed_pft[index]
        seq_pft_MIX_VG = self.pft_MIX_VG[index]
        seq_STD = self.STD[index]

        return seq_x, seq_y, seq_x_12months, seq_y_pft, seq_AgeWeight, seq_InSitu, seq_y_ed_pft, seq_pft_MIX_VG, seq_STD
    
    def __len__(self):
        return self.data_x.shape[0]
    
    def __sliding_window__(self, arr, window_size, stride, axis):
        # Ensure the axis is positive and within the valid range
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("Axis out of bounds")

        # Calculate the shape and strides for the sliding window
        new_shape = list(arr.shape)
        new_shape[axis] = (arr.shape[axis] - window_size) // stride + 1
        new_shape.insert(axis + 1, window_size)
        
        new_strides = list(arr.strides)
        new_strides.insert(axis + 1, new_strides[axis])
        new_strides[axis] = new_strides[axis] * stride

        # Create the sliding window view
        return np.lib.stride_tricks.as_strided(arr, shape=tuple(new_shape), strides=tuple(new_strides))


    def __sliding_window_monthly__(self, arr, window_size=2):
        if arr.ndim != 5:
            raise ValueError("Input array must have shape (age, N, years, months, features)")

        age, N, Y, M, F = arr.shape
        if M != 12:
            raise ValueError("Monthly dimension (axis=3) must be 12")
        
        # Step 1: reshape to merge years and months → (age, N, (Y+1)*12, 10)       
        arr_flat = arr.reshape(age, N, Y * M, F)

        # Step 2: sliding window on axis=2 (flattened time), window=24 (2 years)
        new_shape = (age, N, Y - window_size + 1, window_size * M, F)
        new_strides = (
            arr_flat.strides[0],
            arr_flat.strides[1],
            arr_flat.strides[2] * M,      # step forward by 12 months = 1 year
            arr_flat.strides[2],          # step inside 24 months
            arr_flat.strides[3]
        )

        arr_window = np.lib.stride_tricks.as_strided(
            arr_flat,
            shape=new_shape,
            strides=new_strides
        )
        
        return arr_window
    

    def __generate_random_walk_noise__(self, shape, std):
        random_steps = np.random.normal(loc=0.0, scale=std, size=shape)
        random_walk_noise = np.cumsum(random_steps, axis=2)  # Axis 2 corresponds to the "Y" dimension
        return random_walk_noise
    
    # Function to add random walk noise to the specified feature channels
    def __add_random_walk_noise_to_batch__(self, batch, std):
        N, years, channels, features = batch.shape  # (N, 1, 337, 10)  

        # Generate random walk noise for the specified channels
        noise_shape = (N, years, features)
        random_walk_noise = self.__generate_random_walk_noise__(noise_shape, std)

        # Create a zero tensor with the same shape as the batch
        noise_tensor = np.zeros_like(batch)
        
        # Add the random walk noise to the [:, :, :, 0, :] slice
        noise_tensor[:, :, 0, :] = random_walk_noise
        
        # Add the noise tensor to the batch
        batch_with_noise = batch + noise_tensor
        
        return batch_with_noise
    
    
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Baseline(Dataset):
    def __init__(self, root_path, flag='train', stage= 'finetune',size=None, 
                 features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
                 asi=0, aei=18, val_ratio=0.1, add_noise=False, noise_std=1e-4,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        if flag == 'pred':
            flag = 'test'
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.stage = stage
        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.asi = asi
        self.aei = aei
        self.val_ratio = val_ratio
        self.add_noise = add_noise
        self.noise_std = noise_std
         
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        
        self.__read_data__()

    def __read_data__(self):
        stat_raw = np.load(os.path.join(self.root_path, self.stat_path))
        self.x_mean = stat_raw['x_mean']
        self.x_std = stat_raw['x_std']
        self.y_mean = stat_raw['y_mean']
        self.y_std = stat_raw['y_std']
        
        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        # x = (N, Y, 12, 136)
        # y = (age, N, Y+1, 12, 10)
        pft_raw = np.load(os.path.join(self.root_path, self.pft_path))
        # (N, 18, Y+1, 12)

        if self.set_type == 2: 
            data_x = data_raw[f'x_test'] # (N, 28, 12, 136)
            data_y = data_raw[f'y_test'] # (18, N, 29, 12, 10)
            data_y_agesum = pft_raw[f'InSitu_test'] # (N, 29, 12, 3)
            AgeWeight = pft_raw['AgeWeight_test'].swapaxes(0, 1) # (N, 18)


        else: 
            data_x = data_raw[f'x_train'] # (N, 28, 12, 136)
            data_y = data_raw[f'y_train'] # (18, N, 29, 12, 10)
            data_y_agesum = pft_raw[f'InSitu_train'] # (N, 29, 12, 3)
            AgeWeight = pft_raw['AgeWeight_train'].swapaxes(0, 1) # (N, 18)

        mask = np.ones_like(data_y_agesum, dtype=bool) 
        mask[..., 2] = False                    
        data_y_agesum[mask & (data_y_agesum < 0)] = 0
            
        data_y_agesum_arr = data_y_agesum.copy() # (N, 29, 12, 10)
        data_y_agesum = data_y_agesum.reshape(data_y_agesum_arr.shape[0], data_y_agesum_arr.shape[1]*data_y_agesum_arr.shape[2], data_y_agesum_arr.shape[3]) # (N, 29*12, 10)
        data_y_agesum = data_y_agesum[:, 11:, :] # (N, 337, 10)
        data_y_agesum = data_y_agesum[:, np.newaxis, :, :] # (N, 1, 337, 10)        
        # data_y_agesum = np.transpose(data_y_agesum, (1,0,2,3,4))
        # add random-walk noise to target
        if self.add_noise:
            data_y_agesum =  self.__add_random_walk_noise_to_batch__(data_y_agesum, self.noise_std)
            
        data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean[[4, 9, 7]], self.y_std[[4, 9, 7]])

        mask = np.ones_like(data_y, dtype=bool) 
        mask[..., 7] = False                    
        data_y[mask & (data_y < 0)] = 0
        
        data_y = data_y[self.asi:self.aei, ...] # (age, N, 29, 12, 10)        
        # # prepare inital and target pair 
        # data_y = self.__sliding_window_monthly__(data_y, 2) # (age, N, Y, 24, 10)
        data_y_arr = data_y.copy() # (age, N, 29, 12, 10) 
        data_y = data_y.reshape(data_y_arr.shape[0], data_y_arr.shape[1], data_y_arr.shape[2]*data_y_arr.shape[3], data_y_arr.shape[4]) # (age, N, 29*12, 10)
        data_y = data_y[:, :, 11:, :] # (age, N, 337, 10)
        data_y = data_y[:, :, np.newaxis, :, :] # (age, N, 1, 337, 10)   
        data_y = np.transpose(data_y, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
        # add random-walk noise to target
        if self.add_noise:
            data_y =  self.__add_random_walk_noise_to_batch__(data_y, self.noise_std)


        # dup x for matching ages 
        data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1) # (N, age, 28, 12, 136)
        data_x_arr = data_x.copy()
        data_x = data_x.reshape(data_x_arr.shape[0], data_x_arr.shape[1], data_x_arr.shape[2]*data_x_arr.shape[3], data_x_arr.shape[4]) # (N, age, 28*12, 136)
        data_x = data_x[:, :, np.newaxis, :, :] # (N, age, 1, 28*12, 136)            
        # normalize data
        data_x = self.__norm_data__(data_x, self.x_mean, self.x_std)
        data_y = self.__norm_data__(data_y, self.y_mean, self.y_std)


        # split train and val
        if self.set_type != 2:
            idx = np.arange(data_x.shape[0])
            np.random.shuffle(idx)
            val_count = int(len(idx) * self.val_ratio)
            if val_count < 1:
                val_count = 1
            idx_val = idx[:val_count]
            idx_train = idx[val_count:]

            if self.set_type == 0:
                data_x = data_x[idx_train] # (N, age, 1, 336, 136)        
                data_y = data_y[idx_train] # (N, age, 1, 337, 10)   
                data_y_agesum = data_y_agesum[idx_train] # (N, 1, 337, 10) or (N, 1, 337)


                AgeWeight = AgeWeight[idx_train] # (N, 18)
                # treecover = treecover[idx_train]
            else:
                data_x = data_x[idx_val]            
                data_y = data_y[idx_val]
                data_y_agesum = data_y_agesum[idx_val]

                AgeWeight = AgeWeight[idx_val]

        self.data_x = np.squeeze(data_x, axis=2) # (N, age, 1, 336, 136) to (N, age, 336, 136)
        self.data_y = np.squeeze(data_y, axis=2) # (N, age, 1, 337, 10) to (N, age, 337, 10)
        # self.pft_MIX_BL = pft_MIX_BL.squeeze(2) # (N, age, 1, 336, 1) to (N, age, 336, 1)
        # self.pft_MIX_NL = pft_MIX_NL.squeeze(2)
        # self.pft_MIX_GS = pft_MIX_GS.squeeze(2)
        self.data_x = self.data_x[:,0,:,:]
        # if self.stage == 'pretrain':    
        self.data_y_agesum = data_y_agesum.reshape(-1, data_y_agesum.shape[2], data_y_agesum.shape[3]) # (N, 1, 337, 10) to (-1, 337, 10)

        self.AgeWeight = AgeWeight # (N, 18)
        # self.treecover = treecover


        self.data_y = self.data_y * self.AgeWeight[:, :, np.newaxis, np.newaxis]   # (44,18,337,10)
        self.data_y = self.data_y.sum(axis=1)

        print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, data_y_agesum={self.data_y_agesum.shape},  AgeWeight={self.AgeWeight.shape}')
        

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_y_agesum = self.data_y_agesum[index]
        seq_aw = self.AgeWeight[index]
        # seq_tc = self.treecover[index]

        return seq_x, seq_y, seq_y_agesum, seq_aw
    
    def __len__(self):
        return self.data_x.shape[0]
    
    def __sliding_window__(self, arr, window_size, stride, axis):
        # Ensure the axis is positive and within the valid range
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("Axis out of bounds")

        # Calculate the shape and strides for the sliding window
        new_shape = list(arr.shape)
        new_shape[axis] = (arr.shape[axis] - window_size) // stride + 1
        new_shape.insert(axis + 1, window_size)
        
        new_strides = list(arr.strides)
        new_strides.insert(axis + 1, new_strides[axis])
        new_strides[axis] = new_strides[axis] * stride

        # Create the sliding window view
        return np.lib.stride_tricks.as_strided(arr, shape=tuple(new_shape), strides=tuple(new_strides))


    def __sliding_window_monthly__(self, arr, window_size=2):
        if arr.ndim != 5:
            raise ValueError("Input array must have shape (age, N, years, months, features)")

        age, N, Y, M, F = arr.shape
        if M != 12:
            raise ValueError("Monthly dimension (axis=3) must be 12")
        
        # Step 1: reshape to merge years and months → (age, N, (Y+1)*12, 10)       
        arr_flat = arr.reshape(age, N, Y * M, F)

        # Step 2: sliding window on axis=2 (flattened time), window=24 (2 years)
        new_shape = (age, N, Y - window_size + 1, window_size * M, F)
        new_strides = (
            arr_flat.strides[0],
            arr_flat.strides[1],
            arr_flat.strides[2] * M,      # step forward by 12 months = 1 year
            arr_flat.strides[2],          # step inside 24 months
            arr_flat.strides[3]
        )

        arr_window = np.lib.stride_tricks.as_strided(
            arr_flat,
            shape=new_shape,
            strides=new_strides
        )
        
        return arr_window
    

    def __generate_random_walk_noise__(self, shape, std):
        random_steps = np.random.normal(loc=0.0, scale=std, size=shape)
        random_walk_noise = np.cumsum(random_steps, axis=2)  # Axis 2 corresponds to the "Y" dimension
        return random_walk_noise
    
    # Function to add random walk noise to the specified feature channels
    def __add_random_walk_noise_to_batch__(self, batch, std):
        N, years, channels, features = batch.shape  # (N, 1, 337, 10)  

        # Generate random walk noise for the specified channels
        noise_shape = (N, years, features)
        random_walk_noise = self.__generate_random_walk_noise__(noise_shape, std)

        # Create a zero tensor with the same shape as the batch
        noise_tensor = np.zeros_like(batch)
        
        # Add the random walk noise to the [:, :, :, 0, :] slice
        noise_tensor[:, :, 0, :] = random_walk_noise
        
        # Add the noise tensor to the batch
        batch_with_noise = batch + noise_tensor
        
        return batch_with_noise
    
    
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_KGML(Dataset):
    def __init__(self, root_path, flag='train', stage= 'finetune',size=None, 
                 features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
                 asi=0, aei=15, val_ratio=0.1, add_noise=False, noise_std=1e-4,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        if flag == 'pred':
            flag = 'test'
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.stage = stage
        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.asi = asi
        self.aei = aei
        self.val_ratio = val_ratio
        self.add_noise = add_noise
        self.noise_std = noise_std
         
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        
        self.__read_data__()

    def __read_data__(self):
        stat_raw = np.load(os.path.join(self.root_path, self.stat_path))
        self.x_mean = stat_raw['x_mean']
        self.x_std = stat_raw['x_std']
        self.y_mean = stat_raw['y_mean']
        self.y_std = stat_raw['y_std']
        
        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        # x = (N, Y, 12, 136)
        # y = (age, N, Y+1, 12, 10)
        pft_raw = np.load(os.path.join(self.root_path, self.pft_path))
        # (N, age, Y+1, 12)

        if self.set_type == 2: 
            data_x = data_raw[f'x_test'] # (N, 28, 12, 136)
            data_y = data_raw[f'y_test'] # (age, N, 29, 12, 10)
            if self.stage == 'pretrain':    
                data_y_agesum = data_raw[f'y_test_agesum'] # (N, 29, 12, 10)
                N = data_y_agesum.shape[0]
                # treecover = np.full((N, 29), np.nan)  
            elif self.stage == 'finetune':  
                data_y_agesum = pft_raw[f'InSitu_test'] # (N, 29, 12, 10)

            AgeWeight = pft_raw['AgeWeight_test'].swapaxes(0, 1) # (N, age)


        else: 
            data_x = data_raw[f'x_train'] # (N, 28, 12, 136)
            data_y = data_raw[f'y_train'] # (age, N, 29, 12, 10)
            # data_y_agesum = data_raw[f'y_train_agesum'] # (N, 29, 12, 10)
            if self.stage == 'pretrain':    
                data_y_agesum = data_raw[f'y_train_agesum'] # (N, 29, 12, 10)
                N = data_y_agesum.shape[0]
                # treecover = np.full((N, 29), np.nan)  
            elif self.stage == 'finetune':  
                data_y_agesum = pft_raw[f'InSitu_train']

            AgeWeight = pft_raw['AgeWeight_train'].swapaxes(0, 1) # (N, age)

        if self.stage == 'pretrain':    
            mask = np.ones_like(data_y_agesum, dtype=bool) 
            mask[..., 7] = False                    
            data_y_agesum[mask & (data_y_agesum < 0)] = 0
            # treecover = np.full((N, 29), np.nan)  
        elif self.stage == 'finetune':  
            mask = np.ones_like(data_y_agesum, dtype=bool) 
            mask[..., 2] = False                    
            data_y_agesum[mask & (data_y_agesum < 0)] = 0    

        data_y_agesum_arr = data_y_agesum.copy() # (N, 29, 12, 10)
        data_y_agesum = data_y_agesum.reshape(data_y_agesum_arr.shape[0], data_y_agesum_arr.shape[1]*data_y_agesum_arr.shape[2], data_y_agesum_arr.shape[3]) # (N, 29*12, 7)
        data_y_agesum = data_y_agesum[:, 11:, :] # (N, 337, 10)
        data_y_agesum = data_y_agesum[:, np.newaxis, :, :] # (N, 1, 337, 10)        
        # data_y_agesum = np.transpose(data_y_agesum, (1,0,2,3,4))
        # add random-walk noise to target
        if self.add_noise:
            data_y_agesum =  self.__add_random_walk_noise_to_batch__(data_y_agesum, self.noise_std)
            
        if self.stage == 'pretrain':    

            data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean, self.y_std)
        elif self.stage == 'finetune':  
            data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean[[4, 9, 7]], self.y_std[[4, 9, 7]])
            

        # data_y[data_y < 0] = 0
        mask = np.ones_like(data_y, dtype=bool) 
        mask[..., 7] = False                    
        data_y[mask & (data_y < 0)] = 0
        
        # keep 12 month as target (shuo)
        data_y = data_y[self.asi:self.aei, ...] # (age, N, 29, 12, 10)        

        data_y_arr = data_y.copy() # (age, N, 29, 12, 10) 
        data_y = data_y.reshape(data_y_arr.shape[0], data_y_arr.shape[1], data_y_arr.shape[2]*data_y_arr.shape[3], data_y_arr.shape[4]) # (age, N, 29*12, 10)
        data_y = data_y[:, :, 11:, :] # (age, N, 337, 10)
        data_y = data_y[:, :, np.newaxis, :, :] # (age, N, 1, 337, 10)   
        data_y = np.transpose(data_y, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
        # add random-walk noise to target
        if self.add_noise:
            data_y =  self.__add_random_walk_noise_to_batch__(data_y, self.noise_std)


        # dup x for matching ages 
        data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1) # (N, age, 28, 12, 136)
        data_x_arr = data_x.copy()
        data_x = data_x.reshape(data_x_arr.shape[0], data_x_arr.shape[1], data_x_arr.shape[2]*data_x_arr.shape[3], data_x_arr.shape[4]) # (N, age, 28*12, 136)
        data_x = data_x[:, :, np.newaxis, :, :] # (N, age, 1, 28*12, 136)            
        # normalize data
        data_x = self.__norm_data__(data_x, self.x_mean, self.x_std)
        data_y = self.__norm_data__(data_y, self.y_mean, self.y_std)



        # split train and val
        if self.set_type != 2:
            idx = np.arange(data_x.shape[0])
            np.random.shuffle(idx)
            val_count = int(len(idx) * self.val_ratio)
            if val_count < 1:
                val_count = 1
            idx_val = idx[:val_count]
            idx_train = idx[val_count:]

            if self.set_type == 0:
                data_x = data_x[idx_train] # (N, age, 1, 336, 136)        
                data_y = data_y[idx_train] # (N, age, 1, 337, 10)   
                data_y_agesum = data_y_agesum[idx_train] # (N, 1, 337, 10) or (N, 1, 337)


                AgeWeight = AgeWeight[idx_train] # (N, age)
                # treecover = treecover[idx_train]
            else:
                data_x = data_x[idx_val]            
                data_y = data_y[idx_val]
                data_y_agesum = data_y_agesum[idx_val]


                AgeWeight = AgeWeight[idx_val]
                # treecover = treecover[idx_val]
                # data_tree_cover = 

        self.data_x = np.squeeze(data_x, axis=2) # (N, age, 1, 336, 136) to (N, age, 336, 136)
        self.data_y = np.squeeze(data_y, axis=2) # (N, age, 1, 337, 10) to (N, age, 337, 10)

       
        # if self.stage == 'pretrain':    
        self.data_y_agesum = data_y_agesum.reshape(-1, data_y_agesum.shape[2], data_y_agesum.shape[3]) # (N, 1, 337, 10) to (-1, 337, 10)

        
        self.AgeWeight = AgeWeight # (N, age)
        # self.treecover = treecover

        print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, data_y_agesum={self.data_y_agesum.shape},  AgeWeight={self.AgeWeight.shape}')
        

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_y_agesum = self.data_y_agesum[index]

        seq_aw = self.AgeWeight[index]
        # seq_tc = self.treecover[index]

        return seq_x, seq_y, seq_y_agesum, seq_aw
    
    def __len__(self):
        return self.data_x.shape[0]
    
    def __sliding_window__(self, arr, window_size, stride, axis):
        # Ensure the axis is positive and within the valid range
        axis = axis if axis >= 0 else arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("Axis out of bounds")

        # Calculate the shape and strides for the sliding window
        new_shape = list(arr.shape)
        new_shape[axis] = (arr.shape[axis] - window_size) // stride + 1
        new_shape.insert(axis + 1, window_size)
        
        new_strides = list(arr.strides)
        new_strides.insert(axis + 1, new_strides[axis])
        new_strides[axis] = new_strides[axis] * stride

        # Create the sliding window view
        return np.lib.stride_tricks.as_strided(arr, shape=tuple(new_shape), strides=tuple(new_strides))


    def __sliding_window_monthly__(self, arr, window_size=2):
        if arr.ndim != 5:
            raise ValueError("Input array must have shape (age, N, years, months, features)")

        age, N, Y, M, F = arr.shape
        if M != 12:
            raise ValueError("Monthly dimension (axis=3) must be 12")
        
        # Step 1: reshape to merge years and months → (age, N, (Y+1)*12, 10)       
        arr_flat = arr.reshape(age, N, Y * M, F)

        # Step 2: sliding window on axis=2 (flattened time), window=24 (2 years)
        new_shape = (age, N, Y - window_size + 1, window_size * M, F)
        new_strides = (
            arr_flat.strides[0],
            arr_flat.strides[1],
            arr_flat.strides[2] * M,      # step forward by 12 months = 1 year
            arr_flat.strides[2],          # step inside 24 months
            arr_flat.strides[3]
        )

        arr_window = np.lib.stride_tricks.as_strided(
            arr_flat,
            shape=new_shape,
            strides=new_strides
        )
        
        return arr_window
    

    def __generate_random_walk_noise__(self, shape, std):
        random_steps = np.random.normal(loc=0.0, scale=std, size=shape)
        random_walk_noise = np.cumsum(random_steps, axis=2)  # Axis 2 corresponds to the "Y" dimension
        return random_walk_noise
    
    # Function to add random walk noise to the specified feature channels
    def __add_random_walk_noise_to_batch__(self, batch, std):
        N, years, channels, features = batch.shape  # (N, 1, 337, 10)  

        # Generate random walk noise for the specified channels
        noise_shape = (N, years, features)
        random_walk_noise = self.__generate_random_walk_noise__(noise_shape, std)

        # Create a zero tensor with the same shape as the batch
        noise_tensor = np.zeros_like(batch)
        
        # Add the random walk noise to the [:, :, :, 0, :] slice
        noise_tensor[:, :, 0, :] = random_walk_noise
        
        # Add the noise tensor to the batch
        batch_with_noise = batch + noise_tensor
        
        return batch_with_noise
    
    
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




# class Dataset_KGML(Dataset):
#     def __init__(self, root_path, flag='train', stage= 'finetune',size=None, 
#                  features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
#                  asi=0, aei=18, val_ratio=0.1, add_noise=False, noise_std=1e-4,
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24*4*4
#             self.label_len = 24*4
#             self.pred_len = 24*4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val', 'pred']
#         if flag == 'pred':
#             flag = 'test'
#         self.flag = flag
#         type_map = {'train':0, 'val':1, 'test':2}
#         self.set_type = type_map[flag]
#         self.stage = stage
#         self.root_path = root_path
#         self.data_path = data_path
#         self.stat_path = stat_path
#         self.pft_path = pft_path
#         self.asi = asi
#         self.aei = aei
#         self.val_ratio = val_ratio
#         self.add_noise = add_noise
#         self.noise_std = noise_std
         
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols=cols
        
#         self.__read_data__()

#     def __read_data__(self):
#         stat_raw = np.load(os.path.join(self.root_path, self.stat_path))
#         self.x_mean = stat_raw['x_mean']
#         self.x_std = stat_raw['x_std']
#         self.y_mean = stat_raw['y_mean']
#         self.y_std = stat_raw['y_std']
        
#         data_raw = np.load(os.path.join(self.root_path, self.data_path))
#         # x = (N, Y, 12, 136)
#         # y = (age, N, Y+1, 12, 10)
#         pft_raw = np.load(os.path.join(self.root_path, self.pft_path))
#         # (N, 18, Y+1, 12)

#         if self.set_type == 2: 
#             data_x = data_raw[f'x_test'] # (N, 28, 12, 136)
#             data_y = data_raw[f'y_test'] # (18, N, 29, 12, 10)
#             if self.stage == 'pretrain':    
#                 data_y_agesum = data_raw[f'y_test_agesum'] # (N, 29, 12, 10)
#             elif self.stage == 'finetune':  
#                 data_y_agesum = pft_raw[f'InSitu_test'] # (N, 29, 12, 10)

#             AgeWeight = pft_raw['AgeWeight_test'].swapaxes(0, 1) # (N, 18)


#         else: 
#             data_x = data_raw[f'x_train'] # (N, 28, 12, 136)
#             data_y = data_raw[f'y_train'] # (18, N, 29, 12, 10)
#             # data_y_agesum = data_raw[f'y_train_agesum'] # (N, 29, 12, 10)
#             if self.stage == 'pretrain':    
#                 data_y_agesum = data_raw[f'y_train_agesum'] # (N, 29, 12, 10)
#             elif self.stage == 'finetune':  
#                 data_y_agesum = pft_raw[f'InSitu_train'] # (N, 29, 12)

#             AgeWeight = pft_raw['AgeWeight_train'].swapaxes(0, 1) # (N, 18)

#         if self.stage == 'pretrain':    
#             data_y_agesum[data_y_agesum < 0] = 0 # (N, 29, 12, 10)
#             data_y_agesum_arr = data_y_agesum.copy() # (N, 29, 12, 10)
#             data_y_agesum = data_y_agesum.reshape(data_y_agesum_arr.shape[0], data_y_agesum_arr.shape[1]*data_y_agesum_arr.shape[2], data_y_agesum_arr.shape[3]) # (N, 29*12, 10)
#             data_y_agesum = data_y_agesum[:, 11:, :] # (N, 337, 10)
#             data_y_agesum = data_y_agesum[:, np.newaxis, :, :] # (N, 1, 337, 10)        
#             # data_y_agesum = np.transpose(data_y_agesum, (1,0,2,3,4))
#             # add random-walk noise to target
#             if self.add_noise:
#                 data_y_agesum =  self.__add_random_walk_noise_to_batch__(data_y_agesum, self.noise_std)
#             data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean, self.y_std)
#         elif self.stage == 'finetune':  
#             data_y_agesum[data_y_agesum < 0] = 0 # (N, 29, 12)
#             data_y_agesum_arr = data_y_agesum.copy() # (N, 29, 12)
#             data_y_agesum = data_y_agesum.reshape(data_y_agesum_arr.shape[0], data_y_agesum_arr.shape[1]*data_y_agesum_arr.shape[2]) # (N, 29*12)
#             data_y_agesum = data_y_agesum[:, 11:] # (N, 337)
#             data_y_agesum = data_y_agesum[:, np.newaxis, :] # (N, 1, 337)        
#             # data_y_agesum = np.transpose(data_y_agesum, (1,0,2,3,4))
#             # add random-walk noise to target
#             if self.add_noise:
#                 data_y_agesum =  self.__add_random_walk_noise_to_batch__(data_y_agesum, self.noise_std)
#             data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean[4], self.y_std[4])

#         data_y[data_y < 0] = 0
#         data_y = data_y[self.asi:self.aei, ...] # (age, N, 29, 12, 10)        
#         # # prepare inital and target pair 
#         # data_y = self.__sliding_window_monthly__(data_y, 2) # (age, N, Y, 24, 10)
#         data_y_arr = data_y.copy() # (age, N, 29, 12, 10) 
#         data_y = data_y.reshape(data_y_arr.shape[0], data_y_arr.shape[1], data_y_arr.shape[2]*data_y_arr.shape[3], data_y_arr.shape[4]) # (age, N, 29*12, 10)
#         data_y = data_y[:, :, 11:, :] # (age, N, 337, 10)
#         data_y = data_y[:, :, np.newaxis, :, :] # (age, N, 1, 337, 10)   
#         data_y = np.transpose(data_y, (1,0,2,3,4)) # (N, age, 1, 337, 10)  
#         # add random-walk noise to target
#         if self.add_noise:
#             data_y =  self.__add_random_walk_noise_to_batch__(data_y, self.noise_std)


#         # dup x for matching ages 
#         data_x = np.repeat(data_x[:, np.newaxis, :, :, :], self.aei-self.asi, axis=1) # (N, age, 28, 12, 136)
#         data_x_arr = data_x.copy()
#         data_x = data_x.reshape(data_x_arr.shape[0], data_x_arr.shape[1], data_x_arr.shape[2]*data_x_arr.shape[3], data_x_arr.shape[4]) # (N, age, 28*12, 136)
#         data_x = data_x[:, :, np.newaxis, :, :] # (N, age, 1, 28*12, 136)            
#         # normalize data
#         data_x = self.__norm_data__(data_x, self.x_mean, self.x_std)
#         data_y = self.__norm_data__(data_y, self.y_mean, self.y_std)

#         # split train and val
#         if self.set_type != 2:
#             idx = np.arange(data_x.shape[0])
#             np.random.shuffle(idx)
#             val_count = int(len(idx) * self.val_ratio)
#             if val_count < 1:
#                 val_count = 1
#             idx_val = idx[:val_count]
#             idx_train = idx[val_count:]

#             if self.set_type == 0:
#                 data_x = data_x[idx_train] # (N, age, 1, 336, 136)        
#                 data_y = data_y[idx_train] # (N, age, 1, 337, 10)   
#                 data_y_agesum = data_y_agesum[idx_train] # (N, 1, 337, 10) or (N, 1, 337)

#                 AgeWeight = AgeWeight[idx_train] # (N, 18)
#             else:
#                 data_x = data_x[idx_val]            
#                 data_y = data_y[idx_val]
#                 data_y_agesum = data_y_agesum[idx_val]
#                 AgeWeight = AgeWeight[idx_val]

#         self.data_x = np.squeeze(data_x, axis=2) # (N, age, 1, 336, 136) to (N, age, 336, 136)
#         self.data_y = np.squeeze(data_y, axis=2) # (N, age, 1, 337, 10) to (N, age, 337, 10)

       
#         if self.stage == 'pretrain':    
#             self.data_y_agesum = data_y_agesum.reshape(-1, data_y_agesum.shape[2], data_y_agesum.shape[3]) # (N, 1, 337, 10) to (-1, 337, 10)
#         elif self.stage == 'finetune':  
#             self.data_y_agesum = data_y_agesum.reshape(-1, data_y_agesum.shape[2]) # (N, 1, 337) to (-1, 337)
        
#         self.AgeWeight = AgeWeight # (N, 18)

#         print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, data_y_agesum={self.data_y_agesum.shape},  AgeWeight={self.AgeWeight.shape}')
        

#     def __getitem__(self, index):
#         seq_x = self.data_x[index]
#         seq_y = self.data_y[index]
#         seq_y_agesum = self.data_y_agesum[index]
#         seq_aw = self.AgeWeight[index]

#         return seq_x, seq_y, seq_y_agesum, seq_aw
    
#     def __len__(self):
#         return self.data_x.shape[0]
    
#     def __sliding_window__(self, arr, window_size, stride, axis):
#         # Ensure the axis is positive and within the valid range
#         axis = axis if axis >= 0 else arr.ndim + axis
#         if axis < 0 or axis >= arr.ndim:
#             raise ValueError("Axis out of bounds")

#         # Calculate the shape and strides for the sliding window
#         new_shape = list(arr.shape)
#         new_shape[axis] = (arr.shape[axis] - window_size) // stride + 1
#         new_shape.insert(axis + 1, window_size)
        
#         new_strides = list(arr.strides)
#         new_strides.insert(axis + 1, new_strides[axis])
#         new_strides[axis] = new_strides[axis] * stride

#         # Create the sliding window view
#         return np.lib.stride_tricks.as_strided(arr, shape=tuple(new_shape), strides=tuple(new_strides))


#     def __sliding_window_monthly__(self, arr, window_size=2):
#         if arr.ndim != 5:
#             raise ValueError("Input array must have shape (age, N, years, months, features)")

#         age, N, Y, M, F = arr.shape
#         if M != 12:
#             raise ValueError("Monthly dimension (axis=3) must be 12")
        
#         # Step 1: reshape to merge years and months     
#         arr_flat = arr.reshape(age, N, Y * M, F)

#         # Step 2: sliding window on axis=2 (flattened time), window=24 (2 years)
#         new_shape = (age, N, Y - window_size + 1, window_size * M, F)
#         new_strides = (
#             arr_flat.strides[0],
#             arr_flat.strides[1],
#             arr_flat.strides[2] * M,      # step forward by 12 months = 1 year
#             arr_flat.strides[2],          # step inside 24 months
#             arr_flat.strides[3]
#         )

#         arr_window = np.lib.stride_tricks.as_strided(
#             arr_flat,
#             shape=new_shape,
#             strides=new_strides
#         )
        
#         return arr_window
    

#     def __generate_random_walk_noise__(self, shape, std):
#         random_steps = np.random.normal(loc=0.0, scale=std, size=shape)
#         random_walk_noise = np.cumsum(random_steps, axis=2)  # Axis 2 corresponds to the "Y" dimension
#         return random_walk_noise
    
#     # Function to add random walk noise to the specified feature channels
#     def __add_random_walk_noise_to_batch__(self, batch, std):
#         N, years, channels, features = batch.shape  # (N, 1, 337, 10)  

#         # Generate random walk noise for the specified channels
#         noise_shape = (N, years, features)
#         random_walk_noise = self.__generate_random_walk_noise__(noise_shape, std)

#         # Create a zero tensor with the same shape as the batch
#         noise_tensor = np.zeros_like(batch)
        
#         # Add the random walk noise to the [:, :, :, 0, :] slice
#         noise_tensor[:, :, 0, :] = random_walk_noise
        
#         # Add the noise tensor to the batch
#         batch_with_noise = batch + noise_tensor
        
#         return batch_with_noise
    
    
#     def __norm_data__(self, data, mean, std):
#         return (data-mean)/(std+1e-10)
    
#     def __inverse_norm_data__(self, data, mean, std):
#         return data*(std+1e-10) + mean

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
    
  
