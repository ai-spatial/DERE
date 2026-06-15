import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_ED2(Dataset):
    def __init__(self, root_path, flag='train', stage= 'pretrain', size=None,
                 features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
                 val_ratio=0.1, add_noise=False, noise_std=1e-4,
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.asi = 0
        self.aei = 18
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.val_ratio = val_ratio
        self.add_noise = add_noise
        self.noise_std = noise_std

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.stage = stage

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag

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
                data_y_agesum = pft_raw[f'InSitu_test'] # (N, 29, 12, 3)

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

        data_y_agesum_arr = data_y_agesum.copy() # (N, 29, 12, *)
        data_y_agesum = data_y_agesum.reshape(data_y_agesum_arr.shape[0], data_y_agesum_arr.shape[1]*data_y_agesum_arr.shape[2], data_y_agesum_arr.shape[3]) # (N, 29*12, *)
        data_y_agesum = data_y_agesum[:, 11:, :] # (N, 337, *)
        data_y_agesum = data_y_agesum[:, np.newaxis, :, :] # (N, 1, 337, *)        
        # data_y_agesum = np.transpose(data_y_agesum, (1,0,2,3,4))
        # add random-walk noise to target
        if self.add_noise:
            data_y_agesum =  self.__add_random_walk_noise_to_batch__(data_y_agesum, self.noise_std)
            
        if self.stage == 'pretrain':    

            data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean, self.y_std)
        elif self.stage == 'finetune':  
            data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean[[4, 9, 7]], self.y_std[[4, 9, 7]])
            
        mask = np.ones_like(data_y, dtype=bool) 
        mask[..., 7] = False                    
        data_y[mask & (data_y < 0)] = 0
        
        # keep 12 month as target (shuo)
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
                data_y_agesum = data_y_agesum[idx_train] # (N, 1, 337, *) or (N, 1, 337)

                # pft_MIX_BL = pft_MIX_BL[idx_train] # (N, age, 1, 336, 1)
                # pft_MIX_NL = pft_MIX_NL[idx_train]
                # pft_MIX_GS = pft_MIX_GS[idx_train]
                # pft_MIX_BL_agesum = pft_MIX_BL_agesum[idx_train] # (N, 1, 336, 1)
                # pft_MIX_NL_agesum = pft_MIX_NL_agesum[idx_train]
                # pft_MIX_GS_agesum = pft_MIX_GS_agesum[idx_train]

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
        
        self.data_y_agesum = data_y_agesum.reshape(-1, data_y_agesum.shape[2], data_y_agesum.shape[3]) # (N, 1, 337, *) to (-1, 337, *)
        
        self.AgeWeight = AgeWeight # (N, age)
        # self.treecover = treecover

        print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, data_y_agesum={self.data_y_agesum.shape},  AgeWeight={self.AgeWeight.shape}')

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_y_agesum = self.data_y_agesum[index]

        # seq_pft_BL = self.pft_MIX_BL[index]
        # seq_pft_NL = self.pft_MIX_NL[index]
        # seq_pft_GS = self.pft_MIX_GS[index]
        
        # seq_pft_BL_agesum = self.pft_MIX_BL_agesum[index]
        # seq_pft_NL_agesum = self.pft_MIX_NL_agesum[index]
        # seq_pft_GS_agesum = self.pft_MIX_GS_agesum[index]
        seq_aw = self.AgeWeight[index]
        # seq_tc = self.treecover[index]

        return seq_x, seq_y, seq_y_agesum, seq_aw

    def __len__(self):
        return self.data_x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
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
        
        # Step 1: reshape to merge years and months → (age, N, (Y+1)*12, 7)       
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
        N, years, channels, features = batch.shape  # (N, 1, 481, 7)  

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
    

class Dataset_ED(Dataset):
    def __init__(self, root_path, flag='train', stage= 'pretrain', size=None,
                 features='S', data_path='data_train.npz', stat_path='stat.npz', pft_path = 'pft.npz',
                 val_ratio=0.1, add_noise=False, noise_std=1e-4,
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.asi = 0
        self.aei = 18
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.val_ratio = val_ratio
        self.add_noise = add_noise
        self.noise_std = noise_std

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.stage = stage
        
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag

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
            data_y_agesum = pft_raw[f'InSitu_test'] # (N, 29, 12, 3)
            AgeWeight = pft_raw['AgeWeight_test'].swapaxes(0, 1) # (N, age)

        else: 
            data_x = data_raw[f'x_train'] # (N, 28, 12, 136)
            data_y = data_raw[f'y_train'] # (age, N, 29, 12, 10)
            data_y_agesum = pft_raw[f'InSitu_train'] # (N, 29, 12, 3)
            AgeWeight = pft_raw['AgeWeight_train'].swapaxes(0, 1) # (N, age)

        mask = np.ones_like(data_y_agesum, dtype=bool) 
        mask[..., 2] = False                    
        data_y_agesum[mask & (data_y_agesum < 0)] = 0

        data_y_agesum_arr = data_y_agesum.copy() # (N, 29, 12, 3)
        data_y_agesum = data_y_agesum.reshape(data_y_agesum_arr.shape[0], data_y_agesum_arr.shape[1]*data_y_agesum_arr.shape[2], data_y_agesum_arr.shape[3]) # (N, 29*12, 3)
        data_y_agesum = data_y_agesum[:, 11:, :] # (N, 337, 3)
        data_y_agesum = data_y_agesum[:, np.newaxis, :, :] # (N, 1, 337, 3)        
        # data_y_agesum = np.transpose(data_y_agesum, (1,0,2,3,4))
        # add random-walk noise to target
        if self.add_noise:
            data_y_agesum =  self.__add_random_walk_noise_to_batch__(data_y_agesum, self.noise_std)
            
        # if self.stage == 'pretrain':    

        #     data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean, self.y_std)
        # elif self.stage == 'finetune':  
        data_y_agesum = self.__norm_data__(data_y_agesum, self.y_mean[[4, 9, 7]], self.y_std[[4, 9, 7]])
            # # data_y_agesum[data_y_agesum < 0] = 0 # (N, 29, 12)
            # mask = np.ones_like(data_y_agesum, dtype=bool) 
            # mask[..., 7] = False                    
            # data_y_agesum[mask & (data_y_agesum < 0)] = 0

            # data_y_agesum_arr = data_y_agesum.copy() # (N, 29, 12)
            # data_y_agesum = data_y_agesum.reshape(data_y_agesum_arr.shape[0], data_y_agesum_arr.shape[1]*data_y_agesum_arr.shape[2]) # (N, 29*12)
            # data_y_agesum = data_y_agesum[:, 11:] # (N, 337)
            # data_y_agesum = data_y_agesum[:, np.newaxis, :] # (N, 1, 337)        
            # # data_y_agesum = np.transpose(data_y_agesum, (1,0,2,3,4))
            # # add random-walk noise to target
            # if self.add_noise:
            #     data_y_agesum =  self.__add_random_walk_noise_to_batch__(data_y_agesum, self.noise_std)
            
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

        # (N, age, 28, 12, 1)
        # pft_MIX_BL = pft_MIX_BL.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1) # (N, age, 336, 1)
        # pft_MIX_BL = pft_MIX_BL[:, :, np.newaxis, :, :] # (N, age, 1, 336, 1)
        # pft_MIX_NL = pft_MIX_NL.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1)
        # pft_MIX_NL = pft_MIX_NL[:, :, np.newaxis, :, :]
        # pft_MIX_GS = pft_MIX_GS.reshape(data_y_arr.shape[1], data_y_arr.shape[0], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1)
        # pft_MIX_GS = pft_MIX_GS[:, :, np.newaxis, :, :]

        # (N, 28, 12, 1)
        # pft_MIX_BL_agesum = pft_MIX_BL_agesum.reshape(data_y_arr.shape[1], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1) # (N, 336, 1)
        # pft_MIX_BL_agesum = pft_MIX_BL_agesum[:, np.newaxis, :, :] # (N, 1, 336, 1)
        # pft_MIX_NL_agesum = pft_MIX_NL_agesum.reshape(data_y_arr.shape[1], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1)
        # pft_MIX_NL_agesum = pft_MIX_NL_agesum[:, np.newaxis, :, :]
        # pft_MIX_GS_agesum = pft_MIX_GS_agesum.reshape(data_y_arr.shape[1], (data_y_arr.shape[2]-1)*data_y_arr.shape[3], 1)
        # pft_MIX_GS_agesum = pft_MIX_GS_agesum[:, np.newaxis, :, :]

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
                data_y_agesum = data_y_agesum[idx_train] # (N, 1, 337, 3) or (N, 1, 337)
                AgeWeight = AgeWeight[idx_train] # (N, age)

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
        self.data_y_agesum = data_y_agesum.reshape(-1, data_y_agesum.shape[2], data_y_agesum.shape[3]) # (N, 1, 337, 3) to (-1, 337, 3)
        # elif self.stage == 'finetune':  
        #     self.data_y_agesum = data_y_agesum.reshape(-1, data_y_agesum.shape[2]) # (N, 1, 337) to (-1, 337)
        
        # self.pft_MIX_BL_agesum = pft_MIX_BL_agesum.reshape(-1, pft_MIX_BL_agesum.shape[2], pft_MIX_BL_agesum.shape[3]) # (N, 1, 336, 1) to (-1, 336, 1)
        # self.pft_MIX_NL_agesum = pft_MIX_NL_agesum.reshape(-1, pft_MIX_NL_agesum.shape[2], pft_MIX_NL_agesum.shape[3])
        # self.pft_MIX_GS_agesum = pft_MIX_GS_agesum.reshape(-1, pft_MIX_GS_agesum.shape[2], pft_MIX_GS_agesum.shape[3])
        
        self.AgeWeight = AgeWeight # (N, age)
        # self.treecover = treecover


        self.data_y = self.data_y * self.AgeWeight[:, :, np.newaxis, np.newaxis]   # (44,18,337,10)
        self.data_y = self.data_y.sum(axis=1)

        # self.data_y = self.data_y[...,[4,9,7]]
        print(f'{self.flag} data size x={self.data_x.shape}, y={self.data_y.shape}, data_y_agesum={self.data_y_agesum.shape},  AgeWeight={self.AgeWeight.shape}')

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_y_agesum = self.data_y_agesum[index]

        # seq_pft_BL = self.pft_MIX_BL[index]
        # seq_pft_NL = self.pft_MIX_NL[index]
        # seq_pft_GS = self.pft_MIX_GS[index]
        
        # seq_pft_BL_agesum = self.pft_MIX_BL_agesum[index]
        # seq_pft_NL_agesum = self.pft_MIX_NL_agesum[index]
        # seq_pft_GS_agesum = self.pft_MIX_GS_agesum[index]
        seq_aw = self.AgeWeight[index]
        # seq_tc = self.treecover[index]

        return seq_x, seq_y, seq_y_agesum, seq_aw

    def __len__(self):
        return self.data_x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
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
        
        # Step 1: reshape to merge years and months → (age, N, (Y+1)*12, 7)       
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
        N, years, channels, features = batch.shape  # (N, 1, 481, 7)  

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



    
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', stat_path='stat.npz', pft_path = 'pft.npz',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.stat_path = stat_path
        self.pft_path = pft_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
