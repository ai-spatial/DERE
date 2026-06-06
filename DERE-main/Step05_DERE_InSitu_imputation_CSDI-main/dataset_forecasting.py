import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

class Forecasting_Dataset_GPP(Dataset):
    def __init__(self, datatype, mode="train", 
                 stat_path=None, data_path=None, data_xpath=None, varname=None):
        # self.history_length = 1
        # self.pred_length = 347
        self.mode = mode
        self.datatype = datatype 
        self.val_ratio = 0.1 

        var_index_map = {
            "gpp":  (4, 0),
            "nee":  (7, 2),
            "reco": (9, 1),
            'all': (4,9,7),
        }


        # datafolder = './data/gpp/'
        # # self.test_length= 12*3
        # # self.valid_length = 12*1


        # # self.stat_path = datafolder+'data_stats_with_NEE_Ra.npz' # stat file
        # # self.data_path = datafolder+'pft_dataset_12mean_4types_28years_ESACCI_plusAW_insituless_update.npz'
        # # self.data_xpath = datafolder+'res_train4_test8_extract_4types_28years_insituless_update_with_NEE_Ra.npz'
        # # # self.pft_path = datafolder+''


        # self.stat_path = datafolder+'data_stats_with_NEE_Ra_RECO.npz' # stat file
        # self.data_path = datafolder+'pft_dataset_12mean_4types_28years_ESACCI_plusAW_insituless_update_ameriflux.npz'
        # self.data_xpath = datafolder+'res_train4_test8_extract_4types_28years_insituless_update_ameriflux_with_NEE_Ra_RECO.npz'
        # # self.pft_path = datafolder+''

        if stat_path is None or data_path is None or data_xpath is None:
            raise ValueError("please input stat_path, data_path, data_xpath")

        self.stat_path = stat_path
        self.data_path = data_path
        self.data_xpath = data_xpath

        stat_raw = np.load(self.stat_path)
        self.x_mean = stat_raw['x_mean']
        self.x_std = stat_raw['x_std']

        if datatype == "all":
            data_idx= var_index_map[varname]
            data_idx = list(data_idx)
        else:
            data_idx, insitu_idx = var_index_map[varname]

        self.mean_data = np.array(stat_raw['y_mean'][data_idx])
        self.std_data = np.array(stat_raw['y_std'][data_idx])

        data_raw = np.load(self.data_path)
        data_xraw = np.load(self.data_xpath)

        # # x = (N, 40, 12, 136)
        # # y = (age, N, 41, 12, 7)
        # pft_raw = np.load(self.pft_path)
        # # (N, 15, 41, 12)
        if mode == 'pre': #test
            if datatype == "all":
                data_y_train = data_raw['InSitu_train'][:,1:,:,:]
                data_y_test  = data_raw['InSitu_test'][:,1:,:,:]
            else:
                data_y_train = data_raw['InSitu_train'][:,1:,:,insitu_idx]
                data_y_test  = data_raw['InSitu_test'][:,1:,:,insitu_idx]

            data_x_train = data_xraw['x_train']
            data_x_test  = data_xraw['x_test']

            data_y = np.concatenate([data_y_train, data_y_test], axis=0)
            data_x = np.concatenate([data_x_train, data_x_test], axis=0)            
        
        elif mode == 'test': #test
            if datatype == "all":
                data_y = data_raw[f'InSitu_test'][:,1:,:,:] # (9, 28, 12)
            else:
                data_y = data_raw[f'InSitu_test'][:,1:,:,insitu_idx] # (9, 28, 12)
               
            data_x = data_xraw[f'x_test'] # (9, 28, 12, 136)            
        else: 
            if datatype == "all":
                data_y = data_raw[f'InSitu_train'][:,1:,:,:] # (9, 28, 12)
            else:
                data_y = data_raw[f'InSitu_train'][:,1:,:,insitu_idx] # (9, 28, 12)
            # data_y = data_raw[f'InSitu_train'][:,1:,:,insitu_idx]  # (33, 28, 12)
            data_x = data_xraw[f'x_train'] # (33, 28, 12, 136)   

            repeat_factor = 1
            data_x = np.tile(data_x, (repeat_factor, 1, 1, 1))
            data_y = np.tile(data_y, (repeat_factor, 1, 1))
 
            idx = np.arange(data_y.shape[0])
            np.random.shuffle(idx)
            val_count = int(len(idx) * self.val_ratio)
            if val_count < 1:
                val_count = 1  # at least one
            idx_val = idx[:val_count]
            idx_train = idx[val_count:]

            if mode == 'train':
                data_x = data_x[idx_train]      
                data_y = data_y[idx_train]
                
            elif mode == 'valid':
                data_x = data_x[idx_val]            
                data_y = data_y[idx_val]
    
        self.data_x = (data_x - self.x_mean) / self.x_std
        # self.data_x = self.data_x[...,:136]
        self.aux_x = self.data_x.reshape(self.data_x.shape[0], self.data_x.shape[1] * self.data_x.shape[2], self.data_x.shape[3])
        # self.aux_x = self.aux_x[:, :, 0:1]
            


        # Fill missing values with 0 after normalization
        # self.main_data = np.where(
        #     self.mask_data == 1,
        #     (data_y - self.mean_data) / self.std_data,
        #     np.nan
        # )  # shape: (N, 29, 12), normalized and filled

        self.main_data = (data_y - self.mean_data) / self.std_data
        # shape: (N, 29, 12), normalized and filled
        # Reshape time axis: (N, 29*12) = (N, 348)
        if datatype == "all":
            self.main_data = self.main_data.reshape(self.main_data.shape[0], -1,3)
        else:
            self.main_data = self.main_data.reshape(self.main_data.shape[0], -1)
            self.main_data = self.main_data[:, :, np.newaxis]
        # self.mask_data = self.mask_data.reshape(self.mask_data.shape[0], -1)

        

        self.main_data2 = np.concatenate([self.main_data, self.aux_x], axis=-1)
        self.main_data = self.main_data2

        # Create observed mask: 1 = observed, 0 = missing
        self.mask_data = (~np.isnan(self.main_data)).astype(np.float32)  # shape: (N, 29, 12)
        self.main_data = np.nan_to_num(self.main_data, nan=0.0)
        # Each sample corresponds to one site (entire time series)
        # self.use_index = np.arange(self.main_data.shape[0])
        # self.seq_length = self.history_length + self.pred_length  # for compatibility



        # data_y_arr = data_y.copy() # (N, 29, 12)
        # data_y = data_y.reshape(data_y_arr.shape[0], data_y_arr.shape[1]*data_y_arr.shape[2]) # (N, 29*12)

        # self.main_data = data_y
        # self.mean_data = self.y_mean
        # self.std_data = self.y_std

        # self.main_data = (self.main_data - self.mean_data) / self.std_data # (6001, 370)
        # total_length = len(self.main_data)
        # self.seq_length = self.history_length + self.pred_length

        # start = total_length - self.seq_length - self.test_length + self.pred_length
        # end = total_length - self.seq_length + self.pred_length
        # self.use_index = np.arange(start,end,self.pred_length)

        
    def __getitem__(self, idx):
        self.partial_months = 3
        site_data = self.main_data[idx]  # (336, V)
        site_mask = self.mask_data[idx]  # (336, V)

        site_data = site_data.astype(np.float32)
        site_mask = site_mask.astype(np.float32)
        gt_mask = site_mask.copy()

        # if self.mode in ['test', 'train', 'valid']:
            # === Step 1: reshape  (28, 12, V)
            # site_data_year = site_data.reshape(28, 12, -1)   # (28, 12, V)
        site_mask_year = site_mask.reshape(28, 12, -1)   # (28, 12, V)

        # === Step 2: find years with more than six months
        valid_years = []
        for y in range(28):
            if self.datatype == "all":
                months_with_values = (site_mask_year[y,:,2] > 0).sum()
            else:
                months_with_values = (site_mask_year[y,:,0] > 0).sum()
            if months_with_values >= 6:
                valid_years.append(y)

        # === Step 3: half to predict
        half = len(valid_years) // 2
        chosen_full = valid_years[:half]
        for y in chosen_full:
            if self.datatype == "all":
                gt_mask[y * 12:(y + 1) * 12, :3] = 0
            else:
                gt_mask[y * 12:(y + 1) * 12, 0] = 0

        # === Step 4: predict part of the months for other years 
        remaining = valid_years[half:]
        for y in remaining:
            # 
            if self.datatype == "all":
                months_with_values = np.where(site_mask_year[y,:, 2] > 0)[0]
            else:
                months_with_values = np.where(site_mask_year[y,:, 0] > 0)[0]
            
            if len(months_with_values) > 0:
                num_pick = min(self.partial_months, len(months_with_values))
                chosen_months = np.random.choice(months_with_values, num_pick, replace=False)
                for m in chosen_months:
                    # gt_mask[y * 12 + m, 0] = 0
                    if self.datatype == "all":
                        gt_mask[y * 12 + m, :3] = 0
                    else:
                        gt_mask[y * 12 + m, 0] = 0

        # elif self.mode in ['pre']:
        #     pass


        s = {
            "observed_data": site_data,   # (336, V)
            "observed_mask": site_mask,   # (336, V)
            "gt_mask": gt_mask,           # (336, V)
            "timepoints": np.arange(site_data.shape[0]) * 1.0,  # (336,)
            "feature_id": np.arange(site_data.shape[1]) * 1.0,  # (V,)
        }
        return s
    
    def __len__(self):
        return self.main_data.shape[0]

def get_dataloader(datatype,device,batch_size=8,
                   stat_path=None, data_path=None, data_xpath=None, varname=None):
    dataset = Forecasting_Dataset_GPP(datatype,mode='train',
                                            stat_path=stat_path,
                                            data_path=data_path,
                                            data_xpath=data_xpath, varname=varname)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Forecasting_Dataset_GPP(datatype,mode='valid',
                                            stat_path=stat_path,
                                            data_path=data_path,
                                            data_xpath=data_xpath, varname=varname)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Forecasting_Dataset_GPP(datatype,mode='test',
                                            stat_path=stat_path,
                                            data_path=data_path,
                                            data_xpath=data_xpath, varname=varname)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)
    
    pre_dataset = Forecasting_Dataset_GPP(datatype, mode='pre',
                                          stat_path=stat_path,
                                          data_path=data_path,
                                          data_xpath=data_xpath, varname=varname)
    pre_loader = DataLoader(pre_dataset, batch_size=batch_size, shuffle=0)

    scaler = torch.from_numpy(dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader, pre_loader, scaler, mean_scaler