from data.data_loader import Dataset_AddPure_Extract4types_CombineWith_PFT_Prediction_imputation, Dataset_Baseline, Dataset_KGML, Dataset_AddPure_Extract4types_AgeIndependentCom, Dataset_AddPure_Extract4types_CombineWith_PFT_Prediction, Dataset_AddPure_Extract4types, Dataset_ED_ALLAGE_PFT_Prediction
from exp.exp_basic import Exp_Basic
from models.model import MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure, OneBranch, OneBranch_ED_ALLAGE_PFT_Prediction

from utils.tools import EarlyStopping, EarlyStopping1, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary  

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure(Exp_Basic):
    def __init__(self, args):
        super(Exp_MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure': MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure,
        }
        if self.args.model=='MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure':
            e_layers = self.args.e_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
                
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ED': Dataset_AddPure_Extract4types,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            stat_path = args.stat_path,
            pft_path = args.pft_path,
            asi = args.asi,
            aei = args.aei,
            add_noise = args.add_noise,
            noise_std = args.noise_std,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def compute_total_loss(self, pred, true, criterion):
        """
        Compute total loss = prediction loss + physics-based consistency losses.
        pred/true are normalized. Need to de-normalize before physics loss.
        """
        # ---- 1. Base prediction loss (normalized space is fine here) ----
        loss = criterion(pred, true)

        # ---- 2. Denormalize to physical space ----
        if hasattr(self, "y_mean") and hasattr(self, "y_std"):
            pred_phys = self.__inverse_norm_data__(pred, self.y_mean, self.y_std)
            true_phys = self.__inverse_norm_data__(true, self.y_mean, self.y_std)
        else:
            pred_phys, true_phys = pred, true  # fallback

        # ---- 3. Physics-guided loss terms (physical space) ----
        # NEE = Rh - NPP
        loss_c1 = torch.mean((pred_phys[..., 7] - pred_phys[..., 6] + pred_phys[..., 5])**2)

        # Ra = GPP – NPP
        loss_c2 = torch.mean((pred_phys[..., 8] - pred_phys[..., 4] + pred_phys[..., 5])**2)

        # RECO = NEE + GPP
        loss_c3 = torch.mean((pred_phys[..., 9] - pred_phys[..., 4] - pred_phys[..., 7])**2)

        # ---- 4. Total loss ----
        total_loss = loss + 0.1 * loss_c1 + 0.1 * loss_c2 + 0.1 * loss_c3 
        # + 0.1 * loss_c3

        return total_loss


    def vali(self, vali_data, vali_loader, criterion, stage='pure', type='mix'):
        self.model.eval()
        total_loss = []

        batch_size = self.args.batch_size
        # Branch1 Loader
        dataset_BL = TensorDataset(
            torch.from_numpy(vali_data.data_x_BL).float(),
            torch.from_numpy(vali_data.data_y_BL).float(),
            torch.from_numpy(vali_data.pft_BL_BL).float(),
            torch.from_numpy(vali_data.pft_BL_NL).float(),
            torch.from_numpy(vali_data.pft_BL_GS).float()
        )
        loader_BL = DataLoader(dataset_BL, batch_size=batch_size, shuffle=True)

        # Branch2 Loader
        dataset_NL = TensorDataset(
            torch.from_numpy(vali_data.data_x_NL).float(),
            torch.from_numpy(vali_data.data_y_NL).float(),
            torch.from_numpy(vali_data.pft_NL_BL).float(),
            torch.from_numpy(vali_data.pft_NL_NL).float(),
            torch.from_numpy(vali_data.pft_NL_GS).float()
        )
        loader_NL = DataLoader(dataset_NL, batch_size=batch_size, shuffle=True)

        # Branch3 Loader
        dataset_GS = TensorDataset(
            torch.from_numpy(vali_data.data_x_GS).float(),
            torch.from_numpy(vali_data.data_y_GS).float(),
            torch.from_numpy(vali_data.pft_GS_BL).float(),
            torch.from_numpy(vali_data.pft_GS_NL).float(),
            torch.from_numpy(vali_data.pft_GS_GS).float()
        )
        loader_GS = DataLoader(dataset_GS, batch_size=batch_size, shuffle=True)

        # MIX Loader
        dataset_MIX = TensorDataset(
            torch.from_numpy(vali_data.data_x).float(),
            torch.from_numpy(vali_data.data_y).float(),
            torch.from_numpy(vali_data.pft_MIX_BL).float(),
            torch.from_numpy(vali_data.pft_MIX_NL).float(),
            torch.from_numpy(vali_data.pft_MIX_GS).float()
        )
        loader_MIX = DataLoader(dataset_MIX, batch_size=batch_size, shuffle=True)

        if type == 'BL':
            with torch.no_grad():  
                for i, (batch_x_BL,batch_y_BL,batch_pft_BL_bl, batch_pft_BL_nl, batch_pft_BL_gs) in enumerate(loader_BL):
                    pred_BL, true_BL = self._process_one_batch(
                        None, batch_x_BL, batch_y_BL, batch_pft_BL_bl, batch_pft_BL_nl, batch_pft_BL_gs, stage='pure',type = 'BL')  
                    loss = self.compute_total_loss(pred_BL, true_BL, criterion)               
                    total_loss.append(loss.item())
        if type == 'NL':
            with torch.no_grad():  
                for i, (batch_x_NL,batch_y_NL, batch_pft_NL_bl, batch_pft_NL_nl, batch_pft_NL_gs) in enumerate(loader_NL):
                    pred_NL, true_NL = self._process_one_batch(
                        None, batch_x_NL, batch_y_NL, batch_pft_NL_bl, batch_pft_NL_nl, batch_pft_NL_gs, stage='pure',type = 'NL')  
                    loss = self.compute_total_loss(pred_NL, true_NL, criterion)               
                    total_loss.append(loss.item())
        if type == 'GS':
            with torch.no_grad():  
                for i, (batch_x_GS, batch_y_GS,batch_pft_GS_bl, batch_pft_GS_nl, batch_pft_GS_gs) in enumerate(loader_GS):
                    pred_GS, true_GS = self._process_one_batch(
                        None, batch_x_GS, batch_y_GS, batch_pft_GS_bl, batch_pft_GS_nl, batch_pft_GS_gs, stage='pure',type = 'GS')      						            
                    loss = self.compute_total_loss(pred_GS, true_GS, criterion) 
                    total_loss.append(loss.item())

        if stage == 'mix':
            with torch.no_grad():  
                for i, (batch_x, batch_y,batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs) in enumerate(loader_MIX):     
                    pred_MIX, true_MIX = self._process_one_batch(
                        None, batch_x, batch_y, batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs, stage='mix', type='mix')             
                    loss = self.compute_total_loss(pred_MIX, true_MIX, criterion) 

                    total_loss.append(loss.item())


        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting, stage='pure'):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        batch_size = self.args.batch_size
        # Branch1 Loader
        dataset_BL = TensorDataset(
            torch.from_numpy(train_data.data_x_BL).float(),
            torch.from_numpy(train_data.data_y_BL).float(),
            torch.from_numpy(train_data.pft_BL_BL).float(),
            torch.from_numpy(train_data.pft_BL_NL).float(),
            torch.from_numpy(train_data.pft_BL_GS).float()
        )
        loader_BL = DataLoader(dataset_BL, batch_size=batch_size, shuffle=True)

        # Branch2 Loader
        dataset_NL = TensorDataset(
            torch.from_numpy(train_data.data_x_NL).float(),
            torch.from_numpy(train_data.data_y_NL).float(),
            torch.from_numpy(train_data.pft_NL_BL).float(),
            torch.from_numpy(train_data.pft_NL_NL).float(),
            torch.from_numpy(train_data.pft_NL_GS).float()
        )
        loader_NL = DataLoader(dataset_NL, batch_size=batch_size, shuffle=True)

        # Branch3 Loader
        dataset_GS = TensorDataset(
            torch.from_numpy(train_data.data_x_GS).float(),
            torch.from_numpy(train_data.data_y_GS).float(),
            torch.from_numpy(train_data.pft_GS_BL).float(),
            torch.from_numpy(train_data.pft_GS_NL).float(),
            torch.from_numpy(train_data.pft_GS_GS).float()
        )
        loader_GS = DataLoader(dataset_GS, batch_size=batch_size, shuffle=True)

        # MIX Loader
        dataset_MIX = TensorDataset(
            torch.from_numpy(train_data.data_x).float(),
            torch.from_numpy(train_data.data_y).float(),
            torch.from_numpy(train_data.pft_MIX_BL).float(),
            torch.from_numpy(train_data.pft_MIX_NL).float(),
            torch.from_numpy(train_data.pft_MIX_GS).float()
        )
        loader_MIX = DataLoader(dataset_MIX, batch_size=batch_size, shuffle=True)

        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        
        
        if stage == 'mix':
            early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)

            # load branch 1
            state_dict_full = torch.load(os.path.join(path, 'checkpoint_pure_joint.pth'))

            sd_b1 = {k.replace("branch1.", ""): v for k, v in state_dict_full.items() if k.startswith("branch1.")}
            sd_b2 = {k.replace("branch2.", ""): v for k, v in state_dict_full.items() if k.startswith("branch2.")}
            sd_b3 = {k.replace("branch3.", ""): v for k, v in state_dict_full.items() if k.startswith("branch3.")}

            self.model.branch1.load_state_dict(sd_b1)
            self.model.branch2.load_state_dict(sd_b2)
            self.model.branch3.load_state_dict(sd_b3)

            for name, param in self.model.named_parameters():
                if any(x in name for x in ['shared_enc_embedding', 'shared_encoder']):
                    param.requires_grad = False

            print("Freezing encoder-related layers (enc_embedding, encoder)")

            model_optim = self._select_optimizer()

        elif stage == 'pure':
            early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)

            model_optim = self._select_optimizer()       

        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if stage == 'pure':
            print("\n==== Joint Training: BL + NL + GS (same iteration) ====") 

            iters_per_epoch = min(len(loader_BL), len(loader_NL), len(loader_GS))
            w_BL, w_NL, w_GS =  0.33, 0.33, 0.33 

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss_bl, train_loss_nl, train_loss_gs, train_loss_tot = [], [], [], []

                self.model.train()
                epoch_time = time.time()
                time_now_local = time.time()
                it_BL, it_NL, it_GS = iter(loader_BL), iter(loader_NL), iter(loader_GS)

                for i in range(iters_per_epoch):
                    iter_count += 1

                    batch_x_BL, batch_y_BL, batch_pft_BL_bl, batch_pft_BL_nl, batch_pft_BL_gs = next(it_BL)
                    batch_x_NL, batch_y_NL, batch_pft_NL_bl, batch_pft_NL_nl, batch_pft_NL_gs = next(it_NL)
                    batch_x_GS, batch_y_GS, batch_pft_GS_bl, batch_pft_GS_nl, batch_pft_GS_gs = next(it_GS)


                    model_optim.zero_grad(set_to_none=True) 

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            pred_BL, true_BL = self._process_one_batch(
                                None, batch_x_BL, batch_y_BL,
                                batch_pft_BL_bl, batch_pft_BL_nl, batch_pft_BL_gs,
                                stage='pure', type='BL')

                            loss_BL = self.compute_total_loss(pred_BL, true_BL, criterion)  


                            pred_NL, true_NL = self._process_one_batch(
                                None, batch_x_NL, batch_y_NL,
                                batch_pft_NL_bl, batch_pft_NL_nl, batch_pft_NL_gs,
                                stage='pure', type='NL')


                            loss_NL = self.compute_total_loss(pred_NL, true_NL, criterion)

                            pred_GS, true_GS = self._process_one_batch(
                                None, batch_x_GS, batch_y_GS,
                                batch_pft_GS_bl, batch_pft_GS_nl, batch_pft_GS_gs,
                                stage='pure', type='GS')

                            loss_GS = self.compute_total_loss(pred_GS, true_GS, criterion) 


                            total_loss = w_BL*loss_BL + w_NL*loss_NL + w_GS*loss_GS 

                        scaler.scale(total_loss).backward()
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        pred_BL, true_BL = self._process_one_batch(
                            None, batch_x_BL, batch_y_BL,
                            batch_pft_BL_bl, batch_pft_BL_nl, batch_pft_BL_gs,
                            stage='pure', type='BL')

                        loss_BL = self.compute_total_loss(pred_BL, true_BL, criterion) 
                        
                        pred_NL, true_NL = self._process_one_batch(
                            None, batch_x_NL, batch_y_NL,
                            batch_pft_NL_bl, batch_pft_NL_nl, batch_pft_NL_gs,
                            stage='pure', type='NL')

                        loss_NL = self.compute_total_loss(pred_NL, true_NL, criterion)
                        
                        pred_GS, true_GS = self._process_one_batch(
                            None, batch_x_GS, batch_y_GS,
                            batch_pft_GS_bl, batch_pft_GS_nl, batch_pft_GS_gs,
                            stage='pure', type='GS')

                        loss_GS = self.compute_total_loss(pred_GS, true_GS, criterion) 
                        
                        total_loss = w_BL*loss_BL + w_NL*loss_NL + w_GS*loss_GS
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        model_optim.step()

                    train_loss_bl.append(loss_BL.item())
                    train_loss_nl.append(loss_NL.item())
                    train_loss_gs.append(loss_GS.item())
                    train_loss_tot.append(total_loss.item())

                    if (i+1) % 100==0:
                        print("\titers: {0}, epoch: {1} | loss_BL: {2:.7f} loss_NL: {3:.7f} loss_GS: {4:.7f} total: {5:.7f}"
                            .format(i + 1, epoch + 1,
                                    loss_BL.item(), loss_NL.item(), loss_GS.item(), total_loss.item()))
                        speed = (time.time()-time_now)/iter_count
                        left_time = speed*((self.args.train_epochs - epoch)*iters_per_epoch - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                train_bl  = float(np.average(train_loss_bl))
                train_nl  = float(np.average(train_loss_nl))
                train_gs  = float(np.average(train_loss_gs))
                train_tot = float(np.average(train_loss_tot))

                vali_bl = self.vali(vali_data, vali_loader, criterion, stage=stage, type='BL')
                vali_nl = self.vali(vali_data, vali_loader, criterion, stage=stage, type='NL')
                vali_gs = self.vali(vali_data, vali_loader, criterion, stage=stage, type='GS')

                test_bl = self.vali(test_data, test_loader, criterion, stage=stage, type='BL')
                test_nl = self.vali(test_data, test_loader, criterion, stage=stage, type='NL')
                test_gs = self.vali(test_data, test_loader, criterion, stage=stage, type='GS')

                print("Epoch: {0}, Steps: {1} | Train BL: {2:.7f} NL: {3:.7f} GS: {4:.7f} Tot: {5:.7f} | "
                    "Val BL: {6:.7f} NL: {7:.7f} GS: {8:.7f} | "
                    "Test BL: {9:.7f} NL: {10:.7f} GS: {11:.7f}".format(
                        epoch + 1, iters_per_epoch, train_bl, train_nl, train_gs, train_tot,
                        vali_bl, vali_nl, vali_gs, test_bl, test_nl, test_gs))

                val_joint = w_BL*vali_bl + w_NL*vali_nl + w_GS*vali_gs

                early_stopping(val_joint, self.model, os.path.join(path, 'checkpoint_pure_joint.pth'))


                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch+1, self.args)           

        elif stage == 'mix':  
            print("\n==== Training Branch0123 with Mixdata ====")
            for epoch in range(self.args.train_epochs):    
                iter_count = 0
                train_loss = []
                
                self.model.train()
                epoch_time = time.time()
                for i, (batch_x,batch_y,batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs) in enumerate(loader_MIX):
                    iter_count += 1
                    model_optim.zero_grad()
                                        
                    pred_MIX, true_MIX = self._process_one_batch(
                        None, batch_x, batch_y, batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs, stage='mix', type='mix')                  

                    loss = self.compute_total_loss(pred_MIX, true_MIX, criterion) 

                    train_loss.append(loss.item())
                                    
                    if (i+1) % 100==0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time()-time_now)/iter_count
                        left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward() # Backward Pass, backpropagation
                        model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion, stage=stage)
                test_loss = self.vali(test_data, test_loader, criterion, stage=stage)
                
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                # early_stopping(vali_loss, self.model, path)

                early_stopping(vali_loss, self.model, os.path.join(path, 'checkpoint_mix.pth'))

                if early_stopping.early_stop:
                    print("Early stopping")
                    break


                adjust_learning_rate(model_optim, epoch+1, self.args)

        if stage == 'pure':
            joint_path = os.path.join(path, 'checkpoint_pure_joint.pth')
            sd = torch.load(joint_path, map_location=self.device)
            sd_b1 = {k.replace("branch1.", ""): v for k, v in sd.items() if k.startswith("branch1.")}
            sd_b2 = {k.replace("branch2.", ""): v for k, v in sd.items() if k.startswith("branch2.")}
            sd_b3 = {k.replace("branch3.", ""): v for k, v in sd.items() if k.startswith("branch3.")}
            self.model.branch1.load_state_dict(sd_b1)
            self.model.branch2.load_state_dict(sd_b2)
            self.model.branch3.load_state_dict(sd_b3)

        elif stage == 'mix':
            best_model_path = os.path.join(path, 'checkpoint_mix.pth')
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            raise ValueError(f"Unsupported stage: {stage}")
      
        return self.model

    
    def test(self, setting, stage='mix'):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        test_data, test_loader = self._get_data(flag='test')

        batch_size = self.args.batch_size

        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)

        path = os.path.join(self.args.checkpoints, setting)
        # best_model_path = path+'/'+'checkpoint.pth'
        if stage == 'mix':
            # MIX Loader
            dataset_MIX = TensorDataset(
                torch.from_numpy(test_data.data_x).float(),
                torch.from_numpy(test_data.data_y).float(),
                torch.from_numpy(test_data.pft_MIX_BL).float(),
                torch.from_numpy(test_data.pft_MIX_NL).float(),
                torch.from_numpy(test_data.pft_MIX_GS).float()
            )
            loader_MIX = DataLoader(dataset_MIX, batch_size=batch_size, shuffle=False, drop_last=False)
            best_model_path = os.path.join(path, 'checkpoint_mix.pth')
            self.model.load_state_dict(torch.load(best_model_path))
            print('load model from:', best_model_path)       
            self.model.eval()

            outputs_BLs = []
            outputs_NLs = []
            outputs_GSs = []
            outputs_MIXs = []
            batch_ys = []

            PFTs = []
            COMs = []
            COM0s = []
            with torch.no_grad():  
                for i, (batch_x,batch_y,batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs) in enumerate(loader_MIX):                   
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, batch_y, PFT, COM, COM0 = self._process_one_batch_3branch_pft(None, batch_x, batch_y, batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs, stage='mix', type='mix')
                    outputs_BLs.append(outputs_BL.detach().cpu().numpy())
                    outputs_NLs.append(outputs_NL.detach().cpu().numpy())
                    outputs_GSs.append(outputs_GS.detach().cpu().numpy())                        
                    outputs_MIXs.append(outputs_MIX.detach().cpu().numpy())
                    batch_ys.append(batch_y.detach().cpu().numpy())

                    PFTs.append(PFT.detach().cpu().numpy())
                    COMs.append(COM.detach().cpu().numpy())
                    COM0s.append(COM0.detach().cpu().numpy())

            # outputs_BLs = np.array(outputs_BLs)
            outputs_BLs = np.concatenate(outputs_BLs, axis=0)
            outputs_NLs = np.concatenate(outputs_NLs, axis=0)
            outputs_GSs = np.concatenate(outputs_GSs, axis=0)
            outputs_MIXs = np.concatenate(outputs_MIXs, axis=0)
            batch_ys    = np.concatenate(batch_ys,    axis=0)
            PFTs        = np.concatenate(PFTs,        axis=0)
            COMs        = np.concatenate(COMs,        axis=0)
            COM0s       = np.concatenate(COM0s,       axis=0)
            print('test shape:', outputs_MIXs.shape, batch_ys.shape)           
            outputs_BLs = outputs_BLs.reshape(-1, outputs_BLs.shape[-2], outputs_BLs.shape[-1])
            outputs_NLs = outputs_NLs.reshape(-1, outputs_NLs.shape[-2], outputs_NLs.shape[-1])
            outputs_GSs = outputs_GSs.reshape(-1, outputs_GSs.shape[-2], outputs_GSs.shape[-1])
            outputs_MIXs = outputs_MIXs.reshape(-1, outputs_MIXs.shape[-2], outputs_MIXs.shape[-1])
            batch_ys = batch_ys.reshape(-1, batch_ys.shape[-2], batch_ys.shape[-1])


            PFTs = PFTs.reshape(-1, PFTs.shape[-2], PFTs.shape[-1])
            # COMs = np.array(COMs)
            COMs = COMs.reshape(-1, COMs.shape[-2], COMs.shape[-1])      
            # COM0s = np.array(COM0s)
            COM0s = COM0s.reshape(-1, COM0s.shape[-2], COM0s.shape[-1])  

            # result save
            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae3, mse3, rmse3, mape3, mspe3 = metric(outputs_MIXs, batch_ys)
            print('mse:{}, mae:{}'.format(mse3, mae3))

            np.save(folder_path+'metrics_for_Normalized_Data.npy', np.array([ mae3, mse3, rmse3, mape3, mspe3]))
            np.save(folder_path+'Normalized_pred_BL.npy', outputs_BLs)
            np.save(folder_path+'Normalized_pred_NL.npy', outputs_NLs)
            np.save(folder_path+'Normalized_pred_GS.npy', outputs_GSs)
            np.save(folder_path+'Normalized_pred_MIX.npy', outputs_MIXs)
            np.save(folder_path+'Normalized_true_MIX.npy', batch_ys)
            np.save(folder_path+'pred_MIX_PFT.npy', PFTs)
            np.save(folder_path+'pred_MIX_COM.npy', COMs)
            np.save(folder_path+'pred_MIX_COM0.npy', COM0s)

        
        elif stage == 'pure':
            # Branch1 Loader
            dataset_BL = TensorDataset(
                torch.from_numpy(test_data.data_x_BL).float(),
                torch.from_numpy(test_data.data_y_BL).float(),
                torch.from_numpy(test_data.pft_BL_BL).float(),
                torch.from_numpy(test_data.pft_BL_NL).float(),
                torch.from_numpy(test_data.pft_BL_GS).float()
            )
            loader_BL = DataLoader(dataset_BL, batch_size=batch_size, shuffle=False, drop_last=False)

            # Branch2 Loader
            dataset_NL = TensorDataset(
                torch.from_numpy(test_data.data_x_NL).float(),
                torch.from_numpy(test_data.data_y_NL).float(),
                torch.from_numpy(test_data.pft_NL_BL).float(),
                torch.from_numpy(test_data.pft_NL_NL).float(),
                torch.from_numpy(test_data.pft_NL_GS).float()
            )
            loader_NL = DataLoader(dataset_NL, batch_size=batch_size, shuffle=False, drop_last=False)

            # Branch3 Loader
            dataset_GS = TensorDataset(
                torch.from_numpy(test_data.data_x_GS).float(),
                torch.from_numpy(test_data.data_y_GS).float(),
                torch.from_numpy(test_data.pft_GS_BL).float(),
                torch.from_numpy(test_data.pft_GS_NL).float(),
                torch.from_numpy(test_data.pft_GS_GS).float()
            )
            loader_GS = DataLoader(dataset_GS, batch_size=batch_size, shuffle=False, drop_last=False)
            

            joint_path = os.path.join(path, 'checkpoint_pure_joint.pth')
            sd = torch.load(joint_path, map_location=self.device)
            sd_b1 = {k.replace("branch1.", ""): v for k, v in sd.items() if k.startswith("branch1.")}
            sd_b2 = {k.replace("branch2.", ""): v for k, v in sd.items() if k.startswith("branch2.")}
            sd_b3 = {k.replace("branch3.", ""): v for k, v in sd.items() if k.startswith("branch3.")}
            self.model.branch1.load_state_dict(sd_b1)
            self.model.branch2.load_state_dict(sd_b2)
            self.model.branch3.load_state_dict(sd_b3)
  
            self.model.eval()           
            outputs_BLs = []
            batch_ys = []   
            with torch.no_grad():        
                for i, (batch_x_BL,batch_y_BL,batch_pft_BL_bl, batch_pft_BL_nl, batch_pft_BL_gs) in enumerate(loader_BL):
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, batch_y, PFT, COM = self._process_one_batch_3branch_pft_pure(None, batch_x_BL,batch_y_BL,batch_pft_BL_bl, batch_pft_BL_nl, batch_pft_BL_gs, stage='pure', type='BL')
                    outputs_BLs.append(outputs_BL.detach().cpu().numpy())
                    batch_ys.append(batch_y.detach().cpu().numpy())
            outputs_BLs1 = np.array(outputs_BLs)
            BLs = np.array(batch_ys)
  
    
            outputs_BLs = []
            batch_ys = []  
            with torch.no_grad():            
                for i, (batch_x_NL,batch_y_NL,batch_pft_NL_bl, batch_pft_NL_nl, batch_pft_NL_gs) in enumerate(loader_NL):
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, batch_y, PFT, COM = self._process_one_batch_3branch_pft_pure(None, batch_x_NL,batch_y_NL,batch_pft_NL_bl, batch_pft_NL_nl, batch_pft_NL_gs, stage='pure', type='NL')
                    outputs_BLs.append(outputs_BL.detach().cpu().numpy())
                    batch_ys.append(batch_y.detach().cpu().numpy())
            outputs_NLs = np.array(outputs_BLs)
            NLs = np.array(batch_ys)

          
            outputs_BLs = []
            batch_ys = [] 
            with torch.no_grad():             
                for i, (batch_x_GS,batch_y_GS,batch_pft_GS_bl, batch_pft_GS_nl, batch_pft_GS_gs) in enumerate(loader_GS):
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, batch_y, PFT, COM = self._process_one_batch_3branch_pft_pure(None, batch_x_GS,batch_y_GS,batch_pft_GS_bl, batch_pft_GS_nl, batch_pft_GS_gs, stage='pure', type='GS')
                    outputs_BLs.append(outputs_BL.detach().cpu().numpy())
                    batch_ys.append(batch_y.detach().cpu().numpy())
            outputs_GSs = np.array(outputs_BLs)
            GSs = np.array(batch_ys)
            
            # print('test shape:', outputs_MIXs.shape, batch_ys.shape)
            outputs_BLs = outputs_BLs1.reshape(-1, outputs_BLs1.shape[-2], outputs_BLs1.shape[-1])
            outputs_NLs = outputs_NLs.reshape(-1, outputs_NLs.shape[-2], outputs_NLs.shape[-1])
            outputs_GSs = outputs_GSs.reshape(-1, outputs_GSs.shape[-2], outputs_GSs.shape[-1])

            BLs = BLs.reshape(-1, BLs.shape[-2], BLs.shape[-1])
            NLs = NLs.reshape(-1, NLs.shape[-2], NLs.shape[-1])
            GSs = GSs.reshape(-1, GSs.shape[-2], GSs.shape[-1])

            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path+'Normalized_pred_BL_step1.npy', outputs_BLs)
            np.save(folder_path+'Normalized_pred_NL_step1.npy', outputs_NLs)
            np.save(folder_path+'Normalized_pred_GS_step1.npy', outputs_GSs)
            np.save(folder_path+'Normalized_true_BL_step1.npy', BLs)
            np.save(folder_path+'Normalized_true_NL_step1.npy', NLs)
            np.save(folder_path+'Normalized_true_GS_step1.npy', GSs)
        
        return
    

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
        
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path+'/'+'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #         print('load model from:', best_model_path)

    #     self.model.eval()
        
    #     preds = []
        
    #     for i, (batch_x,batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g) in enumerate(pred_loader):
    #         pred, true = self._process_one_batch(
    #             pred_data, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g)
    #         preds.append(pred.detach().cpu().numpy())

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
    #     # result save
    #     folder_path = './results/' + setting +'/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
        
    #     np.save(folder_path+'real_prediction.npy', preds)
        
    #     return
    

    def _process_one_batch_3branch_pft_pure(self, batch_ws, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_gs, stage='pure', type='BL'):
        batch_x = batch_x.float().to(self.device) # [32, 480, 136]
        batch_y = batch_y.float() # [32, 481, 10]
        # batch_y = batch_y[:,batch_y.shape[1]-self.args.label_len - self.args.pred_len:,:] # Shuo: Adjust label length  # [32, 13, 10]
        batch_pft_bl = batch_pft_bl.float() # [32, 480, 1]
        batch_pft_nl = batch_pft_nl.float() # [32, 480, 1]
        batch_pft_gs = batch_pft_gs.float() # [32, 480, 1]
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 481, 10]

        dec_inp_MIX = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 481, 10]
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,'pure',type)[0]
                else:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,'pure',type)
        else:
            if self.args.output_attention:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,'pure',type)[0]
            else:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,'pure',type)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        

        batch_y = batch_y[:,:,f_dim:]
        outputs_BL = outputs_BL[:,:,f_dim:]
        outputPFT = outs3branchPft[0] # [32,480,3]
        outputCOM = outs3branchPft[1] # [32,480,1]
        

        return outputs_BL, outputs_NL, outputs_GS, outputs_MIX, batch_y, outputPFT, outputCOM
    


    def _process_one_batch_3branch_pft(self, batch_ws, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_gs, stage='mix', type='mix'):
        batch_x = batch_x.float().to(self.device) # [32, 480, 136]
        batch_y = batch_y.float() # [32, 481, 10]
        batch_pft_bl = batch_pft_bl.float() # [32, 480, 1]
        batch_pft_nl = batch_pft_nl.float() # [32, 480, 1]
        batch_pft_gs = batch_pft_gs.float() # [32, 480, 1]
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 481, 10]

        dec_inp_MIX = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 481, 10]
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,'mix')[0]
                else:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,'mix')
        else:
            if self.args.output_attention:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,'mix')[0]
            else:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,'mix')

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        

        batch_y = batch_y[:,:,f_dim:]
        outputs_BL = outputs_BL[:,:,f_dim:]
        outputs_NL = outputs_NL[:,:,f_dim:]
        outputs_GS = outputs_GS[:,:,f_dim:]
        outputs_MIX = outputs_MIX[:,:,f_dim:]

        outputPFT = outs3branchPft[0] # [32,480,3]
        outputCOM = outs3branchPft[1]
        outputCOM0 = outs3branchPft[2]# [32,480,1]
        

        return outputs_BL, outputs_NL, outputs_GS, outputs_MIX, batch_y, outputPFT, outputCOM, outputCOM0

    def _process_one_batch(self, batch_ws, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_gs, stage, type):
        batch_x = batch_x.float().to(self.device) # [32, 480, 136]
        batch_y = batch_y.float() # [32, 481, 10]
        batch_pft_bl = batch_pft_bl.float().to(self.device) # [32, 480, 1]
        batch_pft_nl = batch_pft_nl.float().to(self.device) # [32, 480, 1]
        batch_pft_gs = batch_pft_gs.float().to(self.device) # [32, 480, 1]
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp_MIX = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 481, 10]

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws, stage,type)[0]
                else:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws, stage,type)
        else:
            if self.args.output_attention:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws, stage,type)[0]
            else:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws, stage,type)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        batch_y = batch_y[:,:,f_dim:]
        if stage == 'pure':
            outputs_BL = outputs_BL[:,:,f_dim:]
            outputs = outputs_BL

        elif stage == 'mix':
            outputs_MIX = outputs_MIX[:,:,f_dim:]           
            outputs = outputs_MIX

        return outputs, batch_y 
 

class Exp_MultiBranch_AddPure_AddCom_Finetune_ageindependentCom_ShareStructure(Exp_Basic):
    def __init__(self, args):
        super(Exp_MultiBranch_AddPure_AddCom_Finetune_ageindependentCom_ShareStructure, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure': MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure,
        }
        if self.args.model=='MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure':
            e_layers = self.args.e_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
                
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ED': Dataset_AddPure_Extract4types_AgeIndependentCom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            stat_path = args.stat_path,
            pft_path = args.pft_path,
            asi = args.asi,
            aei = args.aei,
            add_noise = args.add_noise,
            noise_std = args.noise_std,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def compute_total_loss(self, pred, true, criterion):
        # ---- 1. Base prediction loss (normalized space is fine here) ----
        loss = criterion(pred, true)

        # ---- 2. Denormalize to physical space ----
        if hasattr(self, "y_mean") and hasattr(self, "y_std"):
            pred_phys = self.__inverse_norm_data__(pred, self.y_mean, self.y_std)
            true_phys = self.__inverse_norm_data__(true, self.y_mean, self.y_std)
        else:
            pred_phys, true_phys = pred, true  # fallback

        # ---- 3. Physics-guided loss terms (physical space) ----
        # NEE = Rh - NPP
        loss_c1 = torch.mean((pred_phys[..., 7] - pred_phys[..., 6] + pred_phys[..., 5])**2)

        # Ra = GPP – NPP
        loss_c2 = torch.mean((pred_phys[..., 8] - pred_phys[..., 4] + pred_phys[..., 5])**2)

        # RECO = NEE + GPP
        loss_c3 = torch.mean((pred_phys[..., 9] - pred_phys[..., 4] - pred_phys[..., 7])**2)

        # ---- 4. Total loss ----
        total_loss = loss + 0.1 * loss_c1 + 0.1 * loss_c2 + 0.1 * loss_c3

        return total_loss
            
    def vali(self, vali_data, vali_loader, criterion, stage='ageindep', type='mix'):
        self.model.eval()
        total_loss = []

        batch_size = self.args.batch_size

        pure_out_weighted = self._weight_pure_outputs(
            vali_data.data_x,  # shape: (N, 18, T, F)
            vali_data.data_y,
            vali_data.AgeWeight,
            vali_data.pft_MIX_BL,
            vali_data.pft_MIX_NL,
            vali_data.pft_MIX_GS,            
            self.model.branch1, self.model.branch2, self.model.branch3,
            self.device
        )

        dataset_ageindep = TensorDataset(
            pure_out_weighted.float(),                  # (N, T, F)
            torch.from_numpy(vali_data.data_x[:, 0, :, :]).float(),      # (N, T, F)
            torch.from_numpy(vali_data.data_y_agesum).float(),           # (N, T+1, 10)
            torch.from_numpy(vali_data.pft_MIX_BL_agesum).float(),       # (N, T, 1)
            torch.from_numpy(vali_data.pft_MIX_NL_agesum).float(),
            torch.from_numpy(vali_data.pft_MIX_GS_agesum).float()
        )

        loader_ageindep = DataLoader(dataset_ageindep, batch_size=batch_size, shuffle=True)

        if stage == 'ageindep':
            with torch.no_grad():  
                for i, (batch_ws, batch_x, batch_y,batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs) in enumerate(loader_ageindep):
                # loss_mix = criterion(pred_MIX.detach().cpu(), true_MIX.detach().cpu())           
                    pred_MIX, true_MIX = self._process_one_batch(
                        batch_ws, batch_x, batch_y, batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs, stage='ageindep', type='ageindep')             
                    loss = self.compute_total_loss(pred_MIX, true_MIX, criterion) 
                    total_loss.append(loss.item())


        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def _weight_pure_outputs(self, data_x, data_y, AgeWeight, pft_BL, pft_NL, pft_GS, model_branch1, model_branch2, model_branch3, device):
        N, age, T, Ff = data_x.shape
        N1, age1, T1, Vv = data_y.shape
        data_x_flat = data_x.reshape(-1, T, Ff)
        data_y_flat = data_y.reshape(-1, T+1, Vv)


        data_x_tensor = torch.from_numpy(data_x_flat).float()
        data_y_tensor0 = torch.from_numpy(data_y_flat).float()

        batch_size = 8

        outputs_BL = []
        outputs_NL = []
        outputs_GS = []

        for i in range(0, data_x_tensor.size(0), batch_size):
            batch_x = data_x_tensor[i:i+batch_size].to(device, non_blocking=True)
            batch_y0 = data_y_tensor0[i:i+batch_size].to(device, non_blocking=True)
            
            if self.args.padding == 0:
                dec_inp = torch.zeros([batch_y0.shape[0], self.args.pred_len, batch_y0.shape[-1]], device=device)
            elif self.args.padding == 1:
                dec_inp = torch.ones([batch_y0.shape[0], self.args.pred_len, batch_y0.shape[-1]], device=device)

            batch_y = torch.cat([batch_y0[:, :self.args.label_len, :], dec_inp], dim=1)

            with torch.no_grad():
                out_BL = model_branch1(batch_x, batch_y, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
            outputs_BL.append(out_BL.detach().to('cpu'))
            del out_BL
            torch.cuda.empty_cache()

            with torch.no_grad():
                out_NL = model_branch2(batch_x, batch_y, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
            outputs_NL.append(out_NL.detach().to('cpu'))
            del out_NL
            torch.cuda.empty_cache()

            with torch.no_grad():
                out_GS = model_branch3(batch_x, batch_y, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
            outputs_GS.append(out_GS.detach().to('cpu'))
            del out_GS, batch_x, batch_y, batch_y0, dec_inp
            torch.cuda.empty_cache()
            

        out_BL = torch.cat(outputs_BL, dim=0)  # (N*18, T, D)
        out_NL = torch.cat(outputs_NL, dim=0)
        out_GS = torch.cat(outputs_GS, dim=0)
        del outputs_BL, outputs_NL, outputs_GS


        D = out_BL.shape[-1]
        out_BL_mix = out_BL.reshape(N, age, T, D)
        out_NL_mix = out_NL.reshape(N, age, T, D)
        out_GS_mix = out_GS.reshape(N, age, T, D)
        del out_BL, out_NL, out_GS

        y_mean_cpu = self.y_mean.detach().to('cpu')
        y_std_cpu  = self.y_std.detach().to('cpu')
        
        out_denorm = self.__inverse_norm_data__(out_BL_mix, y_mean_cpu, y_std_cpu)
        mask = torch.ones(out_denorm.shape[-1], device=out_BL_mix.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        # out_BL = self.__norm_data__(out_denorm_relu, y_mean_cpu, y_std_cpu)
        out_BL = out_denorm_relu
        del out_BL_mix
        # out_NL = F.relu(self.__inverse_norm_data__(out_NL_mix, y_mean_cpu, y_std_cpu)); 
        out_denorm = self.__inverse_norm_data__(out_NL_mix, y_mean_cpu, y_std_cpu)
        mask = torch.ones(out_denorm.shape[-1], device=out_NL_mix.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        # out1_NL = self.__norm_data__(out_denorm_relu, y_mean_cpu, y_std_cpu)
        out_NL = out_denorm_relu
        del out_NL_mix
        # out_GS = F.relu(self.__inverse_norm_data__(out_GS_mix, y_mean_cpu, y_std_cpu)); 
        
        out_denorm = self.__inverse_norm_data__(out_GS_mix, y_mean_cpu, y_std_cpu)
        mask = torch.ones(out_denorm.shape[-1], device=out_GS_mix.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        # out_GS = self.__norm_data__(out_denorm_relu, y_mean_cpu, y_std_cpu)
        out_GS = out_denorm_relu
        del out_GS_mix
        torch.cuda.empty_cache()


        f_dim = -1 if self.args.features=='MS' else 0
        out_BL = out_BL[:,:,-self.args.pred_len:,f_dim:]
        out_BL = out_BL[:,:,:,f_dim:]
        out_NL = out_NL[:,:,-self.args.pred_len:,f_dim:]
        out_NL = out_NL[:,:,:,f_dim:]
        out_GS = out_GS[:,:,-self.args.pred_len:,f_dim:]
        out_GS = out_GS[:,:,:,f_dim:]
        
        pft_BL_tensor = torch.from_numpy(pft_BL).float()  # (N, age, T, 1)
        pft_NL_tensor = torch.from_numpy(pft_NL).float()
        pft_GS_tensor = torch.from_numpy(pft_GS).float()

        AgeWeight_tensor = torch.from_numpy(AgeWeight).unsqueeze(-1).unsqueeze(-1)  # (N, 18, 1, 1)

        out_fused_chunks = []
        for i in range(0, N, batch_size):
            out_BL_chunk = out_BL[i:i+batch_size].to(device, non_blocking=True)
            out_NL_chunk = out_NL[i:i+batch_size].to(device, non_blocking=True)
            out_GS_chunk = out_GS[i:i+batch_size].to(device, non_blocking=True)

            pft_BL_aw = pft_BL_tensor[i:i+batch_size].to(device, non_blocking=True)
            pft_NL_aw = pft_NL_tensor[i:i+batch_size].to(device, non_blocking=True)
            pft_GS_aw = pft_GS_tensor[i:i+batch_size].to(device, non_blocking=True)

            AgeW_chunk = AgeWeight_tensor[i:i+batch_size].to(device, non_blocking=True)

            fused_age = out_BL_chunk * pft_BL_aw + out_NL_chunk * pft_NL_aw + out_GS_chunk * pft_GS_aw
            fused = (fused_age * AgeW_chunk).sum(dim=1)

            out_fused_chunks.append(fused)

            del out_BL_chunk, out_NL_chunk, out_GS_chunk, pft_BL_aw, pft_NL_aw, pft_GS_aw, AgeW_chunk, fused_age, fused
            torch.cuda.empty_cache()

        out_fused = torch.cat(out_fused_chunks, dim=0)  # (N, T, D)
        
        del out_BL, out_NL, out_GS, pft_BL_tensor, pft_NL_tensor, pft_GS_tensor, AgeWeight_tensor
        torch.cuda.empty_cache()

        return out_fused
    

    def train(self, setting, stage='pure'):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        batch_size = self.args.batch_size

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        best_previous_model_path = os.path.join(path, 'checkpoint_mix.pth')
        self.model.load_state_dict(torch.load(best_previous_model_path))
        print('load model from:', best_previous_model_path) 

        for name, param in self.model.named_parameters():
            if any(x in name for x in ['branch1', 'branch2', 'branch3']):
                param.requires_grad = False
            else:
                param.requires_grad = True

        print("Freezing pure branch and Finetuning Competition")

        pure_out_weighted = self._weight_pure_outputs(
            train_data.data_x,  # shape: (N, 18, T, F)
            train_data.data_y,
            train_data.AgeWeight,
            train_data.pft_MIX_BL, # (N, age, 336, 1)
            train_data.pft_MIX_NL,
            train_data.pft_MIX_GS,            
            self.model.branch1, self.model.branch2, self.model.branch3,
            self.device
        )

        dataset_ageindep = TensorDataset(
            pure_out_weighted.float(),                          # (N * 5, T, F)
            torch.from_numpy(train_data.data_x[:, 0, :, :]).float(),                   # (N * 5, T, F)
            torch.from_numpy(train_data.data_y_agesum).float(),                   # (N * 5, T+1, 10)
            torch.from_numpy(train_data.pft_MIX_BL_agesum).float(),                   # (N * 5, T, 1)
            torch.from_numpy(train_data.pft_MIX_NL_agesum).float(),                   
            torch.from_numpy(train_data.pft_MIX_GS_agesum).float()
        )

        loader_ageindep = DataLoader(dataset_ageindep, batch_size=batch_size, shuffle=True)


        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)


        time_now = time.time()
        
        train_steps = len(loader_ageindep)
                
        # if stage == 'mix':
        early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)



        model_optim = self._select_optimizer()      
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()          

        print("\n==== Training Competition Model ====")
        for epoch in range(self.args.train_epochs):    
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()

            for i, (batch_ws, batch_x,batch_y,batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs) in enumerate(loader_ageindep):
                iter_count += 1
                model_optim.zero_grad()                                              
                pred_MIX, true_MIX = self._process_one_batch(
                    batch_ws, batch_x, batch_y, batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs, stage='ageindep', type='ageindep')                  

                loss = self.compute_total_loss(pred_MIX, true_MIX, criterion) 

                train_loss.append(loss.item())
                                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward() # Backward Pass, backpropagation
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, stage='ageindep')
            test_loss = self.vali(test_data, test_loader, criterion, stage='ageindep')
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # early_stopping(vali_loss, self.model, path)

            early_stopping(vali_loss, self.model, os.path.join(path, 'checkpoint_finetune.pth'))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)


        best_model_path = os.path.join(path, 'checkpoint_finetune.pth')
        self.model.load_state_dict(torch.load(best_model_path))

      
        return self.model

    
    def test(self, setting, stage='mix'):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        test_data, test_loader = self._get_data(flag='test')

        batch_size = self.args.batch_size

        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)

        path = os.path.join(self.args.checkpoints, setting)

        if stage == 'ageindep':
            best_previous_model_path = os.path.join(path, 'checkpoint_finetune.pth')
            self.model.load_state_dict(torch.load(best_previous_model_path))
            print('load model from:', best_previous_model_path)     
            self.model.eval()


            pure_out_weighted = self._weight_pure_outputs(
                test_data.data_x,  # shape: (N, 18, T, F)
                test_data.data_y,
                test_data.AgeWeight,
                test_data.pft_MIX_BL, # (N, age, 336, 1)
                test_data.pft_MIX_NL,
                test_data.pft_MIX_GS,            
                self.model.branch1, self.model.branch2, self.model.branch3,
                self.device
            )


            dataset_ageindep = TensorDataset(
                pure_out_weighted.float(),                  # (N, T, F)
                torch.from_numpy(test_data.data_x[:, 0, :, :]).float(),      # (N, T, F)
                torch.from_numpy(test_data.data_y_agesum).float(),           # (N, T+1, 10)
                torch.from_numpy(test_data.pft_MIX_BL_agesum).float(),       # (N, T, 1)
                torch.from_numpy(test_data.pft_MIX_NL_agesum).float(),
                torch.from_numpy(test_data.pft_MIX_GS_agesum).float()
            )

            loader_ageindep = DataLoader(dataset_ageindep, batch_size=batch_size, shuffle=True)

            outputs_MIXs = []
            batch_ys = []

            COM0s = []
            COMs = []
            PFTs = []
            
            for i, (batch_ws, batch_x,batch_y,batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs) in enumerate(loader_ageindep):               
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, batch_y, PFT, COM, COM0 = self._process_one_batch_3branch_pft(batch_ws, batch_x, batch_y, batch_pft_MIX_bl, batch_pft_MIX_nl, batch_pft_MIX_gs, stage='ageindep', type='ageindep')                    
                outputs_MIXs.append(outputs_MIX.detach().cpu().numpy())
                batch_ys.append(batch_y.detach().cpu().numpy())
                PFTs.append(PFT.detach().cpu().numpy())
                COMs.append(COM.detach().cpu().numpy())
                COM0s.append(COM0.detach().cpu().numpy())

            outputs_MIXs = np.concatenate(outputs_MIXs, axis=0)
            batch_ys = np.concatenate(batch_ys, axis=0)

            print('test shape:', outputs_MIXs.shape, batch_ys.shape)
            outputs_MIXs = outputs_MIXs.reshape(-1, outputs_MIXs.shape[-2], outputs_MIXs.shape[-1])
            batch_ys = batch_ys.reshape(-1, batch_ys.shape[-2], batch_ys.shape[-1])
            print('test shape:', outputs_MIXs.shape, batch_ys.shape)

            PFTs = np.concatenate(PFTs, axis=0)
            COMs = np.concatenate(COMs, axis=0)
            COM0s = np.concatenate(COM0s, axis=0)            
            # PFTs = np.array(PFTs)
            PFTs = PFTs.reshape(-1, PFTs.shape[-2], PFTs.shape[-1])
            # COMs = np.array(COMs)
            COMs = COMs.reshape(-1, COMs.shape[-2], COMs.shape[-1])      
            # COM0s = np.array(COM0s)
            COM0s = COM0s.reshape(-1, COM0s.shape[-2], COM0s.shape[-1])  

            # result save
            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae3, mse3, rmse3, mape3, mspe3 = metric(outputs_MIXs, batch_ys)
            print('mse:{}, mae:{}'.format(mse3, mae3))

            np.save(folder_path+'metrics_for_Normalized_Data_Finetune.npy', np.array([ mae3, mse3, rmse3, mape3, mspe3]))
            np.save(folder_path+'Normalized_pred_Finetune.npy', outputs_MIXs)
            np.save(folder_path+'Normalized_true_Finetune.npy', batch_ys)
            np.save(folder_path+'pred_WeightedSum_Finetune.npy', PFTs)
            np.save(folder_path+'pred_COM_Finetune.npy', COMs)
            np.save(folder_path+'pred_COM0_Finetune.npy', COM0s)

    

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
        
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path+'/'+'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #         print('load model from:', best_model_path)

    #     self.model.eval()
        
    #     preds = []
        
    #     for i, (batch_x,batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g) in enumerate(pred_loader):
    #         pred, true = self._process_one_batch(
    #             pred_data, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g)
    #         preds.append(pred.detach().cpu().numpy())

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
    #     # result save
    #     folder_path = './results/' + setting +'/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
        
    #     np.save(folder_path+'real_prediction.npy', preds)
        
    #     return


    def _process_one_batch_3branch_pft_pure(self, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_gs, stage='pure', type='BL'):
        batch_x = batch_x.float().to(self.device) # [32, 480, 136]
        batch_y = batch_y.float() # [32, 481, 10]
        # batch_y = batch_y[:,batch_y.shape[1]-self.args.label_len - self.args.pred_len:,:] # Shuo: Adjust label length  # [32, 13, 10]
        batch_pft_bl = batch_pft_bl.float() # [32, 480, 1]
        batch_pft_nl = batch_pft_nl.float() # [32, 480, 1]
        batch_pft_gs = batch_pft_gs.float() # [32, 480, 1]
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 481, 10]

        dec_inp_MIX = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 481, 10]
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs,'pure',type)[0]
                else:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs,'pure',type)
        else:
            if self.args.output_attention:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs,'pure',type)[0]
            else:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs,'pure',type)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        
        batch_y = batch_y[:,:,f_dim:]
        outputs_BL = outputs_BL[:,:,f_dim:]

        outputPFT = outs3branchPft[0] # [32,480,3]
        outputCOM = outs3branchPft[1] # [32,480,1]
        
        return outputs_BL, outputs_NL, outputs_GS, outputs_MIX, batch_y, outputPFT, outputCOM
    


    def _process_one_batch_3branch_pft(self, batch_ws, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_gs, stage='mix', type='mix'):
        batch_ws = batch_ws.float().to(self.device) # [32, 336, 10]
        batch_x = batch_x.float().to(self.device) # [32, 480, 136]
        batch_y = batch_y.float() # [32, 481, 10]
        # batch_y = batch_y[:,batch_y.shape[1]-self.args.label_len - self.args.pred_len:,:] # Shuo: Adjust label length  # [32, 13, 10]
        batch_pft_bl = batch_pft_bl.float() # [32, 480, 1]
        batch_pft_nl = batch_pft_nl.float() # [32, 480, 1]
        batch_pft_gs = batch_pft_gs.float() # [32, 480, 1]
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 481, 10]

        dec_inp_MIX = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 481, 10]
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,stage,type)[0]
                else:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,stage,type)
        else:
            if self.args.output_attention:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,stage,type)[0]
            else:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws,stage,type)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        
        if stage == 'mix':
            batch_y = batch_y[:,:,f_dim:]
            outputs_BL = outputs_BL[:,:,f_dim:]
            outputs_NL = outputs_NL[:,:,f_dim:]
            outputs_GS = outputs_GS[:,:,f_dim:]
            outputs_MIX = outputs_MIX[:,:,f_dim:]

            outputPFT = outs3branchPft[0] # [32,480,3]
            outputCOM = outs3branchPft[1]
            outputCOM0 = outs3branchPft[2]# [32,480,1]
        elif stage == 'ageindep':
            batch_y = batch_y[:,:,f_dim:]
            outputs_MIX = outputs_MIX[:,:,f_dim:]

            outputPFT = outs3branchPft[0] # [32,480,3]
            outputCOM = outs3branchPft[1]
            outputCOM0 = outs3branchPft[2]# [32,480,1]

            outputs_BL = None
            outputs_NL = None
            outputs_GS = None
      

        return outputs_BL, outputs_NL, outputs_GS, outputs_MIX, batch_y, outputPFT, outputCOM, outputCOM0

    def _process_one_batch(self, batch_ws, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_gs, stage, type):
        batch_ws = batch_ws.float().to(self.device) # [32, 336, 10]
        batch_x = batch_x.float().to(self.device) # [32, 336, 136]
        batch_y = batch_y.float() # [32, 337, 10]
        batch_pft_bl = batch_pft_bl.float().to(self.device) # [32, 480, 1]
        batch_pft_nl = batch_pft_nl.float().to(self.device) # [32, 480, 1]
        batch_pft_gs = batch_pft_gs.float().to(self.device) # [32, 480, 1]
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp_MIX = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device) # [32, 337, 10]


        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws, stage,type)[0]
                else:
                    outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws, stage,type)
        else:
            if self.args.output_attention:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws, stage,type)[0]
            else:
                outputs_BL, outputs_NL, outputs_GS, outputs_MIX, outs3branchPft = self.model(batch_x, dec_inp_MIX, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws, stage,type)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        batch_y = batch_y[:,:,f_dim:]

        if stage == 'ageindep':
            # outputs_MIX = outputs_MIX[:,:,4]
            outputs_MIX = outputs_MIX[:,:,f_dim:]           
            outputs = outputs_MIX

        return outputs, batch_y 


class Exp_ed_allage_pft_prediction_brief(Exp_Basic):
    def __init__(self, args):
        super(Exp_ed_allage_pft_prediction_brief, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'OneBranch_ED_ALLAGE_PFT_Prediction': OneBranch_ED_ALLAGE_PFT_Prediction,
        }
        if self.args.model=='OneBranch_ED_ALLAGE_PFT_Prediction':
            e_layers = self.args.e_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.args.age_embed_dim,
                self.device
            ).float()
            
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, stage='step1'):
        args = self.args

        data_dict = {
            'ED': Dataset_ED_ALLAGE_PFT_Prediction,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            stat_path = args.stat_path,
            pft_path = args.pft_path,
            asi = args.asi,
            aei = args.aei,
            add_noise = args.add_noise,
            noise_std = args.noise_std,
            flag=flag,
            stage=stage,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean
        
    def vali(self, vali_data, vali_loader, criterion, stage):
        self.model.eval()
        total_loss = []
        if stage == 'step1':
            with torch.no_grad():  
                for i, (batch_x,batch_y) in enumerate(vali_loader):
                    pred, true = self._process_one_batch(
                        vali_data, batch_x, batch_y, stage='step1')
                    
                    loss = criterion(pred.detach().cpu(), true.detach().cpu())
                    total_loss.append(loss)
        elif stage == 'step2':
            with torch.no_grad():  
                for i, (batch_x,batch_y,data_y_ESACCI,data_aw) in enumerate(vali_loader):
                    pred, true = self._process_one_batch(
                        vali_data, batch_x, [batch_y,data_y_ESACCI,data_aw], stage='step2')
                    
                    loss = criterion(pred.detach().cpu(), true.detach().cpu())
                    total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting, stage):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        train_data, train_loader = self._get_data(flag = 'train', stage = stage)
        vali_data, vali_loader = self._get_data(flag = 'val', stage = stage)
        test_data, test_loader = self._get_data(flag = 'test', stage = stage)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)

        if stage == 'step2':
            early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)

            # load
            state_dict_full = torch.load(os.path.join(path, 'checkpoint.pth'))
            self.model.load_state_dict(state_dict_full)

            for name, param in self.model.named_parameters():
                if 'encoder' in name or 'enc_embedding' in name:
                    param.requires_grad = False

            print("Freezing encoder-related layers (enc_embedding, encoder)")

            model_optim = self._select_optimizer()

        elif stage == 'step1':
            early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)
            model_optim = self._select_optimizer()

        
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if stage == 'step1':
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []
                
                self.model.train()
                epoch_time = time.time()
                for i, (batch_x,batch_y) in enumerate(train_loader):
                    iter_count += 1
                    
                    model_optim.zero_grad()
                    pred, true = self._process_one_batch(
                        train_data, batch_x, batch_y, stage='step1')

                    loss = criterion(pred, true)
                    train_loss.append(loss.item())
                    
                    if (i+1) % 100==0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time()-time_now)/iter_count
                        left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward() # Backward Pass, backpropagation
                        model_optim.step() # Optimization Step

                print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion, stage)
                test_loss = self.vali(test_data, test_loader, criterion, stage)
                
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path+'/'+'checkpoint.pth')
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch+1, self.args)
                
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        elif stage == 'step2':
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []
                
                self.model.train()
                epoch_time = time.time()
                for i, (batch_x,batch_y,data_y_ESACCI,data_aw) in enumerate(train_loader):
                    iter_count += 1
                    
                    model_optim.zero_grad()
                    pred, true = self._process_one_batch(
                        train_data, batch_x, [batch_y,data_y_ESACCI,data_aw], stage='step2')

                    loss = criterion(pred, true)
                    train_loss.append(loss.item())
                    
                    if (i+1) % 100==0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time()-time_now)/iter_count
                        left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward() # Backward Pass, backpropagation
                        model_optim.step() # Optimization Step

                print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion, stage)
                test_loss = self.vali(test_data, test_loader, criterion, stage)
                
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path+'/'+'checkpoint_step2.pth')
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch+1, self.args)
                
            best_model_path = path+'/'+'checkpoint_step2.pth'
            self.model.load_state_dict(torch.load(best_model_path))


        return self.model

    
    def test(self, setting, stage):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))

        test_data, test_loader = self._get_data(flag='test', stage = stage)

        if stage == 'step1':       
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            print('load model from:', best_model_path)
            
            self.model.eval()
            
            preds = []
            trues = []
            
            for i, (batch_x,batch_y) in enumerate(test_loader):
                
                pred, true = self._process_one_batch_3branch_pft(test_data, batch_x, batch_y, stage='step1')
                            
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

                
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-3], preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-3], trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))

            np.save(folder_path+'metrics_6_step1.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path+'pred_6_step1.npy', preds)
            np.save(folder_path+'true_6_step1.npy', trues)

        elif stage == 'step2':       
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint_step2.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            print('load model from:', best_model_path)
            
            self.model.eval()
            
            preds = []
            trues = []
            we_preds = []
            ESACCIs = []
            aws = []

            for i, (batch_x,batch_y,data_y_ESACCI,data_aw) in enumerate(test_loader):                
                pred, true = self._process_one_batch_3branch_pft(test_data, batch_x, [batch_y,data_y_ESACCI,data_aw], stage='step2')
                we_pred = pred[0]
                pred = pred[1]
                ESACCI = true[1]
                aw = true[2]
                true = true[0]

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                we_preds.append(we_pred.detach().cpu().numpy())
                ESACCIs.append(ESACCI.detach().cpu().numpy())
                aws.append(aw.detach().cpu().numpy())
                
            preds = np.array(preds)
            trues = np.array(trues)
            we_preds = np.array(we_preds)
            ESACCIs = np.array(ESACCIs)
            aws = np.array(aws)
            print('test shape:', preds.shape, trues.shape, we_preds.shape, ESACCIs.shape, aws.shape)
            preds = preds.reshape(-1, preds.shape[-3], preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-3], trues.shape[-2], trues.shape[-1])
            we_preds = we_preds.reshape(-1, we_preds.shape[-2], we_preds.shape[-1])
            ESACCIs = ESACCIs.reshape(-1, ESACCIs.shape[-2], ESACCIs.shape[-1])
            aws = aws.reshape(-1, aws.shape[-1])
            print('test shape:', preds.shape, trues.shape, we_preds.shape, ESACCIs.shape, aws.shape)

            # result save
            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))

            np.save(folder_path+'metrics_6.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path+'pred_6.npy', preds)
            np.save(folder_path+'true_6.npy', trues)
            np.save(folder_path+'aw_6.npy', aws)
            np.save(folder_path+'ESACCI_6.npy', ESACCIs)
            np.save(folder_path+'we_preds_6.npy', we_preds)
        
        return
    


    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
        
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path+'/'+'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #         print('load model from:', best_model_path)

    #     self.model.eval()
        
    #     preds = []
        
    #     for i, (batch_x,batch_y) in enumerate(pred_loader):
    #         pred, true = self._process_one_batch(
    #             pred_data, batch_x, batch_y)
    #         preds.append(pred.detach().cpu().numpy())

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
    #     # result save
    #     folder_path = './results/' + setting +'/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
        
    #     np.save(folder_path+'real_prediction.npy', preds)
        
    #     return

    def _process_one_batch_3branch_pft(self, dataset_object, batch_x, batch_y, stage):
        if stage == 'step1':
            batch_x = batch_x.float().to(self.device) # [8, 28, 136*12]
            batch_y = batch_y.float() # [8, 18, 29, 3]
            if self.args.padding==0:
                dec_inp = torch.zeros([batch_y.shape[0], batch_y.shape[1], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp = torch.ones([batch_y.shape[0], batch_y.shape[1], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:,:,:self.args.label_len,:], dec_inp], dim=2).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp, stage)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp, stage)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, dec_inp, stage)[0]
                else:
                    outputs = self.model(batch_x, dec_inp, stage)

            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,:,-self.args.pred_len:,f_dim:].to(self.device)
            
            # only output target dimensions
            batch_y = batch_y
            outputs = outputs
            return outputs, batch_y

        elif stage == 'step2':
            batch_x = batch_x.float().to(self.device) # [8, 28, 136*12]
            data_y_ESACCI = batch_y[1].float() # [8, 29, 3]
            data_aw = batch_y[2].float() # (8, 18)
            batch_y = batch_y[0].float() # [8, 18, 29, 3]
            # batch_y = batch_y[:,batch_y.shape[1]-self.args.label_len - self.args.pred_len:,:] # Shuo: Adjust label length  # [32, 13, 10]
            # decoder input
            if self.args.padding==0:
                dec_inp = torch.zeros([batch_y.shape[0], batch_y.shape[1], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp = torch.ones([batch_y.shape[0], batch_y.shape[1], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:,:,:self.args.label_len,:], dec_inp], dim=2).float().to(self.device)
            
            # decoder input
            if self.args.padding==0:
                dec_inp1 = torch.zeros([data_y_ESACCI.shape[0], self.args.pred_len, data_y_ESACCI.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp1 = torch.ones([data_y_ESACCI.shape[0], self.args.pred_len, data_y_ESACCI.shape[-1]]).float()
            dec_inp1 = torch.cat([data_y_ESACCI[:,:self.args.label_len,:], dec_inp1], dim=1).float().to(self.device) # [8, 29, 3]
            
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, [dec_inp,dec_inp1,data_aw], stage)[0]
                    else:
                        outputs = self.model(batch_x, [dec_inp,dec_inp1,data_aw], stage)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, [dec_inp,dec_inp1,data_aw], stage)[0]
                else:
                    outputs = self.model(batch_x, [dec_inp,dec_inp1,data_aw], stage)

            weighted_out_all = outputs[0]
            out_all = outputs[1]
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,:,-self.args.pred_len:,f_dim:].to(self.device)
            data_y_ESACCI = data_y_ESACCI[:,-self.args.pred_len:,f_dim:].to(self.device)
            

            return [weighted_out_all,out_all], [batch_y,data_y_ESACCI,data_aw]

    def _process_one_batch(self, dataset_object, batch_x, batch_y, stage):
        if stage == 'step1':
            batch_x = batch_x.float().to(self.device) # [8, 28, 1632]
            batch_y = batch_y.float() # [8, 18, 29, 3]
            # decoder input
            if self.args.padding==0:
                dec_inp = torch.zeros([batch_y.shape[0], batch_y.shape[1], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp = torch.ones([batch_y.shape[0], batch_y.shape[1], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:,:,:self.args.label_len,:], dec_inp], dim=2).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp, stage)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp, stage)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, dec_inp, stage)[0]
                else:
                    outputs = self.model(batch_x, dec_inp, stage) # [18,8,28,3]
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,:,-self.args.pred_len:,f_dim:].to(self.device)
            
            # only output target dimensions
            batch_y = batch_y
            outputs = outputs

        elif stage == 'step2':        
            batch_x = batch_x.float().to(self.device) # [8, 28, 1632]           
            data_y_ESACCI = batch_y[1].float() # [8, 29, 3]
            data_aw = batch_y[2].float() # (8, 18)
            batch_y = batch_y[0].float() # [8, 18, 29, 3]

            # decoder input
            if self.args.padding==0:
                dec_inp = torch.zeros([batch_y.shape[0], batch_y.shape[1], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp = torch.ones([batch_y.shape[0], batch_y.shape[1], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:,:,:self.args.label_len,:], dec_inp], dim=2).float().to(self.device)

            # decoder input
            if self.args.padding==0:
                dec_inp1 = torch.zeros([data_y_ESACCI.shape[0], self.args.pred_len, data_y_ESACCI.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp1 = torch.ones([data_y_ESACCI.shape[0], self.args.pred_len, data_y_ESACCI.shape[-1]]).float()
            dec_inp1 = torch.cat([data_y_ESACCI[:,:self.args.label_len,:], dec_inp1], dim=1).float().to(self.device) # [8, 29, 3]

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, [dec_inp,dec_inp1,data_aw], stage)[0]
                    else:
                        outputs = self.model(batch_x, [dec_inp,dec_inp1,data_aw], stage)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, [dec_inp,dec_inp1,data_aw], stage)[0]
                else:
                    outputs = self.model(batch_x, [dec_inp,dec_inp1,data_aw], stage) # [18,8,28,3]


            f_dim = -1 if self.args.features=='MS' else 0
            data_y_ESACCI = data_y_ESACCI[:,-self.args.pred_len:,f_dim:].to(self.device)
            
            # only output target dimensions
            batch_y = data_y_ESACCI
            outputs = outputs[0] # [8, 28, 3]

        return outputs, batch_y


class Exp_ED_addPure_finetune_mixdata_insitu_ShareStructure(Exp_Basic):
    def __init__(self, args):
        super(Exp_ED_addPure_finetune_mixdata_insitu_ShareStructure, self).__init__(args)

        self.model1 = self._build_model(args.model)
        self.model2 = self._build_model(args.model2, model2=True)
    
    def _build_model(self, model_name=None, model2=False):
        if model_name is None:
            model_name = self.args.model
        model_dict = {
            'MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure': MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure,
            'OneBranch_ED_ALLAGE_PFT_Prediction': OneBranch_ED_ALLAGE_PFT_Prediction,
        }
        e_layers = self.args.e_layers

        if model2:
            enc_in = self.args.enc_in*12
            dec_in = 3
            c_out = 3
        else:
            enc_in = self.args.enc_in
            dec_in = self.args.dec_in
            c_out = self.args.c_out


        if model_name == 'OneBranch_ED_ALLAGE_PFT_Prediction':
            model = model_dict[model_name](
                enc_in,
                dec_in, 
                c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers,  # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                age_embed_dim=self.args.age_embed_dim, 
                device=self.device                     
            ).float()
        else:

            model = model_dict[model_name](
                enc_in,
                dec_in, 
                c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers,  # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                device=self.device
            ).float()


        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ED': Dataset_AddPure_Extract4types_CombineWith_PFT_Prediction,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            stat_path = args.stat_path,
            pft_path = args.pft_path,
            asi = args.asi,
            aei = args.aei,
            add_noise = args.add_noise,
            noise_std = args.noise_std,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_optimizer_2(self):
        params = list(filter(lambda p: p.requires_grad, self.model1.parameters())) + \
                list(filter(lambda p: p.requires_grad, self.model2.parameters()))
        return torch.optim.Adam(params, lr=self.args.learning_rate)

    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def _weight_pure_ages(self, data_x, data_y, AgeWeight, model_branch1, model_branch2, model_branch3, device):
        N, age, T, Ff = data_x.shape
        N1, age1, T1, Vv = data_y.shape
        data_x_flat = data_x.reshape(-1, T, Ff)
        data_y_flat = data_y.reshape(-1, T+1, Vv)

        data_x_tensor = torch.from_numpy(data_x_flat).float().to(device)
        data_y_tensor0 = torch.from_numpy(data_y_flat).float().to(device)

        if self.args.padding == 0:
            dec_inp = torch.zeros([data_y_tensor0.shape[0], self.args.pred_len, data_y_tensor0.shape[-1]], device=device)
        elif self.args.padding == 1:
            dec_inp = torch.ones([data_y_tensor0.shape[0], self.args.pred_len, data_y_tensor0.shape[-1]], device=device)

        data_y_tensor = torch.cat([data_y_tensor0[:, :self.args.label_len, :], dec_inp], dim=1)

        model_branch1 = model_branch1.to(device)
        model_branch2 = model_branch2.to(device)
        model_branch3 = model_branch3.to(device)

        batch_size = 32

        outputs_BL = []
        outputs_NL = []
        outputs_GS = []

        for i in range(0, data_x_tensor.size(0), batch_size):
            batch_x = data_x_tensor[i:i+batch_size]
            batch_y = data_y_tensor[i:i+batch_size]

            with torch.no_grad():
                out_BL = model_branch1(batch_x, batch_y, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
                out_NL = model_branch2(batch_x, batch_y, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
                out_GS = model_branch3(batch_x, batch_y, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)

            outputs_BL.append(out_BL)
            outputs_NL.append(out_NL)
            outputs_GS.append(out_GS)

        out_BL = torch.cat(outputs_BL, dim=0)  # (N*18, T, D)
        out_NL = torch.cat(outputs_NL, dim=0)
        out_GS = torch.cat(outputs_GS, dim=0)

        D = out_BL.shape[-1]
        out_BL_mix = out_BL.reshape(N, age, T, D)
        out_NL_mix = out_NL.reshape(N, age, T, D)
        out_GS_mix = out_GS.reshape(N, age, T, D)


        out_denorm = self.__inverse_norm_data__(out_BL_mix, self.y_mean, self.y_std)
        mask = torch.ones(out_denorm.shape[-1], device=out_BL_mix.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        out_BL = out_denorm_relu

        out_denorm = self.__inverse_norm_data__(out_NL_mix, self.y_mean, self.y_std)
        mask = torch.ones(out_denorm.shape[-1], device=out_NL_mix.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        out_NL = out_denorm_relu

        out_denorm = self.__inverse_norm_data__(out_GS_mix, self.y_mean, self.y_std)
        mask = torch.ones(out_denorm.shape[-1], device=out_GS_mix.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        out_GS = out_denorm_relu
        
        del out_BL_mix, out_NL_mix, out_GS_mix
        torch.cuda.empty_cache()

        f_dim = -1 if self.args.features=='MS' else 0
        out_BL = out_BL[:,:,-self.args.pred_len:,f_dim:].to(device)
        out_BL = out_BL[:,:,:,f_dim:]
        out_NL = out_NL[:,:,-self.args.pred_len:,f_dim:].to(device)
        out_NL = out_NL[:,:,:,f_dim:]
        out_GS = out_GS[:,:,-self.args.pred_len:,f_dim:].to(device)
        out_GS = out_GS[:,:,:,f_dim:]

        return out_BL, out_NL, out_GS



    def vali(self, vali_data, vali_loader, criterion):
        self.model1.eval()
        self.model2.eval()
        total_loss = []

        batch_size = self.args.batch_size

        out_BL, out_NL, out_GS = self._weight_pure_ages(
            vali_data.data_x, # (N, age, 336, 136)
            vali_data.data_y, # (N, age, 337, 10)
            vali_data.AgeWeight, # (N, age)
            self.model1.branch1, self.model1.branch2, self.model1.branch3,
            self.device
        ) # (N, 336, 10)

        data_x = vali_data.data_x[:, 0, :, :]      # (N, 336, 136)
        AgeWeight = vali_data.AgeWeight            # (N, age)

        AgeWeight_tensor = torch.from_numpy(vali_data.AgeWeight).float()   # (N, age)
        data_y_tensor = torch.from_numpy(vali_data.data_y).float()         # (N, age, 337, 10)

        data_y_agesum = (
            data_y_tensor * AgeWeight_tensor.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)                                                       # (N, 337, 10)

        data_x_12months = vali_data.data_x_12months     # (N, 28, 136*12)
        data_y_pft = vali_data.data_y_pft               # (N, 29, 3)
        pft_MIX_VG = vali_data.pft_MIX_VG               # (N, 29, 1)
        data_y_ed_pft = vali_data.data_y_ed_pft         # (N, 18, 29, 3)
        InSitu = vali_data.InSitu                       # (N, 336, 1)

        dataset_ageindep = TensorDataset(
            out_BL.float(),                    # (N, age, 336, 10)
            out_NL.float(),
            out_GS.float(),
            torch.from_numpy(data_x).float(),          # (N, 336, 136)
            data_y_agesum.float(),                      # (N, 337, 10)
            torch.from_numpy(data_x_12months).float(), # (N, 28, 136*12)
            torch.from_numpy(data_y_pft).float(),      # (N, 29, 3)
            torch.from_numpy(InSitu).float(),          # (N, 336, 1)
            torch.from_numpy(AgeWeight).float(),       # (N, age)
            torch.from_numpy(pft_MIX_VG).float()       # (N, 29, 1)
        )


        loader_val = DataLoader(dataset_ageindep, batch_size=self.args.batch_size, shuffle=True)
        with torch.no_grad():  
            for i, (batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp) in enumerate(loader_val): 
                pred_MIX, true_MIX, COM, COM0, pred_PFT, true_PFT, batch_ED = self._process_one_batch_3branch_pft(batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp)          
                pred_MIX = pred_MIX.reshape(-1)
                true_MIX = true_MIX.reshape(-1)
                nanmask = torch.isnan(true_MIX)
                pred_MIX[nanmask]=0
                true_MIX[nanmask]=0
                loss_GPP = criterion(pred_MIX, true_MIX)
                loss_PFT = criterion(pred_PFT, true_PFT)

                loss = loss_GPP 

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model1.train()
        self.model2.train()
        return total_loss
    
    
    def train(self, setting):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        batch_size = self.args.batch_size

        setting0 = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_2step_{}'.format(self.args.model, self.args.data, 
                    self.args.seq_len, self.args.label_len, self.args.pred_len,
                    self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, self.args.ii)


        path = os.path.join(self.args.checkpoints, setting0)  # _EbdEcdEbd        
     
        best_model_path1 = os.path.join(path, 'checkpoint_finetune.pth')

        self.model1.load_state_dict(torch.load(best_model_path1))
        print('load model from:', best_model_path1) 

        for name, param in self.model1.named_parameters():
            if any(x in name for x in ['branch1', 'branch2', 'branch3']):
                param.requires_grad = False
        print("Freezing branch 1, 2, 3")
        for name, param in self.model1.named_parameters():    
            if 'branch0' in name:
                param.requires_grad = True

        print("No Freezing competition model")

        setting00 = 'OneBranch_ED_ALLAGE_PFT_Prediction_{}_sl28_ll1_pl28_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_PFT_{}'.format(self.args.data, 
                    self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, self.args.ii)

        path = os.path.join(self.args.checkpoints2, setting00)          
        best_model_path2 = os.path.join(path, 'checkpoint_step2.pth')
        self.model2.load_state_dict(torch.load(best_model_path2))

        for name, param in self.model2.named_parameters():
            param.requires_grad = True

        print("No Freezing pft model")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # Set model1 stats
        if hasattr(self.model1, 'module') and hasattr(self.model1.module, 'set_stats'):
            self.model1.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model1.set_stats(self.y_mean, self.y_std)

        time_now = time.time()
        train_steps = len(train_loader)
        # early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)
        early_stopping1 = EarlyStopping1(patience=self.args.patience, verbose=True)
        early_stopping2 = EarlyStopping1(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer_2()
        criterion = self._select_criterion()  

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        out_BL, out_NL, out_GS = self._weight_pure_ages(
            train_data.data_x, # (N, age, 336, 136)
            train_data.data_y, # (N, age, 337, 10)
            train_data.AgeWeight, # (N, age)
            self.model1.branch1, self.model1.branch2, self.model1.branch3,
            self.device
        ) # (N, 336, 10)


        data_x = train_data.data_x[:, 0, :, :]      # (N, 336, 136)
        AgeWeight = train_data.AgeWeight            # (N, age)
        AgeWeight_tensor = torch.from_numpy(train_data.AgeWeight).float()   # (N, age)
        data_y_tensor = torch.from_numpy(train_data.data_y).float()         # (N, age, 337, 10)

        data_y_agesum = (
            data_y_tensor * AgeWeight_tensor.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)                                                       # (N, 337, 10)

        data_x_12months = train_data.data_x_12months     # (N, 28, 136*12)
        data_y_pft = train_data.data_y_pft               # (N, 29, 3)
        pft_MIX_VG = train_data.pft_MIX_VG               # (N, 29, 1)
        data_y_ed_pft = train_data.data_y_ed_pft         # (N, 18, 29, 3)
        InSitu = train_data.InSitu                       # (N, 336, 1)


        dataset_ageindep = TensorDataset(
            out_BL.float(),                       # (N, 18, 336, 10)
            out_NL.float(),
            out_GS.float(),
            torch.from_numpy(data_x).float(),     # (N, 336, 136)
            data_y_agesum.float(),                
            torch.from_numpy(data_x_12months).float(),  # (N, 28, 136*12)
            torch.from_numpy(data_y_pft).float(),       # (N, 29, 3)
            torch.from_numpy(InSitu).float(),           # (N, 336, 1)
            torch.from_numpy(AgeWeight).float(),        # (N, age)
            torch.from_numpy(pft_MIX_VG).float()        # (N, 29, 1)
        )


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model1.train()
            self.model2.train()
            epoch_time = time.time()

            loader_ageindep = DataLoader(dataset_ageindep, batch_size=batch_size, shuffle=True)
            
            for i, (batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp) in enumerate(loader_ageindep):
                iter_count += 1
                model_optim.zero_grad()

                pred_MIX, true_MIX, COM, COM0, pred_PFT, true_PFT, batch_ED = self._process_one_batch_3branch_pft(batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp)                                               
              
                pred_MIX = pred_MIX.reshape(-1)
                true_MIX = true_MIX.reshape(-1)

                nanmask = torch.isnan(true_MIX) | torch.isnan(pred_MIX)
                pred_MIX[nanmask]=0
                true_MIX[nanmask]=0
                loss_GPP = criterion(pred_MIX, true_MIX)

                loss_PFT = criterion(pred_PFT, true_PFT)

                loss = loss_GPP
                        
                train_loss.append(loss.item())
                                
                if (i+1) % 1==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | loss_PFT: {3:.7f}".format(i + 1, epoch + 1, loss_GPP.item(), loss_PFT.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)

                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping1(vali_loss, self.model1, os.path.join(path, 'checkpoint_step1.pth'))
            early_stopping2(vali_loss, self.model2, os.path.join(path, 'checkpoint_step2.pth'))
            

            if early_stopping1.early_stop or early_stopping2.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint_step1.pth')
        self.model1.load_state_dict(torch.load(best_model_path))

        best_model_path2 = os.path.join(path, 'checkpoint_step2.pth')
        self.model2.load_state_dict(torch.load(best_model_path2))
     
        return self.model1

    
    def test(self, setting):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        test_data, test_loader = self._get_data(flag='test')

        batch_size = self.args.batch_size

        path = os.path.join(self.args.checkpoints, setting)          
        best_model_path1 = os.path.join(path, 'checkpoint_step1.pth')
        self.model1.load_state_dict(torch.load(best_model_path1))
        print('load model from:', best_model_path1) 


        path = os.path.join(self.args.checkpoints, setting)          
        best_model_path2 = os.path.join(path, 'checkpoint_step2.pth')
        self.model2.load_state_dict(torch.load(best_model_path2))
       
        self.model1.eval()
        self.model2.eval()

        # # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        # if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
        #     self.model.module.set_stats(self.y_mean, self.y_std)
        # else:
        #     self.model.set_stats(self.y_mean, self.y_std)


        # === Compute pure model outputs ===
        out_BL, out_NL, out_GS = self._weight_pure_ages(
            test_data.data_x,
            test_data.data_y,
            test_data.AgeWeight,
            self.model1.branch1, self.model1.branch2, self.model1.branch3,
            self.device
        )  # (N, 336, 10)



        data_x = test_data.data_x[:, 0, :, :]        # (N, 336, 136)
        AgeWeight = test_data.AgeWeight              # (N, age)

        AgeWeight_tensor = torch.from_numpy(test_data.AgeWeight).float()   # (N, age)
        data_y_tensor = torch.from_numpy(test_data.data_y).float()         # (N, age, 337, 10)
        data_y_agesum = (data_y_tensor * AgeWeight_tensor.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (N, 337, 10)

        data_x_12months = test_data.data_x_12months   # (N, 28, 136*12)
        data_y_pft = test_data.data_y_pft             # (N, 29, 3)
        pft_MIX_VG = test_data.pft_MIX_VG             # (N, 29, 1)
        data_y_ed_pft = test_data.data_y_ed_pft       # (N, 18, 29, 3)
        InSitu = test_data.InSitu                     # (N, 336, 1)

        # === Build dataset and dataloader ===
        dataset_val = TensorDataset(
            out_BL.float(),
            out_NL.float(),
            out_GS.float(),
            torch.from_numpy(data_x).float(),
            data_y_agesum.float(),                     
            torch.from_numpy(data_x_12months).float(),
            torch.from_numpy(data_y_pft).float(),
            torch.from_numpy(InSitu).float(),
            torch.from_numpy(AgeWeight).float(),            # (N, age)
            torch.from_numpy(pft_MIX_VG).float()            # (N, 29, 1)
        )

        loader_val = DataLoader(dataset_val, batch_size=self.args.batch_size, shuffle=False)


        outputs_MIXs = []
        batch_ys = []
        COMs = []
        COM0s = []
        outputs_PFTs = []
        batch_PFTs = []
        batch_EDs = []

        for i, (batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp) in enumerate(loader_val):
            outputs_MIX, batch_y, COM, COM0, pred_PFT, true_PFT, batch_ED = self._process_one_batch_3branch_pft(batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp)                   

            # batch_y
            _y = batch_y.detach().cpu().numpy()
            if _y.ndim == 1:
                _y = _y[None, :]
            batch_ys.append(_y)
              
            outputs_MIXs.append(outputs_MIX.detach().cpu().numpy())
            outputs_PFTs.append(pred_PFT.detach().cpu().numpy())
            batch_PFTs.append(true_PFT.detach().cpu().numpy())
            COMs.append(COM.detach().cpu().numpy())
            COM0s.append(COM0.detach().cpu().numpy())
            batch_EDs.append(batch_ED.detach().cpu().numpy())
          
        outputs_MIXs = np.concatenate(outputs_MIXs, axis=0)
        batch_ys = np.concatenate(batch_ys, axis=0)
        print('test shape:', outputs_MIXs.shape, batch_ys.shape)
        outputs_PFTs = np.concatenate(outputs_PFTs, axis=0)
        batch_PFTs = np.concatenate(batch_PFTs, axis=0)
        print('test shape:', outputs_PFTs.shape, batch_PFTs.shape)
        batch_EDs = np.concatenate(batch_EDs, axis=0)

        COMs = np.concatenate(COMs, axis=0)
        COMs = COMs[:,:,4] # .reshape(-1, COMs.shape[-2], COMs.shape[-1])      
        COM0s = np.concatenate(COM0s, axis=0)
        COM0s = COM0s[:,:,4] # .reshape(-1, COM0s.shape[-2], COM0s.shape[-1])  

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        outputs_MIXs = self.__inverse_norm_data__(outputs_MIXs, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
        batch_ys = self.__inverse_norm_data__(batch_ys, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
        metrics = []
        # channel 0
        pred = outputs_MIXs[..., 0].reshape(-1)
        true = batch_ys[..., 0].reshape(-1)
        mask = ~np.isnan(true)
        mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        # print('channel 0 -> mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        print('channel 0 -> mae:{}, rmse:{}'.format(mae, rmse))
        metrics.append([mae, mse, rmse, mape, mspe])
        # channel 1
        pred = outputs_MIXs[..., 1].reshape(-1)
        true = batch_ys[..., 1].reshape(-1)
        mask = ~np.isnan(true)
        mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        # print('channel 1 -> mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        print('channel 1 -> mae:{}, rmse:{}'.format(mae, rmse))
        metrics.append([mae, mse, rmse, mape, mspe])
        # channel 2
        pred = outputs_MIXs[..., 2].reshape(-1)
        true = batch_ys[..., 2].reshape(-1)
        mask = ~np.isnan(true)
        mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        # print('channel 2 -> mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        metrics.append([mae, mse, rmse, mape, mspe])       
        print('channel 2 -> mae:{}, rmse:{}'.format(mae, rmse))

        np.save(folder_path+'metrics_inverse_norm_.npy', np.array(metrics)) 
        np.save(folder_path+'Normalized_pred_MIX.npy', outputs_MIXs)
        np.save(folder_path+'Normalized_true_MIX.npy', batch_ys)
        np.save(folder_path+'Normalized_pred_PFT.npy', outputs_PFTs)
        np.save(folder_path+'Normalized_true_PFT.npy', batch_PFTs)
        np.save(folder_path+'Normalized_ED.npy', batch_EDs)
        np.save(folder_path+'pred_MIX_COM.npy', COMs)
        np.save(folder_path+'pred_MIX_COM0.npy', COM0s)

        return
    
    

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
        
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path+'/'+'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #         print('load model from:', best_model_path)

    #     self.model.eval()
        
    #     preds = []
        
    #     for i, (batch_x,batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g) in enumerate(pred_loader):
    #         pred, true = self._process_one_batch(
    #             pred_data, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g)
    #         preds.append(pred.detach().cpu().numpy())

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
    #     # result save
    #     folder_path = './results/' + setting +'/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
        
    #     np.save(folder_path+'real_prediction.npy', preds)
        
    #     return


    def _process_one_batch_3branch_pft(self, batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp):
        batch_BL = batch_BL.float().to(self.device) # [32, 18, 336, 10]
        batch_NL = batch_NL.float().to(self.device) # [32, 18, 336, 10]
        batch_GS = batch_GS.float().to(self.device) # [32, 18, 336, 10]        
        
        batch_x = batch_x.float().to(self.device) # [32, 336, 136]
        batch_y = batch_y.float().to(self.device) # [32, 337, 10]
        batch_x_12months = batch_x_12months.float().to(self.device) # [32, 28, 136*12]
        batch_y_pft = batch_y_pft.float().to(self.device) # [32, 29, 3]
        batch_InSitu = batch_InSitu.float().to(self.device)
        # .squeeze() # [32, 336,3]
        batch_vp = batch_vp.float().to(self.device) # [32, 29, 1]
        
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y_pft.shape[0], int(self.args.pred_len / 12), batch_y_pft.shape[-1]], device=self.device).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y_pft.shape[0], int(self.args.pred_len / 12), batch_y_pft.shape[-1]], device=self.device).float()
        dec_inp_PFT = torch.cat([batch_y_pft[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

        if self.args.use_gpu:
            self.model2 = self.model2.to(self.device)   
            self.model1 = self.model1.to(self.device)  

        # shape: [B, T, D] → repeat 18 times
        B = batch_x_12months.shape[0]
        batch_x_12months = batch_x_12months.repeat_interleave(18, dim=0)        # [B*18, T, D]
        age_ids = torch.arange(18).repeat(B).to(self.device)   # [B*18]
        age_emb = self.model2.age_embed(age_ids)                 # [B*18, age_dim]
        age_emb = age_emb.unsqueeze(1).expand(-1, batch_x_12months.shape[1], -1)  # [B*18, T, age_dim]
        batch_x_12months = torch.cat([batch_x_12months, age_emb], dim=-1)       # [B*18, T, D+age_dim]

        dec_inp_PFT = dec_inp_PFT.repeat_interleave(18, dim=0)  # [B, T, 3] to [B*18, T, 3]
        # x_dec = x_dec.reshape(B * 18, *x_dec.shape[2:])  # [B*18, T, 3]

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    out_all = self.model2.branch1(batch_x_12months, dec_inp_PFT, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)[0]       
                else:
                    out_all = self.model2.branch1(batch_x_12months, dec_inp_PFT, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)  
        else:
            if self.args.output_attention:
                out_all = self.model2.branch1(batch_x_12months, dec_inp_PFT, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)[0]   
            else:
                out_all = self.model2.branch1(batch_x_12months, dec_inp_PFT, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)       
        
        out_all = out_all.reshape(B, 18, *out_all.shape[1:])  # [B, 18, T, 3]

        out_all = F.sigmoid(out_all)         
        out_all_sum = out_all.sum(dim=3, keepdim=True) + 1e-12  
        out_all = out_all / out_all_sum

        batch_aw = batch_aw.to(self.device)
        data_aw_expanded = batch_aw.unsqueeze(-1).unsqueeze(-1)  # shape: [8, 18, 1, 1]
        weighted_out = out_all * data_aw_expanded  # [8, 18, 28, 3]
        out_PFT = weighted_out.sum(dim=1)  # shape: [8, 28, 3]
        
        pft_MIX_agesum = out_PFT[:,1:,:] 
        # pft_MIX_agesum shape: [N, 28, 3]
        pft_MIX_agesum = pft_MIX_agesum.unsqueeze(2).repeat(1, 1, 12, 1)  # [N, 28, 12, 3]
        pft_MIX_agesum2 = pft_MIX_agesum.clone()
        pft_MIX_agesum = pft_MIX_agesum.reshape(pft_MIX_agesum2.shape[0], pft_MIX_agesum2.shape[1]*pft_MIX_agesum2.shape[2],pft_MIX_agesum2.shape[3]) # (N, 336, 3)
                             
        x_enc_with_w = torch.cat([batch_x, pft_MIX_agesum], dim=2)  # [N, 336, 136+3]

        pft_VG_agesum = batch_vp[:,1:,:] 
        # pft_MIX_agesum shape: [N, 28, 1]
        pft_VG_agesum = pft_VG_agesum.unsqueeze(2).repeat(1, 1, 12, 1)  # [N, 28, 12, 1]
        pft_VG_agesum2 = pft_VG_agesum.clone()
        pft_VG_agesum = pft_VG_agesum.reshape(pft_VG_agesum2.shape[0], pft_VG_agesum2.shape[1]*pft_VG_agesum2.shape[2],pft_VG_agesum2.shape[3]) # (N, 336, 1)
               
        pft_MIX= out_all[:,:,1:,:].unsqueeze(3).repeat(1, 1, 1, 12, 1)  # [8, 18, 28, 3]  to [8, 18, 28, 12, 3] 
        pft_MIX2 = pft_MIX.clone()
        pft_MIX = pft_MIX.reshape(pft_MIX2.shape[0], pft_MIX2.shape[1], pft_MIX2.shape[2]*pft_MIX2.shape[3],pft_MIX2.shape[4]) # [8, 18, 336, 3] 


        # Weight each branch output by corresponding PFT weight
        weighted_out1 = batch_BL * pft_MIX[:,:,:,0].unsqueeze(-1) # [8, 18, 336, 10]
        weighted_out2 = batch_NL * pft_MIX[:,:,:,1].unsqueeze(-1)
        weighted_out3 = batch_GS * pft_MIX[:,:,:,2].unsqueeze(-1)
   
        fused_age = weighted_out1 + weighted_out2 + weighted_out3
        fused = (fused_age * data_aw_expanded).sum(dim=1)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], device=self.device).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], device=self.device).float()
        dec_inp_MIX = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).to(self.device) # [32, 481, 10]

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    out0_Com_mix0 = self.model1.branch0(x_enc_with_w, dec_inp_MIX, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)[0]
                else:
                    out0_Com_mix0 = self.model1.branch0(x_enc_with_w, dec_inp_MIX, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        else:
            if self.args.output_attention:
                out0_Com_mix0 = self.model1.branch0(x_enc_with_w, dec_inp_MIX, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)[0]
            else:
                out0_Com_mix0 = self.model1.branch0(x_enc_with_w, dec_inp_MIX, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)

        out0_Com_mix = out0_Com_mix0 # No.1
        # Final output: weighted sum of all branches plus the competition output
        final_out = fused + out0_Com_mix

        out_denorm = final_out
        mask = torch.ones(out_denorm.shape[-1], device=final_out.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        out_MIX = self.__norm_data__(out_denorm_relu, self.y_mean, self.y_std)
       
        outputs = out_MIX[:,:,[4, 9, 7]]

        return outputs, batch_InSitu, out0_Com_mix, out0_Com_mix0, out_PFT[:,1:,:], batch_y_pft[:, 1:, :], batch_y[:, 1:, :]


class Exp_addPure_finetune_mixdata_insitu_ShareStructure_imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_addPure_finetune_mixdata_insitu_ShareStructure_imputation, self).__init__(args)

        self.model1 = self._build_model(args.model)
        self.model2 = self._build_model(args.model2, model2=True)

        self.std_weight_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(self.device)

    
    def _build_model(self, model_name=None, model2=False):
        if model_name is None:
            model_name = self.args.model
        model_dict = {
            'MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure': MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure,
            'OneBranch_ED_ALLAGE_PFT_Prediction': OneBranch_ED_ALLAGE_PFT_Prediction,
        }
        e_layers = self.args.e_layers

        if model2:
            enc_in = self.args.enc_in*12
            dec_in = 3
            c_out = 3
        else:
            enc_in = self.args.enc_in
            dec_in = self.args.dec_in
            c_out = self.args.c_out

        if model_name == 'OneBranch_ED_ALLAGE_PFT_Prediction':
            model = model_dict[model_name](
                enc_in,
                dec_in, 
                c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers,  # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                age_embed_dim=self.args.age_embed_dim, 
                device=self.device                     
            ).float()
        else:

            model = model_dict[model_name](
                enc_in,
                dec_in, 
                c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers,  # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                device=self.device
            ).float()


        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ED': Dataset_AddPure_Extract4types_CombineWith_PFT_Prediction_imputation,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            stat_path = args.stat_path,
            pft_path = args.pft_path,
            asi = args.asi,
            aei = args.aei,
            add_noise = args.add_noise,
            noise_std = args.noise_std,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_optimizer_2(self):
        params = list(filter(lambda p: p.requires_grad, self.model1.parameters())) + \
                list(filter(lambda p: p.requires_grad, self.model2.parameters())) + \
                list(self.std_weight_net.parameters())
        return torch.optim.Adam(params, lr=self.args.learning_rate)

    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean


    def _weight_pure_ages(self, data_x, data_y, AgeWeight, model_branch1, model_branch2, model_branch3, device):
        N, age, T, Ff = data_x.shape
        N1, age1, T1, Vv = data_y.shape
        data_x_flat = data_x.reshape(-1, T, Ff)
        data_y_flat = data_y.reshape(-1, T+1, Vv)

        data_x_tensor = torch.from_numpy(data_x_flat).float().to(device)
        data_y_tensor0 = torch.from_numpy(data_y_flat).float().to(device)

        if self.args.padding == 0:
            dec_inp = torch.zeros([data_y_tensor0.shape[0], self.args.pred_len, data_y_tensor0.shape[-1]], device=device)
        elif self.args.padding == 1:
            dec_inp = torch.ones([data_y_tensor0.shape[0], self.args.pred_len, data_y_tensor0.shape[-1]], device=device)

        data_y_tensor = torch.cat([data_y_tensor0[:, :self.args.label_len, :], dec_inp], dim=1)

        model_branch1 = model_branch1.to(device)
        model_branch2 = model_branch2.to(device)
        model_branch3 = model_branch3.to(device)

        batch_size = 32

        outputs_BL = []
        outputs_NL = []
        outputs_GS = []

        for i in range(0, data_x_tensor.size(0), batch_size):
            batch_x = data_x_tensor[i:i+batch_size]
            batch_y = data_y_tensor[i:i+batch_size]

            with torch.no_grad():
                out_BL = model_branch1(batch_x, batch_y, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
                out_NL = model_branch2(batch_x, batch_y, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
                out_GS = model_branch3(batch_x, batch_y, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)

            outputs_BL.append(out_BL)
            outputs_NL.append(out_NL)
            outputs_GS.append(out_GS)

        out_BL = torch.cat(outputs_BL, dim=0)  # (N*18, T, D)
        out_NL = torch.cat(outputs_NL, dim=0)
        out_GS = torch.cat(outputs_GS, dim=0)

        D = out_BL.shape[-1]
        out_BL_mix = out_BL.reshape(N, age, T, D)
        out_NL_mix = out_NL.reshape(N, age, T, D)
        out_GS_mix = out_GS.reshape(N, age, T, D)

        out_denorm = self.__inverse_norm_data__(out_BL_mix, self.y_mean, self.y_std)
        mask = torch.ones(out_denorm.shape[-1], device=out_BL_mix.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        out_BL = out_denorm_relu

        out_denorm = self.__inverse_norm_data__(out_NL_mix, self.y_mean, self.y_std)
        mask = torch.ones(out_denorm.shape[-1], device=out_NL_mix.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        out_NL = out_denorm_relu

        out_denorm = self.__inverse_norm_data__(out_GS_mix, self.y_mean, self.y_std)
        mask = torch.ones(out_denorm.shape[-1], device=out_GS_mix.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        out_GS = out_denorm_relu
        
        del out_BL_mix, out_NL_mix, out_GS_mix
        torch.cuda.empty_cache()

        f_dim = -1 if self.args.features=='MS' else 0
        out_BL = out_BL[:,:,-self.args.pred_len:,f_dim:].to(device)
        out_BL = out_BL[:,:,:,f_dim:]
        out_NL = out_NL[:,:,-self.args.pred_len:,f_dim:].to(device)
        out_NL = out_NL[:,:,:,f_dim:]
        out_GS = out_GS[:,:,-self.args.pred_len:,f_dim:].to(device)
        out_GS = out_GS[:,:,:,f_dim:]

        return out_BL, out_NL, out_GS



    def vali(self, vali_data, vali_loader, criterion):
        self.model1.eval()
        self.model2.eval()
        total_loss = []

        batch_size = self.args.batch_size

        out_BL, out_NL, out_GS = self._weight_pure_ages(
            vali_data.data_x, # (N, age, 336, 136)
            vali_data.data_y, # (N, age, 337, 10)
            vali_data.AgeWeight, # (N, age)
            self.model1.branch1, self.model1.branch2, self.model1.branch3,
            self.device
        ) # (N, 336, 10)


        data_x = vali_data.data_x[:, 0, :, :]        # (N, 336, 136)
        AgeWeight = vali_data.AgeWeight              # (N, age)

        AgeWeight_tensor = torch.from_numpy(vali_data.AgeWeight).float()   # (N, age)
        data_y_tensor = torch.from_numpy(vali_data.data_y).float()         # (N, age, 337, 10)
        data_y_agesum = (
            data_y_tensor * AgeWeight_tensor.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)                                                      # (N, 337, 10)

        data_x_12months = vali_data.data_x_12months   # (N, 28, 136*12)
        data_y_pft = vali_data.data_y_pft             # (N, 29, 3)
        pft_MIX_VG = vali_data.pft_MIX_VG             # (N, 29, 1)
        data_y_ed_pft = vali_data.data_y_ed_pft       # (N, 18, 29, 3)
        InSitu = vali_data.InSitu                     # (N, 336, 1)
        STD = vali_data.STD                        

        # === Build dataset ===
        dataset_ageindep = TensorDataset(
            out_BL.float(),                           # (N, age, 336, 10)
            out_NL.float(),
            out_GS.float(),
            torch.from_numpy(data_x).float(),         # (N, 336, 136)
            data_y_agesum.float(),                    # (N, 337, 10)
            torch.from_numpy(data_x_12months).float(),# (N, 28, 136*12)
            torch.from_numpy(data_y_pft).float(),     # (N, 29, 3)
            torch.from_numpy(InSitu).float(),         # (N, 336, 1)
            torch.from_numpy(AgeWeight).float(),      # (N, age)
            torch.from_numpy(pft_MIX_VG).float(),     # (N, 29, 1)
            torch.from_numpy(STD).float()
        )


        loader_val = DataLoader(dataset_ageindep, batch_size=self.args.batch_size, shuffle=True)
        with torch.no_grad():  
            for i, (batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp, batch_std) in enumerate(loader_val):
                pred_MIX, true_MIX, COM, COM0, pred_PFT, true_PFT, batch_ED, fused = self._process_one_batch_3branch_pft(batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp)          
                pred_MIX = pred_MIX.reshape(-1)       # Model predictions
                true_MIX = true_MIX.reshape(-1)       # Raw target (not used directly here)
                insitu   = batch_InSitu.reshape(-1).to(pred_MIX.device)   # Complete target: observed + imputed
                std_vals = batch_std.reshape(-1).to(pred_MIX.device)      # Std available only at imputation points

                # Create masks
                mask_obs = torch.isnan(std_vals)      # Observed points (std is NaN)
                mask_imp = ~torch.isnan(std_vals)     # Imputation points (std available)

                # ===================== 1. Observed loss =====================
                # At observed points, compute standard MSE between prediction and in-situ data
                if mask_obs.any():
                    loss_obs = 10*criterion(pred_MIX[mask_obs], insitu[mask_obs])
                else:
                    loss_obs = 0.0

                # ===================== 2. Imputation loss =====================
                # At imputation points, use an auxiliary network (std_weight_net) 
                # to produce a weight based on [prediction, std]
                if mask_imp.any():
                    pred_vals = pred_MIX[mask_imp].unsqueeze(-1)   # (N_imp, 1)
                    std_in    = std_vals[mask_imp].unsqueeze(-1)   # (N_imp, 1)

                    # Concatenate prediction and std as input
                    inp = torch.cat([pred_vals, std_in], dim=-1)   # (N_imp, 2)

                    # Pass through small MLP to obtain weights in [0,1]
                    weights = self.std_weight_net(inp).squeeze(-1)
                    weights = torch.clamp(weights, 1e-3, 1.0)      # Avoid zero weight

                    # Compute weighted squared error
                    errors = (pred_MIX[mask_imp] - insitu[mask_imp])**2
                    loss_imp = torch.mean(weights * errors)
                else:
                    loss_imp = 0.0

                # ===================== 3. Final loss =====================
                # Combine observed and imputation losses
                loss_GPP = loss_obs + loss_imp

                loss = loss_GPP 
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model1.train()
        self.model2.train()
        self.std_weight_net.train()
        return total_loss
    
    
    def train(self, setting):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        batch_size = self.args.batch_size

        setting0 = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_2step_{}'.format(self.args.model, self.args.data, 
                    self.args.seq_len, self.args.label_len, self.args.pred_len,
                    self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, self.args.ii)


        path = os.path.join(self.args.checkpoints, setting0)  # _EbdEcdEbd     
        best_model_path1 = os.path.join(path, 'checkpoint_finetune.pth')

        self.model1.load_state_dict(torch.load(best_model_path1))
        print('load model from:', best_model_path1) 
        for name, param in self.model1.named_parameters():
            if any(x in name for x in ['branch1', 'branch2', 'branch3']):
                param.requires_grad = False
        print("Freezing branch 1, 2, 3")
        for name, param in self.model1.named_parameters():    
            if 'branch0' in name:
                param.requires_grad = True

        print("No Freezing competition model")

        setting00 = 'OneBranch_ED_ALLAGE_PFT_Prediction_{}_sl28_ll1_pl28_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_PFT_{}'.format(self.args.data, 
                    self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, self.args.ii)

        path = os.path.join(self.args.checkpoints2, setting00)          
        best_model_path2 = os.path.join(path, 'checkpoint_step2.pth')
        self.model2.load_state_dict(torch.load(best_model_path2))
        for name, param in self.model2.named_parameters():
            param.requires_grad = True
        print("No Freezing pft model")


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # Set model1 stats
        if hasattr(self.model1, 'module') and hasattr(self.model1.module, 'set_stats'):
            self.model1.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model1.set_stats(self.y_mean, self.y_std)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping1 = EarlyStopping1(patience=self.args.patience, verbose=True)
        early_stopping2 = EarlyStopping1(patience=self.args.patience, verbose=True)
        early_stopping_std = EarlyStopping1(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer_2()
        criterion = self._select_criterion()  

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        out_BL, out_NL, out_GS = self._weight_pure_ages(
            train_data.data_x, # (N, age, 336, 136)
            train_data.data_y, # (N, age, 337, 10)
            train_data.AgeWeight, # (N, age)
            self.model1.branch1, self.model1.branch2, self.model1.branch3,
            self.device
        ) # (N, 336, 10)

        data_x = train_data.data_x[:, 0, :, :]        # (N, 336, 136)
        AgeWeight = train_data.AgeWeight              # (N, age)

        AgeWeight_tensor = torch.from_numpy(train_data.AgeWeight).float()   # (N, age)
        data_y_tensor = torch.from_numpy(train_data.data_y).float()         # (N, age, 337, 10)
        data_y_agesum = (
            data_y_tensor * AgeWeight_tensor.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)                                                      # (N, 337, 10)

        data_x_12months = train_data.data_x_12months   # (N, 28, 136*12)
        data_y_pft = train_data.data_y_pft             # (N, 29, 3)
        pft_MIX_VG = train_data.pft_MIX_VG             # (N, 29, 1)
        data_y_ed_pft = train_data.data_y_ed_pft       # (N, 18, 29, 3)
        InSitu = train_data.InSitu                     # (N, 336, 1)
        STD = train_data.STD                           # (N, 336, 1)

        dataset_ageindep = TensorDataset(
            out_BL.float(),                       # (N, age, 336, 10)
            out_NL.float(),
            out_GS.float(),
            torch.from_numpy(data_x).float(),     # (N, 336, 136)
            data_y_agesum.float(),                # (N, 337, 10)
            torch.from_numpy(data_x_12months).float(),  # (N, 28, 136*12)
            torch.from_numpy(data_y_pft).float(),       # (N, 29, 3)
            torch.from_numpy(InSitu).float(),           # (N, 336, 1)
            torch.from_numpy(AgeWeight).float(),        # (N, age)
            torch.from_numpy(pft_MIX_VG).float(),       # (N, 29, 1)
            torch.from_numpy(STD).float()               # (N, 336, 1)
        )


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model1.train()
            self.model2.train()
            self.std_weight_net.train()
            epoch_time = time.time()
            loader_ageindep = DataLoader(dataset_ageindep, batch_size=batch_size, shuffle=True)
            
            for i, (batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp, batch_std) in enumerate(loader_ageindep):
                iter_count += 1
                model_optim.zero_grad()

                pred_MIX, true_MIX, COM, COM0, pred_PFT, true_PFT, batch_ED, fused = self._process_one_batch_3branch_pft(batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp)                                               
                # Flatten predictions, ground truth, and auxiliary inputs
                pred_MIX = pred_MIX.reshape(-1)       # Model predictions
                true_MIX = true_MIX.reshape(-1)       # Raw target (not used directly here)
                insitu   = batch_InSitu.reshape(-1).to(pred_MIX.device)   # Complete target: observed + imputed
                std_vals = batch_std.reshape(-1).to(pred_MIX.device)      # Std available only at imputation points

                # Create masks
                mask_obs = torch.isnan(std_vals)      # Observed points (std is NaN)
                mask_imp = ~torch.isnan(std_vals)     # Imputation points (std available)

                # ===================== 1. Observed loss =====================
                # At observed points, compute standard MSE between prediction and in-situ data
                if mask_obs.any():
                    loss_obs = 10*criterion(pred_MIX[mask_obs], insitu[mask_obs])
                else:
                    loss_obs = 0.0

                # ===================== 2. Imputation loss =====================
                # At imputation points, use an auxiliary network (std_weight_net) 
                # to produce a weight based on [prediction, std]
                if mask_imp.any():
                    pred_vals = pred_MIX[mask_imp].unsqueeze(-1)   # (N_imp, 1)
                    std_in    = std_vals[mask_imp].unsqueeze(-1)   # (N_imp, 1)

                    # Concatenate prediction and std as input
                    inp = torch.cat([pred_vals, std_in], dim=-1)   # (N_imp, 2)

                    # Pass through small MLP to obtain weights in [0,1]
                    weights = self.std_weight_net(inp).squeeze(-1)
                    weights = torch.clamp(weights, 1e-3, 1.0)      # Avoid zero weight

                    # Compute weighted squared error
                    errors = (pred_MIX[mask_imp] - insitu[mask_imp])**2
                    loss_imp = torch.mean(weights * errors)
                else:
                    loss_imp = 0.0

                # ===================== 3. Final loss =====================
                # Combine observed and imputation losses
                loss_GPP = loss_obs + loss_imp
                loss_PFT = criterion(pred_PFT, true_PFT)

                loss = loss_GPP
                        
                train_loss.append(loss.item())
                                
                if (i+1) % 1==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | loss_PFT: {3:.7f}".format(i + 1, epoch + 1, loss_GPP.item(), loss_PFT.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:

                    loss.backward()
                    model_optim.step()


            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping1(vali_loss, self.model1, os.path.join(path, 'checkpoint_step1.pth'))
            early_stopping2(vali_loss, self.model2, os.path.join(path, 'checkpoint_step2.pth'))
            early_stopping_std(vali_loss, self.std_weight_net, os.path.join(path, 'checkpoint_std_weight_net.pth'))  # NEW

            torch.save(self.std_weight_net.state_dict(), os.path.join(path, 'checkpoint_std_weight_net.pth'))

        
            if early_stopping1.early_stop or early_stopping2.early_stop or early_stopping_std.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint_step1.pth')
        self.model1.load_state_dict(torch.load(best_model_path))

        best_model_path2 = os.path.join(path, 'checkpoint_step2.pth')
        self.model2.load_state_dict(torch.load(best_model_path2))

        best_model_path3 = os.path.join(path, 'checkpoint_std_weight_net.pth')
        self.std_weight_net.load_state_dict(torch.load(best_model_path3))
     
        return self.model1

    
    def test(self, setting):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        test_data, test_loader = self._get_data(flag='test')

        batch_size = self.args.batch_size

        path = os.path.join(self.args.checkpoints, setting)          
        best_model_path1 = os.path.join(path, 'checkpoint_step1.pth')
        self.model1.load_state_dict(torch.load(best_model_path1))
        print('load model from:', best_model_path1) 


        path = os.path.join(self.args.checkpoints, setting)          
        best_model_path2 = os.path.join(path, 'checkpoint_step2.pth')
        self.model2.load_state_dict(torch.load(best_model_path2))
        print('load model from:', best_model_path2 ) 

        path = os.path.join(self.args.checkpoints, setting)          
        std_weight_net_path  = os.path.join(path, 'checkpoint_std_weight_net.pth')
        self.std_weight_net.load_state_dict(torch.load(std_weight_net_path))
        print('load model from:', std_weight_net_path ) 
       
        self.model1.eval()
        self.model2.eval()
        self.std_weight_net.eval()

        # # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        # if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
        #     self.model.module.set_stats(self.y_mean, self.y_std)
        # else:
        #     self.model.set_stats(self.y_mean, self.y_std)


        # === Compute pure model outputs ===
        out_BL, out_NL, out_GS = self._weight_pure_ages(
            test_data.data_x,
            test_data.data_y,
            test_data.AgeWeight,
            self.model1.branch1, self.model1.branch2, self.model1.branch3,
            self.device
        )  # (N, 336, 10)

        data_x = test_data.data_x[:, 0, :, :]        # (N, 336, 136)
        AgeWeight = test_data.AgeWeight              # (N, age)

        AgeWeight_tensor = torch.from_numpy(test_data.AgeWeight).float()   # (N, age)
        data_y_tensor = torch.from_numpy(test_data.data_y).float()         # (N, age, 337, 10)
        data_y_agesum = (data_y_tensor * AgeWeight_tensor.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (N, 337, 10)

        data_x_12months = test_data.data_x_12months   # (N, 28, 136*12)
        data_y_pft = test_data.data_y_pft             # (N, 29, 3)
        pft_MIX_VG = test_data.pft_MIX_VG             # (N, 29, 1)
        data_y_ed_pft = test_data.data_y_ed_pft       # (N, 18, 29, 3)
        InSitu = test_data.InSitu                     # (N, 336, 1)
        STD = test_data.STD                           # (N, ?, ?)

        # === Build dataset and dataloader ===
        dataset_val = TensorDataset(
            out_BL.float(),
            out_NL.float(),
            out_GS.float(),
            torch.from_numpy(data_x).float(),
            data_y_agesum.float(),                   
            torch.from_numpy(data_x_12months).float(),
            torch.from_numpy(data_y_pft).float(),
            torch.from_numpy(InSitu).float(),
            torch.from_numpy(AgeWeight).float(),      # (N, age)
            torch.from_numpy(pft_MIX_VG).float(),     # (N, 29, 1)
            torch.from_numpy(STD).float()
        )

        loader_val = DataLoader(dataset_val, batch_size=self.args.batch_size, shuffle=False)


        outputs_MIXs = []
        batch_ys = []
        COMs = []
        COM0s = []
        outputs_PFTs = []
        batch_PFTs = []
        batch_EDs = []
        batch_stds = []
        batch_fuseds = []

        for i, (batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp, batch_std) in enumerate(loader_val):
            outputs_MIX, batch_y, COM, COM0, pred_PFT, true_PFT, batch_ED, fused = self._process_one_batch_3branch_pft(batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp)                   

            # batch_y
            _y = batch_y.detach().cpu().numpy()
            if _y.ndim == 1:
                _y = _y[None, :]
            batch_ys.append(_y)
              
            outputs_MIXs.append(outputs_MIX.detach().cpu().numpy())
            batch_stds.append(batch_std.detach().cpu().numpy())
            outputs_PFTs.append(pred_PFT.detach().cpu().numpy())
            batch_PFTs.append(true_PFT.detach().cpu().numpy())
            COMs.append(COM.detach().cpu().numpy())
            COM0s.append(COM0.detach().cpu().numpy())
            batch_EDs.append(batch_ED.detach().cpu().numpy())
            batch_fuseds.append(fused.detach().cpu().numpy())
          
        pure_out_weighted = np.concatenate(batch_fuseds, axis=0)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        outputs_MIXs = np.concatenate(outputs_MIXs, axis=0)
        batch_ys = np.concatenate(batch_ys, axis=0)
        print('test shape:', outputs_MIXs.shape, batch_ys.shape)
        outputs_PFTs = np.concatenate(outputs_PFTs, axis=0)
        batch_stds = np.concatenate(batch_stds, axis=0)
        batch_PFTs = np.concatenate(batch_PFTs, axis=0)
        print('test shape:', outputs_PFTs.shape, batch_PFTs.shape)
        batch_EDs = np.concatenate(batch_EDs, axis=0)
        COMs = np.concatenate(COMs, axis=0)
        COMs = COMs[:,:,4] # .reshape(-1, COMs.shape[-2], COMs.shape[-1])      
        COM0s = np.concatenate(COM0s, axis=0)
        COM0s = COM0s[:,:,4] # .reshape(-1, COM0s.shape[-2], COM0s.shape[-1])  

        # batch_stds
        mask_invalid = ~np.isnan(batch_stds)
        # outputs_MIXs[mask_invalid] = float("nan")
        batch_ys[mask_invalid]     = float("nan")

        outputs_MIXs = self.__inverse_norm_data__(outputs_MIXs, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
        batch_ys = self.__inverse_norm_data__(batch_ys, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
        metrics = []
        # channel 0
        pred = outputs_MIXs[..., 0].reshape(-1)
        true = batch_ys[..., 0].reshape(-1)
        mask = ~np.isnan(true)
        mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        # print('channel 0 -> mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        print('channel 0 -> mae:{}, rmse:{}'.format(mae, rmse))
        metrics.append([mae, mse, rmse, mape, mspe])
        # channel 1
        pred = outputs_MIXs[..., 1].reshape(-1)
        true = batch_ys[..., 1].reshape(-1)
        mask = ~np.isnan(true)
        mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        # print('channel 1 -> mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        print('channel 1 -> mae:{}, rmse:{}'.format(mae, rmse))
        metrics.append([mae, mse, rmse, mape, mspe])
        # channel 2
        pred = outputs_MIXs[..., 2].reshape(-1)
        true = batch_ys[..., 2].reshape(-1)
        mask = ~np.isnan(true)
        mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        # print('channel 2 -> mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        metrics.append([mae, mse, rmse, mape, mspe])       
        print('channel 2 -> mae:{}, rmse:{}'.format(mae, rmse))

        np.save(folder_path+'metrics_inverse_norm_.npy', np.array(metrics))
        np.save(folder_path+'Normalized_pred_MIX.npy', outputs_MIXs)
        np.save(folder_path+'Normalized_true_MIX.npy', batch_ys)
        np.save(folder_path+'Normalized_pred_PFT.npy', outputs_PFTs)
        np.save(folder_path+'Normalized_true_PFT.npy', batch_PFTs)
        np.save(folder_path+'Normalized_ED.npy', batch_EDs)
        np.save(folder_path+'pred_MIX_COM.npy', COMs)
        np.save(folder_path+'pred_MIX_COM0.npy', COM0s)

        
        return
    
    

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
        
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path+'/'+'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #         print('load model from:', best_model_path)

    #     self.model.eval()
        
    #     preds = []
        
    #     for i, (batch_x,batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g) in enumerate(pred_loader):
    #         pred, true = self._process_one_batch(
    #             pred_data, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g)
    #         preds.append(pred.detach().cpu().numpy())

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
    #     # result save
    #     folder_path = './results/' + setting +'/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
        
    #     np.save(folder_path+'real_prediction.npy', preds)
        
    #     return


    def _process_one_batch_3branch_pft(self, batch_BL, batch_NL, batch_GS, batch_x, batch_y, batch_x_12months, batch_y_pft, batch_InSitu, batch_aw, batch_vp):
        batch_BL = batch_BL.float().to(self.device) # [32, 18, 336, 10]
        batch_NL = batch_NL.float().to(self.device) # [32, 18, 336, 10]
        batch_GS = batch_GS.float().to(self.device) # [32, 18, 336, 10]        
        
        batch_x = batch_x.float().to(self.device) # [32, 336, 136]
        batch_y = batch_y.float().to(self.device) # [32, 337, 10]
        batch_x_12months = batch_x_12months.float().to(self.device) # [32, 28, 136*12]
        batch_y_pft = batch_y_pft.float().to(self.device) # [32, 29, 3]
        batch_InSitu = batch_InSitu.float().to(self.device)
        # .squeeze() # [32, 336,3]
        batch_vp = batch_vp.float().to(self.device) # [32, 29, 1]
        
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y_pft.shape[0], int(self.args.pred_len / 12), batch_y_pft.shape[-1]], device=self.device).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y_pft.shape[0], int(self.args.pred_len / 12), batch_y_pft.shape[-1]], device=self.device).float()
        dec_inp_PFT = torch.cat([batch_y_pft[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

        if self.args.use_gpu:
            self.model2 = self.model2.to(self.device)   
            self.model1 = self.model1.to(self.device)  

        # shape: [B, T, D] → repeat 18 times
        B = batch_x_12months.shape[0]
        batch_x_12months = batch_x_12months.repeat_interleave(18, dim=0)        # [B*18, T, D]
        age_ids = torch.arange(18).repeat(B).to(self.device)   # [B*18]
        age_emb = self.model2.age_embed(age_ids)                 # [B*18, age_dim]
        age_emb = age_emb.unsqueeze(1).expand(-1, batch_x_12months.shape[1], -1)  # [B*18, T, age_dim]
        batch_x_12months = torch.cat([batch_x_12months, age_emb], dim=-1)       # [B*18, T, D+age_dim]

        dec_inp_PFT = dec_inp_PFT.repeat_interleave(18, dim=0)  # [B, T, 3] to [B*18, T, 3]
        # x_dec = x_dec.reshape(B * 18, *x_dec.shape[2:])  # [B*18, T, 3]

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    out_all = self.model2.branch1(batch_x_12months, dec_inp_PFT, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)[0]       
                else:
                    out_all = self.model2.branch1(batch_x_12months, dec_inp_PFT, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)  
        else:
            if self.args.output_attention:
                out_all = self.model2.branch1(batch_x_12months, dec_inp_PFT, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)[0]   
            else:
                out_all = self.model2.branch1(batch_x_12months, dec_inp_PFT, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)       
        
        out_all = out_all.reshape(B, 18, *out_all.shape[1:])  # [B, 18, T, 3]

        out_all = F.sigmoid(out_all)         
        out_all_sum = out_all.sum(dim=3, keepdim=True) + 1e-12  
        out_all = out_all / out_all_sum

        batch_aw = batch_aw.to(self.device)
        data_aw_expanded = batch_aw.unsqueeze(-1).unsqueeze(-1)  # shape: [8, 18, 1, 1]
        weighted_out = out_all * data_aw_expanded  # [8, 18, 28, 3]
        out_PFT = weighted_out.sum(dim=1)  # shape: [8, 28, 3]
        
        pft_MIX_agesum = out_PFT[:,1:,:] 
        # pft_MIX_agesum shape: [N, 28, 3]
        pft_MIX_agesum = pft_MIX_agesum.unsqueeze(2).repeat(1, 1, 12, 1)  # [N, 28, 12, 3]
        pft_MIX_agesum2 = pft_MIX_agesum.clone()
        pft_MIX_agesum = pft_MIX_agesum.reshape(pft_MIX_agesum2.shape[0], pft_MIX_agesum2.shape[1]*pft_MIX_agesum2.shape[2],pft_MIX_agesum2.shape[3]) # (N, 336, 3)
                             
        x_enc_with_w = torch.cat([batch_x, pft_MIX_agesum], dim=2)  # [N, 336, 136+3]

        pft_VG_agesum = batch_vp[:,1:,:] 
        pft_VG_agesum = pft_VG_agesum.unsqueeze(2).repeat(1, 1, 12, 1)  # [N, 28, 12, 1]
        pft_VG_agesum2 = pft_VG_agesum.clone()
        pft_VG_agesum = pft_VG_agesum.reshape(pft_VG_agesum2.shape[0], pft_VG_agesum2.shape[1]*pft_VG_agesum2.shape[2],pft_VG_agesum2.shape[3]) # (N, 336, 1)
               
        pft_MIX= out_all[:,:,1:,:].unsqueeze(3).repeat(1, 1, 1, 12, 1)  # [8, 18, 28, 3]  to [8, 18, 28, 12, 3] 
        pft_MIX2 = pft_MIX.clone()
        pft_MIX = pft_MIX.reshape(pft_MIX2.shape[0], pft_MIX2.shape[1], pft_MIX2.shape[2]*pft_MIX2.shape[3],pft_MIX2.shape[4]) # [8, 18, 336, 3] 


        # Weight each branch output by corresponding PFT weight
        weighted_out1 = batch_BL * pft_MIX[:,:,:,0].unsqueeze(-1) # [8, 18, 336, 10]
        weighted_out2 = batch_NL * pft_MIX[:,:,:,1].unsqueeze(-1)
        weighted_out3 = batch_GS * pft_MIX[:,:,:,2].unsqueeze(-1)
   
        fused_age = weighted_out1 + weighted_out2 + weighted_out3
        fused = (fused_age * data_aw_expanded).sum(dim=1)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], device=self.device).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], device=self.device).float()
        dec_inp_MIX = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).to(self.device) # [32, 481, 10]

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    out0_Com_mix0 = self.model1.branch0(x_enc_with_w, dec_inp_MIX, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)[0]
                else:
                    out0_Com_mix0 = self.model1.branch0(x_enc_with_w, dec_inp_MIX, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        else:
            if self.args.output_attention:
                out0_Com_mix0 = self.model1.branch0(x_enc_with_w, dec_inp_MIX, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)[0]
            else:
                out0_Com_mix0 = self.model1.branch0(x_enc_with_w, dec_inp_MIX, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)

        out0_Com_mix = out0_Com_mix0 # No.1

        # Final output: weighted sum of all branches plus the competition output
        final_out = fused + out0_Com_mix

        out_denorm = final_out
        mask = torch.ones(out_denorm.shape[-1], device=final_out.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
        out_MIX = self.__norm_data__(out_denorm_relu, self.y_mean, self.y_std)

        outputs = out_MIX[:,:,[4, 9, 7]]


        return outputs, batch_InSitu, out0_Com_mix, out0_Com_mix0, out_PFT[:,1:,:], batch_y_pft[:, 1:, :], batch_y[:, 1:, :], fused


class Exp_Baseline(Exp_Basic):
    def __init__(self, args):
        super(Exp_Baseline, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'OneBranch': OneBranch,
        }
        if self.args.model=='OneBranch':
            e_layers = self.args.e_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
            
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, stage):
        args = self.args

        data_dict = {
            'ED': Dataset_Baseline,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            stat_path = args.stat_path,
            pft_path = args.pft_path,
            asi = args.asi,
            aei = args.aei,
            add_noise = args.add_noise,
            noise_std = args.noise_std,
            flag=flag,
            stage=stage,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean
            
    def vali(self, vali_data, vali_loader, criterion, stage, paras):
        self.model.eval()
        total_loss = []
        with torch.no_grad(): 
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(vali_loader):
                pred, true = self._process_one_batch(batch_x,batch_y, batch_y_agesum, stage)
                pred = pred.reshape(-1)
                true = true.reshape(-1).to(pred.device).type_as(pred) 
                nanmask = torch.isnan(true)
                pred[nanmask]=0
                true[nanmask]=0
                loss = criterion(pred, true)

                total_loss.append(loss.item())  
        total_loss = np.nanmean(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting, stage, paras):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)


        train_data, train_loader = self._get_data(flag = 'train', stage=stage)
        vali_data, vali_loader = self._get_data(flag = 'val', stage=stage)
        test_data, test_loader = self._get_data(flag = 'test', stage=stage)


        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)

        # elif stage == 'pretrain':
        early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()


        # model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(batch_x,batch_y, batch_y_agesum, stage)
                 
                pred = pred.reshape(-1)
                # pred = pred.float()
                true = true.reshape(-1).to(pred.device).type_as(pred)  
                nanmask = torch.isnan(true)
                pred[nanmask]=0
                true[nanmask]=0
                # true = true.to(pred.device)
                loss = criterion(pred, true)
                # loss = loss.float()

                train_loss.append(loss.item())
                
                if (i+1) % 1==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward() # Backward Pass, backpropagation
                    model_optim.step() # Optimization Step

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.nanmean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, stage, paras)
            test_loss = self.vali(test_data, test_loader, criterion, stage, paras)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, os.path.join(path, 'checkpoint.pth'))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    
    def test(self, setting, stage, paras):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        test_data, test_loader = self._get_data(flag='test', stage=stage)

        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)

        path = os.path.join(self.args.checkpoints, setting)  
        best_model_path = path+'/'+'checkpoint.pth'

        state = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(state)
        print('load model from:', best_model_path)

        self.model.eval()

        old_output_attn = getattr(self.args, 'output_attention', False)
        self.args.output_attention = False

        preds = []
        trues = []

        with torch.no_grad():
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(test_loader):           
                pred, true = self._process_one_batch(batch_x,batch_y, batch_y_agesum, stage)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
        
        self.args.output_attention = old_output_attn

        preds = np.concatenate(preds, axis=0)   # (sum_N, T)
        trues = np.concatenate(trues, axis=0)   # (sum_N, T)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-1])  # (-1, T)
        trues = trues.reshape(-1, trues.shape[-1])  # (-1, T)

        print('test shape:', preds.shape, trues.shape)

        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        metrics = []
        preds = self.__inverse_norm_data__(preds, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
        trues = self.__inverse_norm_data__(trues, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
        # channel 0
        pred = preds[..., 0].reshape(-1)
        true = trues[..., 0].reshape(-1)
        mask = ~np.isnan(true)
        mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        print('channel 0 -> mse:{}, mae:{}'.format(mse, mae))
        metrics.append([mae, mse, rmse, mape, mspe])
        # channel 1
        pred = preds[..., 1].reshape(-1)
        true = trues[..., 1].reshape(-1)
        mask = ~np.isnan(true)
        mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        print('channel 1 -> mse:{}, mae:{}'.format(mse, mae))
        metrics.append([mae, mse, rmse, mape, mspe])
        # channel 2
        pred = preds[..., 2].reshape(-1)
        true = trues[..., 2].reshape(-1)
        mask = ~np.isnan(true)
        mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        print('channel 2 -> mse:{}, mae:{}'.format(mse, mae))
        metrics.append([mae, mse, rmse, mape, mspe])

        np.save(folder_path+'metrics_inverse_norm_{}.npy'.format(paras), np.array(metrics))
        # np.save(folder_path+'metrics_6_train_{}.npy'.format(paras), np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred_inverse_norm_{}.npy'.format(paras), preds)
        np.save(folder_path+'true_inverse_norm_{}.npy'.format(paras), trues)
        
        return


    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
        
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path+'/'+'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #         print('load model from:', best_model_path)

    #     self.model.eval()
        
    #     preds = []
        
    #     for i, (batch_x,batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g) in enumerate(pred_loader):
    #         pred, true = self._process_one_batch(
    #             pred_data, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g)
    #         preds.append(pred.detach().cpu().numpy())

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
    #     # result save
    #     folder_path = './results/' + setting +'/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
        
    #     np.save(folder_path+'real_prediction.npy', preds)
        
    #     return


    def _process_one_batch(self, batch_x, batch_y, batch_insitu, stage):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float() # [8, 18, 337, 10]

        N, T, Ff = batch_x.shape
        N1, T1, Vv = batch_y.shape

        data_x_flat = batch_x.reshape(-1, T, Ff)
        data_y_flat = batch_y.reshape(-1, T+1, Vv)
        # batch_tc_flat = batch_tc[:, 0:1].unsqueeze(-1).unsqueeze(-1).expand(-1, 18, -1, -1).reshape(-1, 1, 1)  
        data_x_tensor = data_x_flat.float().to(self.device)
        data_y_tensor0 = data_y_flat.float().to(self.device)
        if self.args.padding == 0:
            dec_inp = torch.zeros([data_y_tensor0.shape[0], self.args.pred_len, data_y_tensor0.shape[-1]], device=self.device)
        elif self.args.padding == 1:
            dec_inp = torch.ones([data_y_tensor0.shape[0], self.args.pred_len, data_y_tensor0.shape[-1]], device=self.device)

        data_y_tensor = torch.cat([data_y_tensor0[:, :self.args.label_len, :], dec_inp], dim=1)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(data_x_tensor, data_y_tensor, stage)[0]
                else:
                    outputs = self.model(data_x_tensor, data_y_tensor, stage)
        else:
            if self.args.output_attention:
                outputs = self.model(data_x_tensor, data_y_tensor, stage)[0]
            else:
                outputs = self.model(data_x_tensor, data_y_tensor, stage)

        outputs = outputs[...,[4,9,7]]
       
        # only output target dimensions
        batch_y = batch_insitu[:,1:,:]
        outputs = outputs

        return outputs, batch_y


class Exp_KGML(Exp_Basic):
    def __init__(self, args):
        super(Exp_KGML, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'OneBranch': OneBranch,
        }
        if self.args.model=='OneBranch':
            e_layers = self.args.e_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
            
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, stage):
        args = self.args

        data_dict = {
            'ED': Dataset_KGML,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            stat_path = args.stat_path,
            pft_path = args.pft_path,
            asi = args.asi,
            aei = args.aei,
            add_noise = args.add_noise,
            noise_std = args.noise_std,
            flag=flag,
            stage=stage,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean

    def compute_total_loss(self, pred, true, criterion):
        """
        Compute total loss = prediction loss + physics-based consistency losses.
        pred/true are normalized. Need to de-normalize before physics loss.
        """
        # ---- 1. Base prediction loss (normalized space is fine here) ----
        loss = criterion(pred, true)

        # ---- 2. Denormalize to physical space ----
        if hasattr(self, "y_mean") and hasattr(self, "y_std"):
            pred_phys = self.__inverse_norm_data__(pred, self.y_mean, self.y_std)
            true_phys = self.__inverse_norm_data__(true, self.y_mean, self.y_std)
        else:
            pred_phys, true_phys = pred, true  # fallback

        # ---- 3. Physics-guided loss terms (physical space) ----
        # NEE = Rh - NPP
        loss_c1 = torch.mean((pred_phys[..., 7] - pred_phys[..., 6] + pred_phys[..., 5])**2)

        # Ra = GPP – NPP
        loss_c2 = torch.mean((pred_phys[..., 8] - pred_phys[..., 4] + pred_phys[..., 5])**2)

        # RECO = NEE + GPP
        loss_c3 = torch.mean((pred_phys[..., 9] - pred_phys[..., 4] - pred_phys[..., 7])**2)

        # ---- 4. Total loss ----
        total_loss = loss + 0.1 * loss_c1 + 0.1 * loss_c2 + 0.1 * loss_c3

        return total_loss
            
    def vali(self, vali_data, vali_loader, criterion, stage, paras):
        self.model.eval()
        total_loss = []
        with torch.no_grad(): 
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    batch_x, batch_y, stage)
                
                if stage == 'pretrain':    
                    loss = self.compute_total_loss(pred, true, criterion) 
                elif stage == 'finetune':
                    batch_aw = torch.as_tensor(batch_aw, device=pred.device).type_as(pred)   # ★
                    pred = (pred * batch_aw.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [N, T, 10]

                    pred = pred[:, :, [4, 9, 7]]
                    true = batch_y_agesum[:,1:,:].to(pred.device).type_as(pred)  
                    pred = pred.reshape(-1)
                    # pred = pred.float()
                    true = true.reshape(-1)
                    nanmask = torch.isnan(true)
                    pred[nanmask]=0
                    true[nanmask]=0
                    loss = criterion(pred, true)

                total_loss.append(loss.item())  
        total_loss = np.nanmean(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting, stage, paras):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)


        train_data, train_loader = self._get_data(flag = 'train', stage=stage)
        vali_data, vali_loader = self._get_data(flag = 'val', stage=stage)
        test_data, test_loader = self._get_data(flag = 'test', stage=stage)


        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        

        if stage == 'finetune':
            early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)
            # load
            setting0 = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_KGML_{}'.format(self.args.model, self.args.data,
            self.args.seq_len, self.args.label_len, self.args.pred_len,
            self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, self.args.ii)

            pathstep1 = os.path.join(self.args.checkpoints, setting0)
            state_dict_full = torch.load(os.path.join(pathstep1, 'checkpoint.pth'))
            self.model.load_state_dict(state_dict_full)


            for name, param in self.model.named_parameters():
                param.requires_grad = True 

            print("Finetuning the full model (all layers are trainable)")
            model_optim = self._select_optimizer()

        elif stage == 'pretrain':
            early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)
            model_optim = self._select_optimizer()


        # model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(batch_x,batch_y, stage)

                if stage == 'pretrain':       
                    loss = self.compute_total_loss(pred, true, criterion)                

                elif stage == 'finetune':
                    # batch_aw = batch_aw.to(pred.device) # [3,18]
                    batch_aw = torch.as_tensor(batch_aw, device=pred.device).type_as(pred)   # ★

                    pred = (pred * batch_aw.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [N, T, 10]
                    pred = pred[:, :, [4, 9, 7]]
                    true = batch_y_agesum[:,1:,:].to(pred.device).type_as(pred)  
                    pred = pred.reshape(-1)
                    # pred = pred.float()
                    true = true.reshape(-1)
                    nanmask = torch.isnan(true)
                    pred[nanmask]=0
                    true[nanmask]=0
                    # true = true.to(pred.device)
                    loss = criterion(pred, true)
                    # loss = loss.float()

                train_loss.append(loss.item())
                
                if (i+1) % 1==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward() # Backward Pass, backpropagation
                    model_optim.step() # Optimization Step

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.nanmean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, stage, paras)
            test_loss = self.vali(test_data, test_loader, criterion, stage, paras)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if stage == 'pretrain':    
                early_stopping(vali_loss, self.model, os.path.join(path, 'checkpoint.pth'))
            elif stage == 'finetune':
                early_stopping(vali_loss, self.model, os.path.join(path, 'checkpoint_step2.pth'))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        if stage == 'pretrain':    
            best_model_path = path+'/'+'checkpoint.pth'
        elif stage == 'finetune':            
            best_model_path = path+'/'+'checkpoint_step2.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    
    def test(self, setting, stage, paras):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        test_data, test_loader = self._get_data(flag='test', stage=stage)

        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)

        path = os.path.join(self.args.checkpoints, setting)
        if stage == 'pretrain':    
            best_model_path = path+'/'+'checkpoint.pth'
        elif stage == 'finetune':            
            best_model_path = path+'/'+'checkpoint_step2.pth'
        state = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(state)
        print('load model from:', best_model_path)

        self.model.eval()

        old_output_attn = getattr(self.args, 'output_attention', False)
        self.args.output_attention = False

        preds = []
        trues = []

        with torch.no_grad():
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(test_loader):
                
                pred, true = self._process_one_batch(batch_x,batch_y, stage)  
                if stage == 'finetune':
                    if not isinstance(batch_aw, torch.Tensor):
                        batch_aw = torch.tensor(batch_aw, dtype=torch.float32)
                    pred = pred.detach().cpu()   
                    batch_aw_cpu = batch_aw.detach().to('cpu')

                    pred = (pred*batch_aw_cpu.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # shape: [8, 336, 10]

                    pred = pred[:, :, [4, 9, 7]]
                    if not isinstance(batch_y_agesum, torch.Tensor):
                        batch_y_agesum = torch.tensor(batch_y_agesum, dtype=torch.float32)
                    true = batch_y_agesum[:, 1:].float().cpu()

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
        
        self.args.output_attention = old_output_attn


        if stage == 'pretrain': 
            preds = np.concatenate(preds, axis=0)   # (sum_N, age, T, C)
            trues = np.concatenate(trues, axis=0)   # (sum_N, age, T, C)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])  # (-1, T, C)
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])  # (-1, T, C)

        elif stage == 'finetune':
            preds = np.concatenate(preds, axis=0)   # (sum_N, T)
            trues = np.concatenate(trues, axis=0)   # (sum_N, T)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-1])  # (-1, T)
            trues = trues.reshape(-1, trues.shape[-1])  # (-1, T)

        print('test shape:', preds.shape, trues.shape)

        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        if stage == 'pretrain':
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            np.save(folder_path+'metrics_6_step1.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path+'pred_6_step1.npy', preds)
            np.save(folder_path+'true_6_step1.npy', trues)
        elif stage == 'finetune':
            metrics = []
            preds = self.__inverse_norm_data__(preds, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
            trues = self.__inverse_norm_data__(trues, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
            # channel 0
            pred = preds[..., 0].reshape(-1)
            true = trues[..., 0].reshape(-1)
            mask = ~np.isnan(true)
            mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
            print('channel 0 -> mae:{}, rmse:{}'.format(mae, rmse))
            metrics.append([mae, mse, rmse, mape, mspe])
            # channel 1
            pred = preds[..., 1].reshape(-1)
            true = trues[..., 1].reshape(-1)
            mask = ~np.isnan(true)
            mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
            print('channel 1 -> mae:{}, rmse:{}'.format(mae, rmse))
            metrics.append([mae, mse, rmse, mape, mspe])
            # channel 2
            pred = preds[..., 2].reshape(-1)
            true = trues[..., 2].reshape(-1)
            mask = ~np.isnan(true)
            mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
            print('channel 2 -> mae:{}, rmse:{}'.format(mae, rmse))
            metrics.append([mae, mse, rmse, mape, mspe])

            np.save(folder_path+'metrics_inverse_norm_{}.npy'.format(paras), np.array(metrics))
            # np.save(folder_path+'metrics_6_train_{}.npy'.format(paras), np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path+'pred_inverse_norm_{}.npy'.format(paras), preds)
            np.save(folder_path+'true_inverse_norm_{}.npy'.format(paras), trues)
        
        return


    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
        
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path+'/'+'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #         print('load model from:', best_model_path)

    #     self.model.eval()
        
    #     preds = []
        
    #     for i, (batch_x,batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g) in enumerate(pred_loader):
    #         pred, true = self._process_one_batch(
    #             pred_data, batch_x, batch_y, batch_pft_bl, batch_pft_nl, batch_pft_g)
    #         preds.append(pred.detach().cpu().numpy())

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
    #     # result save
    #     folder_path = './results/' + setting +'/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
        
    #     np.save(folder_path+'real_prediction.npy', preds)
        
    #     return

    def _process_one_batch(self, batch_x, batch_y, stage):
        batch_x = batch_x.float().to(self.device) # [8, 18, 336, 136]
        batch_y = batch_y.float() # [8, 18, 337, 10]

        N, age, T, Ff = batch_x.shape
        N1, age1, T1, Vv = batch_y.shape

        data_x_flat = batch_x.reshape(-1, T, Ff)
        data_y_flat = batch_y.reshape(-1, T+1, Vv)
        data_x_tensor = data_x_flat.float().to(self.device)
        data_y_tensor0 = data_y_flat.float().to(self.device)
        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([data_y_tensor0.shape[0], self.args.pred_len, data_y_tensor0.shape[-1]], device=self.device)
        elif self.args.padding == 1:
            dec_inp = torch.ones([data_y_tensor0.shape[0], self.args.pred_len, data_y_tensor0.shape[-1]], device=self.device)

        data_y_tensor = torch.cat([data_y_tensor0[:, :self.args.label_len, :], dec_inp], dim=1)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(data_x_tensor, data_y_tensor, stage)[0]
                else:
                    outputs = self.model(data_x_tensor, data_y_tensor, stage)
        else:
            if self.args.output_attention:
                outputs = self.model(data_x_tensor, data_y_tensor, stage)[0]
            else:
                outputs = self.model(data_x_tensor, data_y_tensor, stage)

        D = outputs.shape[-1]
        outputs = outputs.reshape(N, age, T, D)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,:,-self.args.pred_len:,f_dim:].to(self.device)
        
        # only output target dimensions
        batch_y = batch_y[:,:,:,f_dim:]
        outputs = outputs[:,:,:,f_dim:]

        return outputs, batch_y



