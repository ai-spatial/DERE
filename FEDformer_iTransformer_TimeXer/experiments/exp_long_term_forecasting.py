from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, EarlyStopping1, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast_2steps(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_2steps, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, stage):
        data_set, data_loader = data_provider(self.args, flag, stage)
        return data_set, data_loader

    def set_stats(self, mean, std):
        self.y_mean = mean
        self.y_std = std


    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
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

    def vali(self, vali_data, vali_loader, criterion, stage):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                N, age, T, Ff = batch_x.shape
                N1, age1, T1, Vv = batch_y.shape

                data_x_flat = batch_x.reshape(-1, T, Ff)
                data_y_flat = batch_y.reshape(-1, T+1, Vv)
                batch_x = data_x_flat.float().to(self.device)
                data_y_tensor0 = data_y_flat.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(data_y_tensor0[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([data_y_tensor0[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                batch_x = torch.cat([batch_x, dec_inp[:,:-1,:]], dim=-1) 

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp)

                D = outputs.shape[-1]
                outputs = outputs.reshape(N, age, T, D)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, :, -self.args.pred_len:, f_dim:].to(self.device)

                if stage == 'pretrain':    
                    # loss = criterion(pred, true)
                    # loss_c1 = torch.mean((pred[...,7] - pred[...,6] + pred[...,5])**2)
                    # loss_c2 = torch.mean((pred[...,8] - pred[...,4] + pred[...,5])**2)
                    # loss_c3 = torch.mean((pred[...,9] - pred[...,4] - pred[...,7])**2)
                    # loss = loss + 0.1*loss_c1 + 0.1*loss_c2 + 0.1*loss_c3
                    loss = self.compute_total_loss(outputs, batch_y, criterion) 
                    # loss = loss + 0.1*loss_c1 + 0.1*loss_c2
                elif stage == 'finetune':
                    # batch_aw = batch_aw.to(pred.device)
                    # pred = (pred*batch_aw.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # shape: [8, 336, 7]
                    # pred = pred*batch_tc
                    batch_aw = torch.as_tensor(batch_aw, device=outputs.device).type_as(outputs)   # ★
                    # batch_tc = torch.as_tensor(batch_tc, device=pred.device).type_as(pred)   # ★

                    outputs = (outputs * batch_aw.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [N, T, 7]
                    # pred = pred * batch_tc[:, 0].unsqueeze(-1).unsqueeze(-1)         # ★

                    # pred = pred[:,:,4]
                    # true = batch_y_agesum[:,1:]
                    # pred = pred.reshape(-1)
                    # pred = pred.float()
                    # true = true.reshape(-1)
                    # nanmask = torch.isnan(true)
                    # pred[nanmask]=0
                    # true[nanmask]=0
                    # true = true.to(pred.device)
                    # loss = criterion(pred, true)
                    # loss = loss.float()

                    outputs = outputs[:, :, [4, 9, 7]]
                    batch_y = batch_y_agesum[:,1:,:].to(outputs.device).type_as(outputs)  
                    outputs = outputs.reshape(-1)
                    # pred = pred.float()
                    batch_y = batch_y.reshape(-1)
                    nanmask = torch.isnan(batch_y)
                    outputs[nanmask]=0
                    batch_y[nanmask]=0
                    # true = true.to(pred.device)
                    loss = criterion(outputs, batch_y)


                # outputs = outputs[...,[4,9,7]]
                # batch_y = batch_y_agesum[:,1:,:]



                # outputs = outputs.reshape(-1)
                # # pred = pred.float()
                # batch_y = batch_y.reshape(-1).to(outputs.device).type_as(outputs)  
                # nanmask = torch.isnan(batch_y)
                # outputs[nanmask]=0
                # batch_y[nanmask]=0

                # pred = outputs.detach().cpu()
                # true = batch_y.detach().cpu()

                # loss = criterion(pred, true)

                total_loss.append(loss.item() )
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, stage):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        train_data, train_loader = self._get_data(flag='train', stage=stage)
        vali_data, vali_loader = self._get_data(flag='val', stage=stage)
        test_data, test_loader = self._get_data(flag='test', stage=stage)

        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)


        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # model_optim = self._select_optimizer()

        if stage == 'finetune':
            early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)
            # load
            setting0 = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_TwoSteps'.format(
                    self.args.model_id,
                    self.args.model,
                    self.args.data,
                    self.args.features,
                    self.args.seq_len,
                    self.args.label_len,
                    self.args.pred_len,
                    self.args.d_model,
                    self.args.n_heads,
                    self.args.e_layers,
                    self.args.d_layers,
                    self.args.d_ff,
                    self.args.factor,
                    self.args.embed,
                    self.args.distil,
                    self.args.des,
                    self.args.class_strategy, self.args.ii)
            pathstep1 = os.path.join(self.args.checkpoints, setting0)
            state_dict_full = torch.load(os.path.join(pathstep1, 'checkpoint.pth'))
            self.model.load_state_dict(state_dict_full)

            # for name, param in self.model.named_parameters():
            #     if 'encoder' in name or 'enc_embedding' in name:
            #         param.requires_grad = False

            # print("Freezing encoder-related layers (enc_embedding, encoder)")

            for name, param in self.model.named_parameters():
                param.requires_grad = True 

            print("Finetuning the full model (all layers are trainable)")


            model_optim = self._select_optimizer()

        elif stage == 'pretrain':
            early_stopping = EarlyStopping1(patience=self.args.patience, verbose=True)
            model_optim = self._select_optimizer()
        

        criterion = self._select_criterion()

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
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                N, age, T, Ff = batch_x.shape
                N1, age1, T1, Vv = batch_y.shape

                data_x_flat = batch_x.reshape(-1, T, Ff)
                data_y_flat = batch_y.reshape(-1, T+1, Vv)
                batch_x = data_x_flat.float().to(self.device)
                data_y_tensor0 = data_y_flat.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(data_y_tensor0[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([data_y_tensor0[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                batch_x = torch.cat([batch_x, dec_inp[:,:-1,:]], dim=-1) 

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp)

                D = outputs.shape[-1]
                outputs = outputs.reshape(N, age, T, D)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :,-self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, :,-self.args.pred_len:, f_dim:].to(self.device)



                if stage == 'pretrain':    
                    # loss = criterion(pred, true)
                    # loss_c1 = torch.mean((pred[...,7] - pred[...,6] + pred[...,5])**2)
                    # loss_c2 = torch.mean((pred[...,8] - pred[...,4] + pred[...,5])**2)
                    # loss_c3 = torch.mean((pred[...,9] - pred[...,4] - pred[...,7])**2)
                    # loss = loss + 0.1*loss_c1 + 0.1*loss_c2 + 0.1*loss_c3     
                    loss = self.compute_total_loss(outputs, batch_y, criterion)                
                    # loss = loss + 0.1*loss_c1 + 0.1*loss_c2

                elif stage == 'finetune':
                    # batch_aw = batch_aw.to(pred.device) # [3,18]
                    batch_aw = torch.as_tensor(batch_aw, device=outputs.device).type_as(outputs)   # ★
                    # batch_tc = torch.as_tensor(batch_tc, device=pred.device).type_as(pred)   # ★

                    outputs = (outputs * batch_aw.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [N, T, 7]
                    # pred = pred * batch_tc[:, 0].unsqueeze(-1).unsqueeze(-1)         # ★

                    outputs = outputs[...,[4,9,7]]
                    batch_y = batch_y_agesum[:,1:,:]



                    outputs = outputs.reshape(-1)
                    # pred = pred.float()
                    batch_y = batch_y.reshape(-1).to(outputs.device).type_as(outputs)  
                    nanmask = torch.isnan(batch_y)
                    outputs[nanmask]=0
                    batch_y[nanmask]=0

                    loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, stage)
            test_loss = self.vali(test_data, test_loader, criterion, stage)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # early_stopping(vali_loss, self.model, path)
            if stage == 'pretrain':    
                early_stopping(vali_loss, self.model, os.path.join(path, 'checkpoint.pth'))
            elif stage == 'finetune':
                early_stopping(vali_loss, self.model, os.path.join(path, 'checkpoint_step2_.pth'))
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        # best_model_path = path + '/' + 'checkpoint.pth'
        if stage == 'pretrain':    
            best_model_path = path+'/'+'checkpoint.pth'
        elif stage == 'finetune':            
            best_model_path = path+'/'+'checkpoint_step2_.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, stage, test=0):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)
    
        test_data, test_loader = self._get_data(flag='test', stage=stage)

        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)


        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting)
            if stage == 'pretrain':    
                best_model_path = path+'/'+'checkpoint.pth'
            elif stage == 'finetune':            
                best_model_path = path+'/'+'checkpoint_step2_.pth'
            # self.model.load_state_dict(torch.load(best_model_path))
            # print('load model from:', best_model_path)


        state = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(state)
        print('load model from:', best_model_path)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        old_output_attn = getattr(self.args, 'output_attention', False)
        self.args.output_attention = False


        with torch.no_grad():
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                N, age, T, Ff = batch_x.shape
                N1, age1, T1, Vv = batch_y.shape


                data_x_flat = batch_x.reshape(-1, T, Ff)
                data_y_flat = batch_y.reshape(-1, T+1, Vv)
                batch_x = data_x_flat.float().to(self.device)
                data_y_tensor0 = data_y_flat.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(data_y_tensor0[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([data_y_tensor0[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                batch_x = torch.cat([batch_x, dec_inp[:,:-1,:]], dim=-1) 

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]

                    else:
                        outputs = self.model(batch_x, dec_inp)


                D = outputs.shape[-1]
                outputs = outputs.reshape(N, age, T, D)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, :, -self.args.pred_len:, f_dim:].to(self.device)                       

                if stage == 'finetune':
                    if not isinstance(batch_aw, torch.Tensor):
                        batch_aw = torch.tensor(batch_aw, dtype=torch.float32)
                    outputs = outputs.detach().cpu()   
                    batch_aw_cpu = batch_aw.detach().to('cpu')
                    # batch_tc_cpu = batch_tc.detach().to('cpu')

                    # batch_aw = batch_aw.to(pred.device)
                    outputs = (outputs*batch_aw_cpu.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # shape: [8, 336, 7]
                    # pred = pred*batch_tc_cpu[:, 0].unsqueeze(-1).unsqueeze(-1)
                    # pred = pred[:,:,4]
                    outputs = outputs[:, :, [4, 9, 7]]
                    # true = batch_y_agesum[:,1:,:].to(pred.device).type_as(pred)  
                    # pred = pred.reshape(-1)
                    # # pred = pred.float()
                    # true = true.reshape(-1)
                    # nanmask = torch.isnan(true)
                    # pred[nanmask]=0
                    # true[nanmask]=0
                    # # true = true.to(pred.device)
                    # loss = criterion(pred, true)

                    # true = batch_y_agesum[:,1:]
                    # pred = pred.float()   
                    if not isinstance(batch_y_agesum, torch.Tensor):
                        batch_y_agesum = torch.tensor(batch_y_agesum, dtype=torch.float32)
                    batch_y = batch_y_agesum[:, 1:].float().cpu()
                
                
                
                # outputs = outputs[...,[4,9,7]]
                # batch_y = batch_y_agesum[:,1:,:]


                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # preds = np.array(preds)
        # trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1,  preds.shape[-1])
        # trues = trues.reshape(-1,  trues.shape[-1])
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
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # metrics = []
        # preds = self.__inverse_norm_data__(preds, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
        # trues = self.__inverse_norm_data__(trues, self.y_mean[[4, 9, 7]].cpu().numpy(), self.y_std[[4, 9, 7]].cpu().numpy())
        # # channel 0
        # pred = preds[..., 0].reshape(-1)
        # true = trues[..., 0].reshape(-1)
        # mask = ~np.isnan(true)
        # mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        # print('channel 0 -> mse:{}, mae:{}'.format(mse, mae))
        # metrics.append([mae, mse, rmse, mape, mspe])
        # # channel 1
        # pred = preds[..., 1].reshape(-1)
        # true = trues[..., 1].reshape(-1)
        # mask = ~np.isnan(true)
        # mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        # print('channel 1 -> mse:{}, mae:{}'.format(mse, mae))
        # metrics.append([mae, mse, rmse, mape, mspe])
        # # channel 2
        # pred = preds[..., 2].reshape(-1)
        # true = trues[..., 2].reshape(-1)
        # mask = ~np.isnan(true)
        # mae, mse, rmse, mape, mspe = metric(pred[mask], true[mask])
        # print('channel 2 -> mse:{}, mae:{}'.format(mse, mae))
        # metrics.append([mae, mse, rmse, mape, mspe])

        # np.save(folder_path+'metrics_inverse_norm_.npy', np.array(metrics))
        # # np.save(folder_path+'metrics_6_train_{}.npy'.format(paras), np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path+'pred_inverse_norm_.npy', preds)
        # np.save(folder_path+'true_inverse_norm_.npy', trues)  
        # 
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

            np.save(folder_path+'metrics_inverse_norm_{}.npy', np.array(metrics))
            # np.save(folder_path+'metrics_6_train_{}.npy'.format(paras), np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path+'pred_inverse_norm_{}.npy', preds)
            np.save(folder_path+'true_inverse_norm_{}.npy', trues)
        

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    




class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, stage):
        data_set, data_loader = data_provider(self.args, flag, stage)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def set_stats(self, mean, std):
        self.y_mean = mean
        self.y_std = std


    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean
    
    def vali(self, vali_data, vali_loader, criterion, stage):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                N, T, Ff = batch_x.shape
                N1, T1, Vv = batch_y.shape

                data_x_flat = batch_x.reshape(-1, T, Ff)
                data_y_flat = batch_y.reshape(-1, T+1, Vv)
                batch_x = data_x_flat.float().to(self.device)
                batch_y = data_y_flat.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                batch_x = torch.cat([batch_x, dec_inp[:,:-1,:]], dim=-1) 

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp)

                outputs = outputs[...,[4,9,7]]
                batch_y = batch_y_agesum[:,1:,:]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.reshape(-1)
                # pred = pred.float()
                batch_y = batch_y.reshape(-1).to(outputs.device).type_as(outputs)  
                nanmask = torch.isnan(batch_y)
                outputs[nanmask]=0
                batch_y[nanmask]=0

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, stage):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        train_data, train_loader = self._get_data(flag='train', stage=stage)
        vali_data, vali_loader = self._get_data(flag='val', stage=stage)
        test_data, test_loader = self._get_data(flag='test', stage=stage)

        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

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
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                N, T, Ff = batch_x.shape
                N1, T1, Vv = batch_y.shape


                data_x_flat = batch_x.reshape(-1, T, Ff)
                data_y_flat = batch_y.reshape(-1, T+1, Vv)
                batch_x = data_x_flat.float().to(self.device)
                batch_y = data_y_flat.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                batch_x = torch.cat([batch_x, dec_inp[:,:-1,:]], dim=-1) 

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp)

                outputs = outputs[...,[4,9,7]]
                batch_y = batch_y_agesum[:,1:,:]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.reshape(-1)
                # pred = pred.float()
                batch_y = batch_y.reshape(-1).to(outputs.device).type_as(outputs)  
                nanmask = torch.isnan(batch_y)
                outputs[nanmask]=0
                batch_y[nanmask]=0

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, stage)
            test_loss = self.vali(test_data, test_loader, criterion, stage)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, stage, test=0):
        stat_raw = np.load(os.path.join(self.args.root_path, self.args.stat_path))
        self.y_mean = torch.tensor(stat_raw['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(stat_raw['y_std'], dtype=torch.float32, device=self.device)

        test_data, test_loader = self._get_data(flag='test', stage=stage)

        # Set the internal y_mean and y_std for the model, used for inverse normalization before ReLU and re-normalization afterward
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'set_stats'):
            self.model.module.set_stats(self.y_mean, self.y_std)
        else:
            self.model.set_stats(self.y_mean, self.y_std)


        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x,batch_y, batch_y_agesum, batch_aw) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                N, T, Ff = batch_x.shape
                N1, T1, Vv = batch_y.shape


                data_x_flat = batch_x.reshape(-1, T, Ff)
                data_y_flat = batch_y.reshape(-1, T+1, Vv)
                batch_x = data_x_flat.float().to(self.device)
                batch_y = data_y_flat.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                batch_x = torch.cat([batch_x, dec_inp[:,:-1,:]], dim=-1) 

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]

                    else:
                        outputs = self.model(batch_x, dec_inp)

                outputs = outputs[...,[4,9,7]]
                batch_y = batch_y_agesum[:,1:,:]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1,  preds.shape[-1])
        trues = trues.reshape(-1,  trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
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

        np.save(folder_path+'metrics_inverse_norm_.npy', np.array(metrics))
        # np.save(folder_path+'metrics_6_train_{}.npy'.format(paras), np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred_inverse_norm_.npy', preds)
        np.save(folder_path+'true_inverse_norm_.npy', trues)    

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return