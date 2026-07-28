import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast, Exp_Long_Term_Forecast_2steps
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np
from utils.tools import dotdict

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args = dotdict()
    # ===== basic config =====
    args.is_training = 1
    args.model_id = 'test'
    args.model = 'TimeXer'   

    # ===== data loader =====
    args.data = 'ed2'
    args.features = 'M'   # options: [M, S, MS]
    args.target = 'OT'
    args.freq = 'h'       # options: [s, t, h, d, b, w, m, 15min, 3h ...]
    args.checkpoints = './checkpoints/'
    args.add_noise = False
    args.noise_std = 0.0
    # ===== forecasting task =====
    args.seq_len = 336
    args.label_len = 1
    args.pred_len = 336

    # ===== model define =====
    args.enc_in = 136+10 # encoder input size
    args.dec_in = 10 # decoder input size
    args.c_out = 10 # output size
    args.d_model = 512 # dimension of model
    args.n_heads = 8 # num of heads
    args.e_layers = 2 # num of encoder layers
    args.d_layers = 1 # num of decoder layers
    args.d_ff = 2048 # dimension of fcn in model
    args.moving_avg = 25
    args.factor = 5
    args.distil = True
    args.dropout = 0.05
    args.embed = 'timeF'     # options: [timeF, fixed, learned]
    args.activation = 'gelu'
    args.output_attention = False
    args.do_predict = False

    # ===== optimization =====
    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 100
    args.batch_size = 2
    args.patience = 5
    args.learning_rate = 0.0001
    args.des = 'exp' # test
    args.loss = 'MSE'
    args.lradj = 'type1'
    args.use_amp = False

    # ===== GPU =====
    args.use_gpu = True
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0'

    # ===== iTransformer =====
    args.exp_name = 'MTSF'  # options: [MTSF, partial_train]
    args.channel_independence = False
    args.inverse = False
    args.class_strategy = 'projection'   # options: [projection, average, cls_token]

    args.efficient_training = False
    args.use_norm = True
    args.partial_start_index = 0


    # args = parser.parse_args([])
    # args = dotdict()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # args.exp_name = 'MTSF'
    # args.is_training = True
    # args.itr = 1


    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast_2steps


    args.root_path = './data/ED/'
    # args.data_path = 'electricity.csv'
    args.data_path = 'res_train4_test8_extract_28years_ageindependent_update_with_NEE_Ra_RECO.npz' # data file
    args.stat_path = 'data_stats_with_NEE_Ra_RECO.npz' # stat file
    args.pft_path = 'pft_dataset_12mean_28years_ageindependent_plus_ageweight_update.npz' # pft file

    args.target_root_path = args.root_path
    args.target_data_path = args.data_path
    # setting record of experiments
    if args.is_training:
        for ii in range(args.itr):
            args.ii = ii
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_TwoSteps'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting, stage='pretrain')

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, stage='pretrain', test=1)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            torch.cuda.empty_cache()

            

    networks = ['above', "ameriflux", "fluxnet", "icos-ww", "multiple"]
    for net in networks:
        args.root_path = './data/ED/'
        # args.data_path = 'electricity.csv'
        args.data_path = f'res_train4_test8_extract_4types_28years_{net}.npz' # data file
        args.pft_path = f'pft_dataset_12mean_4types_28years_ESACCI_plusAW_{net}.npz' # pft file
        args.stat_path = 'data_stats_with_NEE_Ra_RECO.npz' # stat file

        args.target_root_path = args.root_path
        args.target_data_path = args.data_path
        # setting record of experiments
        if args.is_training:
            for ii in range(args.itr):
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_TwoSteps_{}'.format(
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    args.class_strategy, ii, net)

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting, stage='finetune')

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, stage='finetune', test=1)

                # if args.do_predict:
                #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                #     exp.predict(setting, True)

                torch.cuda.empty_cache()
        
