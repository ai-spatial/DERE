import torch
import random
import numpy as np

from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast, Exp_Long_Term_Forecast_2steps
from model.SimpleTM import Model
from utils.tools import dotdict

if __name__ == '__main__':
    args = dotdict()

    # basic config
    args.is_training = 1
    args.model_id = 'test'
    args.model = 'SimpleTM'

    # data loader
    args.data = 'ed'
    args.root_path = './dataset/ED/'
    args.features = 'M'
    args.target = 'OT'
    args.freq = 'h'
    args.checkpoints = './checkpoints/'

    # forecasting task
    args.seq_len = 336
    args.label_len = 1
    args.pred_len = 336

    # model define
    args.enc_in = 136+10 # encoder input size
    args.dec_in = 136+10  # decoder input size
    args.c_out = 10 # output size
    args.n_heads = 8
    args.d_layers = 1
    args.moving_avg = 25
    args.factor = 5
    args.distil = True
    args.dropout = 0.05
    args.geomattn_dropout = 0.5
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.do_predict = False
    args.add_noise = False
    args.noise_std = 0.0
    
    # optimization
    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 100
    args.batch_size = 8
    args.patience = 5
    args.learning_rate = 0.0001
    args.des = 'test'
    args.loss = 'MSE'
    args.lradj = 'type1'
    args.pct_start = 0.2
    args.use_amp = False

    # GPU
    args.use_gpu = True
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    # others
    args.exp_name = 'MTSF'
    args.channel_independence = False
    args.inverse = False
    args.class_strategy = 'projection'
    args.efficient_training = False
    args.use_norm = True
    args.partial_start_index = 0

    # SimpleTM Arguments
    args.requires_grad = True
    args.wv = 'db1'
    args.m = 3
    args.kernel_size = None
    args.alpha = 0.0
    args.l1_weight = 0.0
    args.d_model = 256
    args.d_ff = 1024
    args.e_layers = 1
    args.compile = False
    args.output_attention = False

    # seed
    args.fix_seed = 2023


    # args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    fix_seed = args.fix_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast


    networks = ['above', "ameriflux", "fluxnet", "icos-ww", "mix"]
    for net in networks:
        args.data_path = f'res_train4_test8_extract_4types_28years_{net}.npz' # data file
        args.pft_path = f'pft_dataset_12mean_4types_28years_ESACCI_plusAW_{net}.npz' # pft file
        args.stat_path = 'data_stats_with_NEE_Ra_RECO.npz' # stat file

        args.target_root_path = args.root_path
        args.target_data_path = args.data_path
        # setting record of experiments
        if args.is_training:
            for ii in range(args.itr):
                args.ii = ii
                setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_OneStep_{}'.format(
                    args.model_id,          # {}
                    args.model,             # {}
                    args.data,              # {}
                    args.features,          # {}
                    args.seq_len,           # sl{}
                    args.label_len,         # ll{}
                    args.pred_len,          # pl{}
                    args.d_model,           # dm{}
                    args.n_heads,           # nh{}
                    args.e_layers,          # el{}
                    args.d_layers,          # dl{}
                    args.d_ff,              # df{}
                    args.factor,            # fc{}
                    args.embed,             # eb{}
                    int(args.distil),       # dt{}
                    args.des,               # {}
                    args.class_strategy,    # {}
                    args.ii,                     # {}
                    net                     # {}
                )

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting, stage='')

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, stage='', test=1)

                # if args.do_predict:
                #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                #     exp.predict(setting, True)

                torch.cuda.empty_cache()
 