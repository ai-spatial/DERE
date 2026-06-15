import os
from utils.tools import dotdict
from exp.exp_informer import Exp_KGML
import torch

args = dotdict()

args.model =  'OneBranch' # 'MultiBranch_AddPure' # 'MultiBranch' # 'informer_noT' # model of experiment, options: [informer, informerstack, informerlight(TBD)]
args.data = 'ED' # data
args.root_path = 'Dataset' # root path of data file
# args.data_path = 'res_train4_test8_extract_28years_ageindependent.npz' # data file
args.stat_path = 'data_stats_with_NEE_Ra_RECO.npz' # stat file
# args.pft_path = 'pft_dataset_12mean_28years_ageindependent_plus_ageweight.npz' # pft file
args.asi = 0
args.aei = 18

args.seq_len = 336 # [idx : idx+seq_len]: input sequence length of Informer encoder
args.label_len = 1 # [idx+seq_len-label_len : idx+seq_len]: start token length of Informer decoder; overlap with seq_len counting from the end of seq_len
args.pred_len = 336 # [idx+seq_len : idx+seq_len+pred_len]: prediction sequence length; no overlap iwth seq_len
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 136 # encoder input size
args.dec_in = 10 # decoder input size
args.c_out = 10 # output size
args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.05 # dropout
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True # whether to use distilling in encoder
args.output_attention = False # whether to output attention in ecoder
args.mix = True
args.padding = 0

args.batch_size = 8

args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False # whether to use automatic mixed precision training

args.num_workers = 0
args.itr = 1
args.train_epochs = 100
args.learning_rate = 0.0001
args.patience = 5

args.des = 'exp'

args.use_gpu = True
# if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0'

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    

args.checkpoints = './checkpoints_finetune_mixdata_kgml' # location of model checkpoints

Exp = Exp_KGML

for ii in range(args.itr):
    args.ii = ii
    args.data_path = 'res_train4_test8_extract_28years_ageindependent_update_with_NEE_Ra_RECO.npz' # data file
    args.stat_path = 'data_stats_with_NEE_Ra_RECO.npz' # stat file
    args.pft_path = 'pft_dataset_12mean_28years_ageindependent_plus_ageweight_update.npz' # pft file
    setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_KGML_{}'.format(args.model, args.data,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)
    # set experiments
    exp = Exp(args)
    # train
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting=setting, stage='pretrain', paras='')
    # test             
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting= setting, stage='pretrain', paras='')


    networks = ['above',"ameriflux", "fluxnet", "icos-ww", "mix"]
    for net in networks:
        
        setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_KGML_{}_{}'.format(args.model, args.data, 
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, net, args.ii)
        args.data_path = f'res_train4_test8_extract_4types_28years_{net}.npz' # data file
        args.pft_path = f'pft_dataset_12mean_4types_28years_ESACCI_plusAW_{net}.npz' # pft file

        # set experiments
        exp = Exp(args)
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting=setting, stage='finetune', paras='')

        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting=setting, stage='finetune', paras='')


        torch.cuda.empty_cache()

