import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding, DataEmbedding_noT



class Informer_noT(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer_noT, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding_noT(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding_noT(dec_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class Informer_noT_Shared(nn.Module):
    """
    Informer_noT with optional shared encoder/decoder/projection modules
    for memory-efficient multi-branch structure.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'),
                shared_enc_embedding=None, shared_encoder=None, shared_dec_embedding = None):
                # ,
                # shared_dec_embedding=None, shared_decoder=None, shared_projection=None
        super(Informer_noT_Shared, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # # Encoding
        # self.enc_embedding = DataEmbedding_noT(enc_in, d_model, dropout)
        # self.dec_embedding = DataEmbedding_noT(dec_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention

        # Shared or default encoding layers
        self.enc_embedding = shared_enc_embedding if shared_enc_embedding else DataEmbedding_noT(enc_in, d_model, dropout)
        self.encoder = shared_encoder if shared_encoder else Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.dec_embedding = shared_dec_embedding if shared_dec_embedding else DataEmbedding_noT(dec_in, d_model, dropout)

        # Decoder
        # self.dec_embedding = DataEmbedding_noT(dec_in, d_model, dropout)
        self.decoder = Decoder(
        # self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
   

class MultiBranch_AddPure_AddCom_Finetune_Mixdata_ShareStructure(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super().__init__()

        # Prepare shared components for BL/NL/GS branches
        self.shared_enc_embedding = DataEmbedding_noT(enc_in, d_model, dropout)
        self.shared_encoder = Encoder([
        EncoderLayer(
        AttentionLayer((ProbAttention if attn=='prob' else FullAttention)(False, factor, attention_dropout=dropout, output_attention=output_attention),
        d_model, n_heads, mix=False),
        d_model, d_ff, dropout=dropout, activation=activation)
        for _ in range(e_layers)
        ],
        [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
        norm_layer=torch.nn.LayerNorm(d_model))


        self.shared_dec_embedding = DataEmbedding_noT(dec_in, d_model, dropout)
        self.shared_decoder = Decoder([
        DecoderLayer(
        AttentionLayer((ProbAttention if attn=='prob' else FullAttention)(True, factor, attention_dropout=dropout, output_attention=False),
        d_model, n_heads, mix=mix),
        AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
        d_model, n_heads, mix=False),
        d_model, d_ff, dropout=dropout, activation=activation)
        for _ in range(d_layers)
        ], norm_layer=torch.nn.LayerNorm(d_model))


        self.shared_projection = nn.Linear(d_model, c_out, bias=True)

        # super(MultiBranch_noT, self).__init__()
        informer_args = (enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor, d_model, n_heads, e_layers, d_layers, d_ff, 
                dropout, attn, embed, freq, activation,
                output_attention, distil, mix, device)

        # Three shared-structure branches
        self.branch1 = Informer_noT_Shared(*informer_args,
        shared_enc_embedding=self.shared_enc_embedding,
        shared_encoder=self.shared_encoder,
        shared_dec_embedding=self.shared_dec_embedding)

        self.branch2 = Informer_noT_Shared(*informer_args,
        shared_enc_embedding=self.shared_enc_embedding,
        shared_encoder=self.shared_encoder,
        shared_dec_embedding=self.shared_dec_embedding)


        self.branch3 = Informer_noT_Shared(*informer_args,
        shared_enc_embedding=self.shared_enc_embedding,
        shared_encoder=self.shared_encoder,
        shared_dec_embedding=self.shared_dec_embedding)


        # branch0 uses PFT weights, thus extra input features and not shared
        informer_args_com = (enc_in+3, dec_in, c_out, seq_len, label_len, out_len, 
                factor, d_model, n_heads, e_layers, d_layers, d_ff, 
                dropout, attn, embed, freq, activation,
                output_attention, distil, mix, device)        
        self.branch0 = Informer_noT_Shared(*informer_args_com)

    def set_stats(self, mean, std):
        self.y_mean = mean
        self.y_std = std


    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean


    def forward(self, x_enc, x_dec, batch_pft_bl, batch_pft_nl, batch_pft_gs, batch_ws, stage, type='mix',
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        pft: Tensor of shape [B, 3] before softmax
        """
        
        # Concatenate three PFT weight variables into shape [B, L, 3]
        device = x_enc.device  # Get the device of the input tensor, typically 'cuda:0'
        pft_weight = torch.cat((batch_pft_bl, batch_pft_nl, batch_pft_gs), dim=2).to(device)
        self.y_mean = self.y_mean.to(device)
        self.y_std = self.y_std.to(device)

        if stage == 'pure':
            # Obtain raw outputs from each individual branch (BL, NL, GS)
            if type == 'BL':            
                out_BL = self.branch1(x_enc, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
            elif type == 'NL':            
                out_BL = self.branch2(x_enc, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
            if type == 'GS':            
                out_BL = self.branch3(x_enc, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask) 
                
            # out_BL = self.__norm_data__(F.relu(self.__inverse_norm_data__(out_BL, self.y_mean, self.y_std)), self.y_mean, self.y_std)
            out_denorm = self.__inverse_norm_data__(out_BL, self.y_mean, self.y_std)
            mask = torch.ones(out_denorm.shape[-1], device=out_BL.device, dtype=torch.bool)
            mask[7] = False   
            out_denorm_relu = out_denorm.clone()
            out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
            out_BL = self.__norm_data__(out_denorm_relu, self.y_mean, self.y_std)

            out_NL = None
            out_GS = None
            out_MIX = None
            pft_weight = None
            out0_Com_mix0 = None
            out0_Com_mix = None 

        elif stage == 'mix':
            # Obtain raw outputs from each branch using the mixed decoder input
            out1_BL_mix = self.branch1(x_enc, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
            out2_NL_mix = self.branch2(x_enc, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
            out3_GS_mix = self.branch3(x_enc, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)

            # Concatenate PFT weights to the encoder input: [B, L, D+3]
            x_enc_with_w = torch.cat([x_enc, pft_weight], dim=2)  # [B, L, D+3]
            # Output from competition branch with PFT weights included
            out0_Com_mix0 = self.branch0(x_enc_with_w, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
            out0_Com_mix = out0_Com_mix0 # No.1

            out_denorm = self.__inverse_norm_data__(out1_BL_mix, self.y_mean, self.y_std)
            mask = torch.ones(out_denorm.shape[-1], device=out1_BL_mix.device, dtype=torch.bool)
            mask[7] = False   
            out_denorm_relu = out_denorm.clone()
            out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
            # out1 = self.__norm_data__(out_denorm_relu, self.y_mean, self.y_std)
            out1 = out_denorm_relu

            out_denorm = self.__inverse_norm_data__(out2_NL_mix, self.y_mean, self.y_std)
            mask = torch.ones(out_denorm.shape[-1], device=out2_NL_mix.device, dtype=torch.bool)
            mask[7] = False   
            out_denorm_relu = out_denorm.clone()
            out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
            # out2 = self.__norm_data__(out_denorm_relu, self.y_mean, self.y_std)
            out2 = out_denorm_relu

            out_denorm = self.__inverse_norm_data__(out3_GS_mix, self.y_mean, self.y_std)
            mask = torch.ones(out_denorm.shape[-1], device=out3_GS_mix.device, dtype=torch.bool)
            mask[7] = False   
            out_denorm_relu = out_denorm.clone()
            out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
            # out3 = self.__norm_data__(out_denorm_relu, self.y_mean, self.y_std)
            out3 = out_denorm_relu

            # Weight each branch output by corresponding PFT weight
            weighted_out1 = out1 * pft_weight[:,:,0].unsqueeze(-1)
            weighted_out2 = out2 * pft_weight[:,:,1].unsqueeze(-1)
            weighted_out3 = out3 * pft_weight[:,:,2].unsqueeze(-1)

            # Final output: weighted sum of all branches plus the competition output
            final_out = weighted_out1 + weighted_out2 + weighted_out3 + out0_Com_mix
            
            out_denorm = final_out
            mask = torch.ones(out_denorm.shape[-1], device=final_out.device, dtype=torch.bool)
            mask[7] = False   
            out_denorm_relu = out_denorm.clone()
            out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
            out_MIX = self.__norm_data__(out_denorm_relu, self.y_mean, self.y_std)

            out_BL = out1
            out_NL = out2
            out_GS = out3

        elif stage == 'ageindep':
            # Concatenate PFT weights to the encoder input: [B, L, D+3]
            x_enc_with_w = torch.cat([x_enc, pft_weight], dim=2)  # [B, L, D+3]
            # Output from competition branch with PFT weights included
            out0_Com_mix0 = self.branch0(x_enc_with_w, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
            out0_Com_mix = out0_Com_mix0 # No.4

            # Final output: weighted sum of all branches plus the competition output
            final_out = batch_ws + out0_Com_mix
            
            out_denorm = final_out
            mask = torch.ones(out_denorm.shape[-1], device=final_out.device, dtype=torch.bool)
            mask[7] = False   
            out_denorm_relu = out_denorm.clone()
            out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])
            out_MIX = self.__norm_data__(out_denorm_relu, self.y_mean, self.y_std)

            pft_weight = batch_ws
            out_BL = None
            out_NL = None
            out_GS = None

        # , out1, out2, out3, pft_weight
        return out_BL, out_NL, out_GS, out_MIX, [pft_weight ,out0_Com_mix,out0_Com_mix0]
    

class OneBranch_ED_ALLAGE_PFT_Prediction(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                age_embed_dim=32,
                device=torch.device('cuda:0')):
        super().__init__()
        # super(MultiBranch_noT, self).__init__()
        enc_in_total = enc_in + age_embed_dim
        informer_args = (enc_in_total, dec_in, c_out, seq_len, label_len, out_len, 
                factor, d_model, n_heads, e_layers, d_layers, d_ff, 
                dropout, attn, embed, freq, activation,
                output_attention, distil, mix, device)

        self.branch1 = Informer_noT(*informer_args)
        self.age_embed = nn.Embedding(18, age_embed_dim)

    def set_stats(self, mean, std):
        self.y_mean = mean
        self.y_std = std


    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean


    def forward(self, x_enc, x_dec, stage,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        pft: Tensor of shape [B, 3] before softmax
        """
        
        # Concatenate three PFT weight variables into shape [B, L, 3]
        device = x_enc.device  # Get the device of the input tensor, typically 'cuda:0'
        B = x_enc.shape[0]
        if stage == 'step1':
            # shape: [B, T, D] → repeat 18 times
            x_enc = x_enc.repeat_interleave(18, dim=0)        # [B*18, T, D]
            age_ids = torch.arange(18).repeat(B).to(device)   # [B*18]
            age_emb = self.age_embed(age_ids)                 # [B*18, age_dim]
            age_emb = age_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)  # [B*18, T, age_dim]
            x_enc = torch.cat([x_enc, age_emb], dim=-1)       # [B*18, T, D+age_dim]

            x_dec = x_dec[:, 0:18, :, :]  # [B, 18, T, 3]
            x_dec = x_dec.reshape(B * 18, *x_dec.shape[2:])  # [B*18, T, 3]

            out_all = self.branch1(x_enc, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)  # [B*18, T, 3]
            out_all = out_all.reshape(B, 18, *out_all.shape[1:])  # [B, 18, T, 3]


            out_all = F.sigmoid(out_all)
            out_all_sum = out_all.sum(dim=3, keepdim=True) + 1e-12 
            out_all = out_all / out_all_sum


        elif stage == 'step2':
            data_y_ESACCI = x_dec[1] # [8, 29, 3]
            # data_y_ESACCI = data_y_ESACCI.unsqueeze(1).repeat(1, 18, 1, 1)  # shape: (N, 18, T, D)
            data_aw = x_dec[2] # (8, 18)
            x_dec = x_dec[0] # [8, 18, 29, 3] 

            # shape: [B, T, D] → repeat 18 times
            x_enc = x_enc.repeat_interleave(18, dim=0)        # [B*18, T, D]
            age_ids = torch.arange(18).repeat(B).to(device)   # [B*18]
            age_emb = self.age_embed(age_ids)                 # [B*18, age_dim]
            age_emb = age_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)  # [B*18, T, age_dim]
            x_enc = torch.cat([x_enc, age_emb], dim=-1)       # [B*18, T, D+age_dim]

            x_dec = data_y_ESACCI.repeat_interleave(18, dim=0)  # [B, T, 3] to [B*18, T, 3]
            # x_dec = x_dec.reshape(B * 18, *x_dec.shape[2:])  # [B*18, T, 3]

            out_all = self.branch1(x_enc, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)  # [B*18, T, 3]
            out_all = out_all.reshape(B, 18, *out_all.shape[1:])  # [B, 18, T, 3]


            out_all = F.sigmoid(out_all)
            out_all_sum = out_all.sum(dim=3, keepdim=True) + 1e-12  
            out_all = out_all / out_all_sum

            data_aw = data_aw.to(x_enc.device)

            data_aw_expanded = data_aw.unsqueeze(-1).unsqueeze(-1)  # shape: [8, 18, 1, 1]
            weighted_out = out_all * data_aw_expanded  # [8, 18, 28, 3]
            out_final = weighted_out.sum(dim=1)  # shape: [8, 28, 3]
            out_all = [out_final, out_all]

        # , out1, out2, out3, pft_weight
        return out_all
  

class OneBranch(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super().__init__()
        # super(MultiBranch_noT, self).__init__()
        informer_args = (enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor, d_model, n_heads, e_layers, d_layers, d_ff, 
                dropout, attn, embed, freq, activation,
                output_attention, distil, mix, device)
        self.branch1 = Informer_noT(*informer_args)

    def set_stats(self, mean, std):
        self.y_mean = mean
        self.y_std = std


    def __norm_data__(self, data, mean, std):
        return (data-mean)/(std+1e-10)
    
    def __inverse_norm_data__(self, data, mean, std):
        return data*(std+1e-10) + mean


    def forward(self, x_enc, x_dec, stage,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        pft: Tensor of shape [B, 3] before softmax
        """

        device = x_enc.device 
        self.y_mean = self.y_mean.to(device)
        self.y_std = self.y_std.to(device)

        out1_raw = self.branch1(x_enc, x_dec, enc_self_mask, dec_self_mask, dec_enc_mask)


        out_denorm = self.__inverse_norm_data__(out1_raw, self.y_mean, self.y_std)
        mask = torch.ones(out_denorm.shape[-1], device=out1_raw.device, dtype=torch.bool)
        mask[7] = False   
        out_denorm_relu = out_denorm.clone()
        out_denorm_relu[..., mask] = F.relu(out_denorm_relu[..., mask])

        final_out = self.__norm_data__(out_denorm_relu, self.y_mean, self.y_std)

        return final_out


