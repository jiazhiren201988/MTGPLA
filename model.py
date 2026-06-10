import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np

d_model=128
dim_feedforward = 512
n_heads = 4
vocab_size=26
n_layers=4

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src):
        output = src
        for mod in self.layers:
            output,attn = mod(output)
        if self.norm is not None:
            output = self.norm(output)
        return output,attn

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']
    def __init__(self, d_model, nhead, dim_feedforward=dim_feedforward, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=batch_first)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        src2,attn = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attn[:,0,:]
    
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None):
        y = self.self_attention_norm(x)
        kv = self.self_attention_norm(kv)
        y,att = self.self_attention(y, kv, kv, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x,att
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        #        self.gelu = GELU()
        self.gelu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        att = x

        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x,att

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

class CNN_Transformer(nn.Module):
    def __init__(self, ):
        super(CNN_Transformer, self).__init__()
        dropout = 0.3
        self.emb = nn.Embedding(65, 128, 0)
        self.self_att = EncoderLayer(128, 128, dropout, dropout, 2)

        self.smi_module = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 0),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, 1, 0),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.emb(x.long().to(self.emb.weight.device))
        x,_ = self.self_att(x,x)
        x = self.smi_module(x.permute(0, 2, 1))
        return x

    
class DeepTTG(torch.nn.Module):
    def __init__(self, n_output=1,MLP_dim=96, dropout=0.1,
                 c_feature=108,vocab_size=vocab_size,d_model =d_model,n_heads = n_heads,n_layers=n_layers):
        super(DeepTTG, self).__init__()
        # TransformerEncoder for extracting protein features
        self.protein_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.protein_transformer = TransformerEncoder(self.protein_encoder_layer, num_layers=n_layers)
        
        self.pocket_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        # self.pocket_transformer = TransformerEncoder(self.pocket_encoder_layer, num_layers=n_layers)
        
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(vocab_size, d_model),freeze=True)
        self.domain_emb = nn.Embedding(3, d_model)

        self.CNN_Transformer = CNN_Transformer()
        self.smiles_fc = nn.Linear(128, 120) 

        self.cross_attention = EncoderLayer(128, 128, dropout, dropout, 2)
        self.smiles_proj = nn.Linear(128, d_model)
        self.smiles_bn = nn.BatchNorm1d(128)
        self.cross_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # GIN model for extracting compound features
        nn1 = Sequential(Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(MLP_dim)
        nn2 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(MLP_dim)
        nn3 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(MLP_dim)
        nn4 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(MLP_dim)

        # self.fc1_c = Linear(MLP_dim, 120)
        # self.poc_fc = Linear(120, 60)

        self.fc1 = nn.Linear(128*4+96, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, self.n_output)



    def forward(self, data):
        x, edge_index , batch = data.x, data.edge_index,data.batch
        target = data.protein
        pocket = data.pocket
        smiles = data.smiles

        # Feature extraction from the compound graphs
        x =  F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x =  F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x =  F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x =  F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = global_add_pool(x, batch)
        # x =  F.relu(self.fc1_c(x))
        x = self.dropout(x)

        target = self.src_emb(target)+self.pos_emb(target)
        protein_encoded, att1 = self.protein_transformer(target)  
        # pro = protein_encoded.mean(dim=1)
        pro = protein_encoded[:,0,:] 

        pocket = self.src_emb(pocket)+self.pos_emb(pocket)
        pocket_embedded_xt, att2 = self.protein_transformer(pocket)
        # poc = pocket_embedded_xt.mean(dim=1)
        poc = pocket_embedded_xt[:, 0, :]

        smiles_cnn_feat = self.smiles_bn(self.CNN_Transformer(smiles)).permute(0, 2, 1)
        smiles_global = smiles_cnn_feat.mean(dim=1)
        cross_feat, _ = self.cross_attention(
            smiles_cnn_feat,        
            pocket_embedded_xt
            
        ) 
        cross_feat = self.cross_norm(cross_feat).mean(dim=1)
        x= torch.cat([pro,poc,cross_feat,smiles_global,x], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        
        return out,att1,att2


