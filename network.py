from torch import nn
from torch.nn.functional import normalize
import torch
from torch.nn import MultiheadAttention
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src):
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)  # ✅ 获取注意力权重
        src = self.norm1(src + attn_output)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + ff_output)
        return src, attn_weights 
class SGDVA(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, device):
        super(SGDVA, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        self.view = view
        self.TransformerEncoderLayer = TransformerEncoderLayerWithAttn(d_model=low_feature_dim, nhead=1, dim_feedforward=256)
    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return  xrs, zs, hs
    def SGDVA(self, xs,epoch,temperature,warmup_epochs):
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
        commonz = torch.stack(zs, 1)
        aggregated_features,S = self.TransformerEncoderLayer(commonz)
        weight = get_view_importance_from_attention(S,temperature)
        if(epoch >= warmup_epochs):
            commonz = (weight.unsqueeze(-1) * aggregated_features).sum(dim=1)
        else:
            commonz = aggregated_features.mean(dim=1)

        commonz = normalize(self.Common_view(commonz), dim=1)

        with torch.no_grad():
            temp = max(commonz.size(1) ** 0.5, 1.0) 
            sim_matrix = torch.matmul(commonz, commonz.t()) / (temp + 1e-8)
            S_sample_similarity = torch.softmax(sim_matrix, dim=1)
        return commonz, S_sample_similarity,weight

def get_view_importance_from_attention(S, temperature=1):
    view_importance = S.sum(dim=1) # -> [batch_size, num_views]
    return F.softmax(view_importance / temperature, dim=1)