import torch
import torch.nn as nn
from scipy.interpolate import interp1d


class SynthMy(nn.Module):
    def __init__(self, in_dims, out_dims, reduce_factor=4):  # L, emb
        super().__init__()

        reduced_dims = out_dims // reduce_factor
        self.dense = nn.Linear(in_dims, reduced_dims)
        self.reduce_factor = reduce_factor
        self.se = SELayer(out_dims, r=4)
        self.value_reduce = nn.Linear(out_dims, reduced_dims)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        energy = self.dense(x)  # b, channel, reduced_dim
        energy = self.se(energy)

        attention = self.softmax(energy)
        value = x
        if self.reduce_factor > 1:
            value = self.value_reduce(value.transpose(1, 2)).transpose(1, 2)  # b, reduced_dim, t

        out = torch.bmm(attention, value)
        return out


class SynthMix(nn.Module):
    def __init__(self, temporal_dim, channel_dim, proj_drop):  # L, emb
        super().__init__()
        self.synth_token = SynthMy(channel_dim, temporal_dim)  # reduce factor for window # l, emb -> l, l * l, emb

        self.proj = nn.Linear(channel_dim, channel_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.synth_token(x)  # b, t, c
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, r=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # batch, channel
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SynthEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout, temporal_dim):
        super().__init__()
        self.synth_att = SynthMix(temporal_dim=temporal_dim, channel_dim=d_model, proj_drop=dropout)
        self.ff = Mlp(in_features=d_model, hidden_features=d_model * expansion_factor, out_features=d_model,
                      drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.synth_att(x))
        x = self.norm2(x + self.ff(x))
        return x


class SynthEncoder(nn.TransformerEncoder):
    def __init__(self, d_model=128, expansion_factor=2, dropout=0.1, num_layers=5, window_size=11):
        encoder_layer = SynthEncoderLayer(d_model=d_model, expansion_factor=expansion_factor, dropout=dropout,
                                          temporal_dim=window_size)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MySynth(nn.Module):
    def __init__(self,
                 sample_interval=10,
                 step=1,
                 joint_dim=15 * 2,
                 emb_dim=128,
                 num_encoder_layers=5,
                 expansion_factor=2,
                 dropout=0.1,
                 device=torch.device("cuda")):
        super().__init__()
        self.sample_interval = sample_interval
        self.step = step
        self.window_size = sample_interval * step + 1

        self.encoder_emb = nn.Linear(joint_dim, emb_dim)

        self.encoder = SynthEncoder(d_model=emb_dim,
                                    expansion_factor=expansion_factor,
                                    dropout=dropout,
                                    num_layers=num_encoder_layers,
                                    window_size=self.window_size)

        self.norm = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, joint_dim)
        self.device = device

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_seq):

        input_seq = input_seq.to(torch.float32)

        choose_index = range(0, self.window_size, self.sample_interval)
        interp_f = interp1d(choose_index,input_seq[:, choose_index, :].cpu().numpy(),kind="quadratic",  # "cubic"
                            axis=1, copy=True, bounds_error=None, fill_value="extrapolate", assume_sorted=False)
        interp_poses = torch.tensor(interp_f(range(0, self.window_size))).to(self.device)

        interp_poses = interp_poses.to(torch.float32)
        interp = interp_poses.clone()

        src = self.encoder_emb(interp_poses)
        mem = self.encoder.forward(src)
        mem = self.norm(mem)
        recover = self.fc(mem) + interp

        return recover
