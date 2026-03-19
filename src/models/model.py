import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from spikingjelly.activation_based import functional as spk_functional
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import IFNode


class TemporalSpike(nn.Module):
    """Apply IF spiking over synthetic time steps and collapse to analog features."""

    def __init__(self, timestep=4):
        super().__init__()
        self.timestep = timestep
        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        spk_functional.set_step_mode(self, step_mode="m")

    def forward(self, x):
        # x: [B, L, D]
        if hasattr(self.neuron, "reset"):
            self.neuron.reset()
        x_seq = x.unsqueeze(0).repeat(self.timestep, 1, 1, 1)  # [T,B,L,D]
        s = self.neuron(x_seq)
        coef = torch.linspace(1.0, 0.5, self.timestep, device=x.device).view(self.timestep, 1, 1, 1)
        return (s * coef).sum(dim=0) / coef.sum()


class SNNRidgeRegression(nn.Module):
    """Subject-specific ridge-style linear projection with SNN activation."""

    def __init__(self, input_sizes, hidden_dim, timestep=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linears = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for in_dim in input_sizes])
        self.norm = nn.LayerNorm(hidden_dim)
        self.spike = TemporalSpike(timestep=timestep)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x, subj_idx):
        # x: [B, 1, voxel_dim]
        h = self.linears[subj_idx](x[:, 0]).unsqueeze(1)  # [B,1,H]
        h = self.spike(self.norm(h))
        return h


class SpikingResidualMLP(nn.Module):
    """Residual MLP block over feature dimension with SNN gating."""

    def __init__(self, hidden_dim, drop=0.15, mlp_ratio=2.0, timestep=4):
        super().__init__()
        mlp_hidden = int(hidden_dim * mlp_ratio)

        self.norm = nn.LayerNorm(hidden_dim)
        self.spike = TemporalSpike(timestep=timestep)
        self.fc1 = nn.Linear(hidden_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden_dim)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        # x: [B,S,H]
        h = self.spike(self.norm(x))
        h = self.fc2(self.drop(F.gelu(self.fc1(h))))
        return x + h


class SpikingTokenMixer(nn.Module):
    """Residual MLP block over sequence dimension with SNN gating."""

    def __init__(self, seq_len, drop=0.15, mlp_ratio=2.0, timestep=4):
        super().__init__()
        mix_hidden = int(seq_len * mlp_ratio)

        self.norm = nn.LayerNorm(seq_len)
        self.spike = TemporalSpike(timestep=timestep)
        self.fc1 = nn.Linear(seq_len, mix_hidden)
        self.fc2 = nn.Linear(mix_hidden, seq_len)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        # x: [B,S,H]
        y = x.permute(0, 2, 1)  # [B,H,S]
        y = self.spike(self.norm(y))
        y = self.fc2(self.drop(F.gelu(self.fc1(y))))
        y = y + x.permute(0, 2, 1)
        return y.permute(0, 2, 1)


class SharedSNNFeatureAE(nn.Module):
    """
    Shared AE that maps [B,1,hidden_dim] -> [B,fmri_feature_seq_len,fmri_feature_dim].
    """

    def __init__(
        self,
        hidden_dim,
        fmri_feature_dim=1664,
        fmri_feature_seq_len=256,
        enc_depth=2,
        dec_depth=2,
        drop=0.15,
        timestep=4,
    ):
        super().__init__()
        self.fmri_feature_seq_len = fmri_feature_seq_len
        self.fmri_feature_dim = fmri_feature_dim

        self.seed_proj = nn.Linear(hidden_dim, fmri_feature_dim)
        self.feature_queries = nn.Parameter(torch.zeros(1, fmri_feature_seq_len, fmri_feature_dim))

        self.enc_hidden = nn.ModuleList(
            [SpikingResidualMLP(fmri_feature_dim, drop=drop, timestep=timestep) for _ in range(enc_depth)]
        )
        self.enc_token = nn.ModuleList(
            [SpikingTokenMixer(fmri_feature_seq_len, drop=drop, timestep=timestep) for _ in range(enc_depth)]
        )

        # Middle feature bridge (between encoder and decoder), keeps [B, Sf, F].
        self.mid_norm = nn.LayerNorm(fmri_feature_dim)
        self.mid_spike = TemporalSpike(timestep=timestep)

        self.dec_hidden = nn.ModuleList(
            [SpikingResidualMLP(fmri_feature_dim, drop=drop, timestep=timestep) for _ in range(dec_depth)]
        )
        self.dec_token = nn.ModuleList(
            [SpikingTokenMixer(fmri_feature_seq_len, drop=drop, timestep=timestep) for _ in range(dec_depth)]
        )

        # Decode features back to one hidden token for subject-specific recon head.
        self.out_norm = nn.LayerNorm(fmri_feature_dim)
        self.out_spike = TemporalSpike(timestep=timestep)
        self.feature_to_hidden = nn.Linear(fmri_feature_dim, hidden_dim)
        self.hidden_refine = SpikingResidualMLP(hidden_dim, drop=drop, timestep=timestep)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        init.trunc_normal_(self.feature_queries, std=0.02)

    def forward(self, x):
        # x: [B,1,hidden_dim]
        seed = self.seed_proj(x[:, 0]).unsqueeze(1)  # [B,1,F]
        h = seed + self.feature_queries              # [B,Sf,F]

        for hb, tb in zip(self.enc_hidden, self.enc_token):
            h = hb(h)
            h = tb(h)

        # feature between encoder and decoder
        middle_feature = self.mid_spike(self.mid_norm(h))
        h = middle_feature

        for hb, tb in zip(self.dec_hidden, self.dec_token):
            h = hb(h)
            h = tb(h)

        h = self.out_spike(self.out_norm(h)).mean(dim=1, keepdim=True)  # [B,1,F]
        h = self.feature_to_hidden(h)                                    # [B,1,H]
        decoded_hidden = self.hidden_refine(h)                           # [B,1,H]
        return decoded_hidden, middle_feature


class FMRIModel(nn.Module):
    """
    SNN fMRI model:
    1) SNN ridge regression: voxel -> hidden token
    2) Shared SNN feature AE: hidden -> fmri features (intermediate)
    3) Shared SNN decoder: fmri features -> hidden token
    4) Subject-specific recon head: hidden -> voxel
    """

    def __init__(
        self,
        voxel_dims,
        hidden_dim=4096,
        fmri_feature_dim=1664,
        fmri_feature_seq_len=256,
        timestep=4,
        ae_enc_depth=2,
        ae_dec_depth=2,
        drop=0.15,
    ):
        super().__init__()
        self.num_subjects = len(voxel_dims)
        self.hidden_dim = hidden_dim
        self.fmri_feature_dim = fmri_feature_dim
        self.fmri_feature_seq_len = fmri_feature_seq_len

        self.ridge = SNNRidgeRegression(
            input_sizes=voxel_dims,
            hidden_dim=hidden_dim,
            timestep=timestep,
        )

        self.shared_ae = SharedSNNFeatureAE(
            hidden_dim=hidden_dim,
            fmri_feature_dim=fmri_feature_dim,
            fmri_feature_seq_len=fmri_feature_seq_len,
            enc_depth=ae_enc_depth,
            dec_depth=ae_dec_depth,
            drop=drop,
            timestep=timestep,
        )

        self.recon_heads = nn.ModuleList([nn.Linear(hidden_dim, vdim) for vdim in voxel_dims])
        self._init_recon_heads()

    def _init_recon_heads(self):
        for m in self.recon_heads:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x, subj_idx, returnFeatures=False):
        # x: [B,1,voxel_dim] or [B,voxel_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        subj_tokens = self.ridge(x, subj_idx)                       # [B,1,H]
        decoded_tokens, middle_feature = self.shared_ae(subj_tokens) # [B,1,H], [B,Sf,F]
        recon = self.recon_heads[subj_idx](decoded_tokens[:, 0])     # [B,V]

        if returnFeatures:
            features = middle_feature.detach()
            return recon.unsqueeze(1), features

        return recon.unsqueeze(1)
