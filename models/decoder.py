from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


class GRUDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(GRUDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.repeat(self.num_modes, 1).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
            return torch.cat((loc, scale),
                             dim=-1).view(self.num_modes, -1, self.future_steps, 4), pi  # [F, N, H, 4], [N, F]
        else:
            return loc.view(self.num_modes, -1, self.future_steps, 2), pi  # [F, N, H, 2], [N, F]


class MLPDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(MLPDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        out = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale  # [F, N, H, 2]
            return torch.cat((loc, scale), dim=-1), pi, out  # [F, N, H, 4], [N, F]
        else:
            return loc, pi  # [F, N, H, 2], [N, F]


class MLPDeltaDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 with_cumsum:int=0,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(MLPDeltaDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale
        self.with_cumsum = False if with_cumsum==0 else True

        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2))
        # self.pi = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.LayerNorm(self.hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pi = self.pi(global_embed).squeeze(-1).t()
        loc = self.loc(global_embed).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        if self.uncertain:
            if not self.with_cumsum:
                scale = F.elu_(self.scale(global_embed), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            # scale = F.elu_(self.scale(global_embed), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2)
                scale = scale + self.min_scale  # [F, N, H, 2]
            else:
                # only to (0,+inf)
                scale = F.elu_(self.scale(global_embed), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            return torch.cat((loc, scale), dim=-1)  # [F, N, H, 4], [N, F]
        else:
            return loc  # [F, N, H, 2], [N, F]



class MLPDeltaDecoderPi(nn.Module):

    def __init__(self,
                 embed_dim:int) -> None:
        super(MLPDeltaDecoderPi, self).__init__()
        self.hidden_size = embed_dim

        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(global_embed).squeeze(-1).t()
        return pi



class MLPDeltaDecoderScore(nn.Module):

    def __init__(self,
                 embed_dim:int, 
                 with_last:bool=False) -> None:
        super(MLPDeltaDecoderScore, self).__init__()
        self.hidden_size = embed_dim
        self.with_last = with_last
        if not self.with_last:
            self.pi = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 1),
                # nn.Sigmoid()
                nn.Tanh()
            )
            # cross entropy
        else:
            self.pi = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()
            )
        self.apply(init_weights)

    def forward(self,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(global_embed).squeeze(-1).transpose(0,1)
        # 0, 1
        pi = (pi+1)/2
        return pi