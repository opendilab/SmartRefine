import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from losses import ScoreRegL1Loss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import TargetRegion
from collections import OrderedDict

from utils import TemporalData



class Refine(pl.LightningModule):

    def __init__(self,
                 cls_temperture: int,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 rotate: bool,

                 future_steps: int,
                 num_modes: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 seg_num: int,
                 refine_num: int,
                 refine_radius: int,
                 r_lo: int,
                 r_hi: int,
                 **kwargs) -> None:
        super(Refine, self).__init__()
        self.save_hyperparameters()

        self.cls_temperture = cls_temperture

        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate

        self.refine_num = refine_num

        self.target_encoder = TargetRegion(
                                          future_steps=future_steps,
                                          num_modes=num_modes,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          refine_num=refine_num,
                                          seg_num=seg_num,
                                          refine_radius=refine_radius,
                                          r_lo=r_lo,
                                          r_hi=r_hi,
                                          **kwargs)
    

        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        self.score_loss = ScoreRegL1Loss()

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()

    def to_global_coord(self, data):
        data_angles = data['theta']
        data_rotate_angle = data['rotate_angles'][data['agent_index']]

        rotate_local = torch.empty(data['agent_index'].shape[0], 2, 2, device=self.device)
        sin_vals_angle = torch.sin(-data_rotate_angle)
        cos_vals_angle = torch.cos(-data_rotate_angle)
        rotate_local[:, 0, 0] = cos_vals_angle
        rotate_local[:, 0, 1] = -sin_vals_angle
        rotate_local[:, 1, 0] = sin_vals_angle
        rotate_local[:, 1, 1] = cos_vals_angle
        # agent to av
        data.rotate_local = rotate_local

        rotate_mat = torch.empty(data['agent_index'].shape[0], 2, 2, device=self.device)
        sin_vals = torch.sin(-data_angles)
        cos_vals = torch.cos(-data_angles)
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals
        rotate_mat[:, 1, 0] = sin_vals
        rotate_mat[:, 1, 1] = cos_vals
        # av to global
        data.rotate_mat_ = rotate_mat

        rotate_mat_ = torch.empty(data['agent_index'].shape[0], 2, 2, device=self.device)
        sin_vals = torch.sin(data_angles)
        cos_vals = torch.cos(data_angles)
        rotate_mat_[:, 0, 0] = cos_vals
        rotate_mat_[:, 0, 1] = -sin_vals
        rotate_mat_[:, 1, 0] = sin_vals
        rotate_mat_[:, 1, 1] = cos_vals
        # global to av
        data.r_rotate_mat_ = rotate_mat_

    def refine(self, data, ys_hat, embed):

        assert ys_hat.shape[-1] == 2

        y_hat_ego = ys_hat.reshape(ys_hat.shape[1]*self.num_modes, -1, 2) # n*f, t,2

        refine_y_hat, refine_pi = self.target_encoder(data, y_hat_ego, embed)

        return refine_y_hat, refine_pi

    def forward(self, data: TemporalData, p1_data=None):
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        self.to_global_coord(data)
        
        ys_hat_ego = p1_data['traj'].transpose(0,1)
        traj_embed = p1_data['embed'].transpose(0,1)

        ys_refine, pis_refine  = self.refine(data, ys_hat_ego, traj_embed)

        # concat for later laplace sigma computation.
        return torch.cat((ys_hat_ego, ys_hat_ego), -1), None, ys_refine, pis_refine

    def training_step(self, data, batch_idx):
        data, p1_data = data
        reg_mask = ~data['padding_mask'][:, -self.future_steps:]
        reg_mask_ego = ~data['padding_mask'][data.agent_index][:, -self.future_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        valid_steps_ego = reg_mask_ego.sum(dim=-1)
        cls_mask = valid_steps > 0
        cls_mask_ego = valid_steps_ego > 0

        ys_hat_ego, _, refine_y_hat_deltas, refine_pis = self(data, p1_data)

        refine_pi, refine_score = refine_pis
        y_agent = data.y[data.agent_index]

        reg_loss_refines = 0
        cls_loss_refines = 0
        score_loss_refines=0  

        max_val = (torch.norm(ys_hat_ego[..., :2] - y_agent, p=2, dim=-1) * reg_mask_ego).sum(dim=-1)
        max_val = max_val.min(0)[0]
        y_i = ys_hat_ego.clone()
        min_vals = []
        min_vals.append(max_val)
        for i in range(self.refine_num):
            y_i = y_i + refine_y_hat_deltas[i]
            l2_norm = (torch.norm(y_i[..., :2] - y_agent, p=2, dim=-1) * reg_mask_ego).sum(dim=-1)
            min_vals.append(l2_norm.min(0)[0])
        min_vals = torch.stack(min_vals)

        min_val = min_vals.min(0)[0]
        max_val = min_vals.max(0)[0]
        min_id = min_vals.min(0)[1]
        max_id = min_vals.max(0)[1]

        refine_y_hat = ys_hat_ego
        refine_score_i = refine_score[0].transpose(0,1)
        l2_norm = (torch.norm(refine_y_hat[..., :2] - y_agent, p=2, dim=-1) * reg_mask_ego).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)

        target_score_i = ((max_val - l2_norm.min(0)[0]) / ((max_val - min_val)+1e-6))
        target_score_i = torch.clamp(target_score_i,0,1)
        refine_score_i = refine_score_i[best_mode, torch.arange(data.num_graphs)]
        score_loss_refine = self.score_loss(refine_score_i, target_score_i)
        score_loss_refines += score_loss_refine

        for i in range(self.refine_num):
            refine_y_hat_i = refine_y_hat_deltas[i]
            refine_pi_i = refine_pi[i]
            refine_score_i = refine_score[i+1].transpose(0,1)

            refine_y_hat[...,:2] = refine_y_hat[...,:2] + refine_y_hat_i[...,:2]
            refine_y_hat[...,2:] = refine_y_hat_i[...,2:]

            l2_norm = (torch.norm(refine_y_hat[..., :2] - y_agent, p=2, dim=-1) * reg_mask_ego).sum(dim=-1)  # [F, N]
            best_mode = l2_norm.argmin(dim=0)
            refine_y_hat_best = refine_y_hat[best_mode, torch.arange(data.num_graphs)] # n, t, 4
            reg_loss_refine = self.reg_loss(refine_y_hat_best[reg_mask_ego], y_agent[reg_mask_ego])
            reg_loss_refines += reg_loss_refine
            
            soft_target = F.softmax((-l2_norm[:, cls_mask_ego] / valid_steps_ego[cls_mask_ego])/self.cls_temperture, dim=0).t().detach()
            cls_loss_refine = self.cls_loss(refine_pi_i[cls_mask_ego], soft_target)
            cls_loss_refines += cls_loss_refine
            
            target_score_i = ((max_val - l2_norm.min(0)[0]) / ((max_val - min_val)+1e-6))
            refine_score_i = refine_score_i[best_mode, torch.arange(data.num_graphs)]
            target_score_i = torch.clamp(target_score_i,0,1)
            score_loss_refine = self.score_loss(refine_score_i, target_score_i)
            score_loss_refines += score_loss_refine

        self.log('refine_reg_loss', reg_loss_refines/self.refine_num, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('refine_cls_loss', cls_loss_refines/self.refine_num, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('refine_score_loss', score_loss_refines/(self.refine_num+1), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)# else:
        
        loss = reg_loss_refines/self.refine_num + cls_loss_refines/self.refine_num

        loss += 0.01*(score_loss_refines)/(self.refine_num+1)

        return loss

    def validation_step(self, data, batch_idx):
        data, p1_data = data
        reg_mask = ~data['padding_mask'][data.agent_index][:, -self.future_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0

        y_hat_init_ego, _, refine_y_hat_deltas, refine_pis = self(data, p1_data)

        refine_pis, refine_scores = refine_pis

        y_agent = data.y[data.agent_index]

        max_val = (torch.norm(y_hat_init_ego[..., :2] - y_agent, p=2, dim=-1) * reg_mask).sum(dim=-1)

        y_i = y_hat_init_ego.clone()
        min_vals = []
        max_val = max_val.min(0)[0]
        min_vals.append(max_val)
        for i in range(self.refine_num):
            y_i += refine_y_hat_deltas[i]
            l2_norm = (torch.norm(y_i[..., :2] - y_agent, p=2, dim=-1) * reg_mask).sum(dim=-1)
            min_vals.append(l2_norm.min(0)[0])
        min_vals = torch.stack(min_vals)
        min_val = min_vals.min(0)[0]
        max_val = min_vals.max(0)[0]

        score_loss_refines=0
        refine_y_hat = y_hat_init_ego.clone()

        refine_score_i = refine_scores[0].transpose(0,1)
        l2_norm = (torch.norm(refine_y_hat[..., :2] - y_agent, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(0)

        target_score_i = ((max_val - l2_norm.min(0)[0]) / ((max_val - min_val)))
        refine_score_i = refine_score_i[best_mode, torch.arange(data.num_graphs)]
        score_loss_refine = self.score_loss(refine_score_i, target_score_i)
        score_loss_refines += score_loss_refine
        for i in range(self.refine_num):
            refine_y_hat[...,:2] += refine_y_hat_deltas[i][...,:2]
            refine_y_hat[...,2:] = refine_y_hat_deltas[i][...,2:]

            refine_score_i = refine_scores[i+1].transpose(0,1)
            l2_norm = (torch.norm(refine_y_hat[..., :2] - y_agent, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
            best_mode = l2_norm.argmin(0)
            target_score_i = ((max_val - l2_norm.min(0)[0]) / ((max_val - min_val)))
            refine_score_i = refine_score_i[best_mode, torch.arange(data.num_graphs)]
            score_loss_refine = self.score_loss(refine_score_i, target_score_i)
            score_loss_refines += score_loss_refine

        refine_pi = refine_pis[-1]

        l2_norm = (torch.norm(refine_y_hat[..., :2] - y_agent, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        refine_y_hat_best = refine_y_hat[best_mode, torch.arange(data.num_graphs)] # n, t, 4
        reg_loss_refine = self.reg_loss(refine_y_hat_best[reg_mask], y_agent[reg_mask])
        soft_target = F.softmax((-l2_norm[:, cls_mask] / valid_steps[cls_mask])/self.cls_temperture, dim=0).t().detach()
        cls_loss_refine = self.cls_loss(refine_pi[cls_mask], soft_target)
        self.log('val_refine_reg_loss', reg_loss_refine, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log('val_refine_cls_loss', cls_loss_refine, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log('val_refine_score_loss', score_loss_refines/(self.refine_num+1), prog_bar=False, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = refine_y_hat[..., : 2]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=False, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=False, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=False, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
  
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay)) if 'encoder_phase1' not in param_name],
             "lr": self.lr,
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay)) if 'encoder_phase1' not in param_name],
             "lr": self.lr,
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Refine')
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--local_radius', type=int, default=150)
        parser.add_argument('--cls_temperture', type=int, default=1)
        

        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, required=True)
        parser.add_argument('--seg_num', type=int, default=2)
        parser.add_argument('--refine_num', type=int, required=True)
        parser.add_argument('--refine_radius', type=int, default=-1)
        parser.add_argument('--r_lo', type=int, default=2)
        parser.add_argument('--r_hi', type=int, default=10)
        return parent_parser
