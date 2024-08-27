import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MLPDecoder, MLPDeltaDecoder, MLPDeltaDecoderPi, MLPDeltaDecoderScore
from models.local_encoder import ALEncoder, ALEncoderWithAo
from itertools import permutations
from utils import TemporalData
from utils import DistanceDropEdge
from torch_geometric.utils import subgraph
from itertools import product
import numpy as np
from utils import init_weights
from torch_geometric.utils import dense_to_sparse
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch


class TargetRegion(nn.Module):

    def __init__(self,
                 future_steps: int,
                 num_modes: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 refine_num: int,
                 seg_num: int,
                 refine_radius: int,
                 r_lo: int,
                 r_hi: int,
                 **kwargs) -> None:
        super(TargetRegion, self).__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps        
        self.embed_dim = embed_dim
        self.radius = refine_radius
        if self.radius == -1:
            self._radius = [0.8, 0.8*1/2, 0.8*1/4, 0.8*1/8, 0.8*1/16]
        self.refine_num = refine_num
        self.seg_num = seg_num
        self.r_lo = r_lo
        self.r_hi = r_hi

        assert embed_dim == 64

        fc_module = []
        #! 128 for hivt
        fc_module.append(nn.Linear(128, embed_dim))
        self.fc_encoder = nn.Sequential(*fc_module)

        fusion_module = []
        for i in range(self.seg_num):
            fusion_module.append(ALEncoderWithAo(node_dim=node_dim,
                                        edge_dim=edge_dim,
                                        embed_dim=embed_dim))
        self.target_al_encoder = nn.Sequential(*fusion_module)

        dec_module = []
        dec_module.append(MLPDeltaDecoder(local_channels = embed_dim,
                                        global_channels = embed_dim,
                                        future_steps = future_steps//self.seg_num,   # cut to chunk
                                        num_modes = num_modes,
                                        with_cumsum=0))
        self.refine_decoder = nn.Sequential(*dec_module)

        dec_pi_module = []
        dec_pi_module.append(MLPDeltaDecoderPi(embed_dim=embed_dim,))
        self.refine_pi_decoder = nn.Sequential(*dec_pi_module)

        self.pos_embed = nn.Parameter(torch.zeros(self.refine_num+1, 1, embed_dim))

        score_module = []
        score_module.append(nn.GRU(input_size=embed_dim,hidden_size=embed_dim))
        score_module.append(MLPDeltaDecoderScore(embed_dim=embed_dim, with_last=False))
        self.refine_score_decoder = nn.Sequential(*score_module)

        self.apply(init_weights)


    def forward(self, data: TemporalData, y_hat, ego_embed):

        y_hat_init = y_hat
        
        rotate_local_modes = data.rotate_local.repeat(self.num_modes, 1, 1)
        data_local_origin_modes = data.positions[data['agent_index'], 19, :].repeat(self.num_modes, 1)
        
        num_ego = data.agent_index.shape[0]
        new_agent_index = torch.arange(data.agent_index.shape[0]*self.num_modes).to(ego_embed.device) # n*f f1 f2 ... fn

        mask_dst = torch.ones((num_ego, self.num_modes)).to(ego_embed.device).bool()
        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]
        
        tar_lane_positions = data.tar_lane_positions
        tar_lane_vectors = data.tar_lane_vectors
        tar_is_intersections = data.tar_is_intersections
        tar_turn_directions = data.tar_turn_directions
        tar_traffic_controls = data.tar_traffic_controls
        
        trajs = []
        pis = []
        scores = []
        embeds = []

        ego_embed = self.fc_encoder[0](ego_embed)

        ego_embed = ego_embed.reshape(self.num_modes*num_ego, -1)
        score = self.refine_score_decoder[0]((ego_embed.unsqueeze(0)))[0][-1]
        score = self.refine_score_decoder[1](score.reshape(self.num_modes, num_ego, -1)+self.pos_embed[:1])
        embeds.append(ego_embed.detach())
        ego_embed = ego_embed.reshape(self.num_modes, num_ego, -1)
        scores.append(score)

        for refine_iter in range(self.refine_num):

            if refine_iter == 0:
                y_hat_agent_cord = y_hat_init.clone()
                y_hat = torch.bmm(y_hat_init, rotate_local_modes)+data_local_origin_modes.unsqueeze(1)
            else:
                y_hat_init = y_hat_init + y_hat_delta
                y_hat_agent_cord = y_hat_init.clone()
                y_hat = torch.bmm(y_hat_init, rotate_local_modes)+data_local_origin_modes.unsqueeze(1)

            # argo predict 30 timesteps
            if self.seg_num == 1:
                idx = [-1]
            elif self.seg_num == 2:
                idx = [-16, -1]
            elif self.seg_num == 3:
                idx = [-21, -11, -1]
            elif self.seg_num == 5:
                idx = [-25, -19, -13, -7, -1]
            elif self.seg_num == 6:
                idx = [-26, -21, -16, -11, -6, -1]
            else:
                assert False

            target_hats = [y_hat[:, id].reshape(self.num_modes, num_ego, -1) for id in idx]

            refine_cum_sum = []

            for tar_id, target_hat in enumerate(target_hats):

                ego_embed = ego_embed.reshape(self.num_modes*num_ego, -1)

                tar_index = []
                split_len = 0
                tar_lane_actor_vectors = []
                for i, tar_lane_point_num in enumerate(data.tar_lane_points_num): # batch
                    num_point = tar_lane_point_num
                    index_lo, index_hi = split_len, split_len + num_point
                    tar_lane_positions_i = tar_lane_positions[index_lo:index_hi]

                    tar_lane_actor_vectors_i = \
                        tar_lane_positions_i.repeat_interleave(self.num_modes, dim=0) - target_hat[:,i].repeat(tar_lane_positions_i.size(0), 1)

                    index_this = [i+j*num_ego for j in range(self.num_modes)]
                    index_i = torch.cartesian_prod(torch.arange(index_lo, index_hi).long().to(ego_embed.device), new_agent_index[index_this].long())

                    tar_index.append(index_i)
                    tar_lane_actor_vectors.append(tar_lane_actor_vectors_i) # p*f 
                    
                    split_len = index_hi

                tar_lane_actor_index = torch.cat(tar_index).t().contiguous().to(ego_embed.device)

                tar_lane_actor_vectors = torch.cat(tar_lane_actor_vectors).to(ego_embed.device)

                #! use api
                # pos_m = data.positions[:,seg_end-1]
                # pos_m = y_hat
                # num_batch = num_ego
                # batch_x = torch.tensor([i for i in range(num_batch)]).to(ego_embed.device)
                # batch_x = torch.cat([batch_x for i in range(self.num_modes)], dim=0)
                # batch_y = []
                # for i, n_p in enumerate(data.tar_lane_points_num):
                #     batch_y += [i]*n_p
                # batch_y = torch.tensor(batch_y, dtype=torch.int64).to(batch_x.device)
                # # batch_y = torch.cat([batch_y + i*num_batch for i in range(self.num_modes)], dim=0)
                
                # lane_positions = data.tar_lane_positions

                # batch_x = batch_x.repeat(30)
                
                # edge_index_pt2m = radius(
                #     x=pos_m.transpose(0,1).reshape(-1,2),
                #     # x=pos_m[:,-1],
                #     y=lane_positions,
                #     r=10,
                #     batch_x=batch_x if isinstance(data, Batch) else None,
                #     batch_y=batch_y if isinstance(data, Batch) else None,
                #     max_num_neighbors=300)

                # edge_index_pt2m[1] = edge_index_pt2m[1] % (pos_m.shape[0])
                # edge_index_pt2m = torch.unique(edge_index_pt2m, dim=1)
                # edge_attr_pt2m = lane_positions[edge_index_pt2m[0]] - pos_m[:,-1][edge_index_pt2m[1]]
                # tar_lane_actor_index = edge_index_pt2m
                # tar_lane_actor_vectors = edge_attr_pt2m


                if self.radius == -1:
                    dis_prefix = torch.cat((torch.zeros(self.num_modes*num_ego, 1, 2).to(ego_embed.device), y_hat_agent_cord[:,:-1]), dim=1) 
                    dis = torch.norm(y_hat_agent_cord-dis_prefix,dim=-1).sum(-1)
                     
                    dis = dis*self._radius[refine_iter]
                    
                    dis[dis<self.r_lo] = self.r_lo
                    dis[dis>self.r_hi] = self.r_hi
                    dis_this = dis[tar_lane_actor_index[1,:]]
                    mask = torch.norm(tar_lane_actor_vectors, p=2, dim=-1) < dis_this
                else:
                    mask = torch.norm(tar_lane_actor_vectors, p=2, dim=-1) < self.radius

                tar_lane_actor_index = tar_lane_actor_index[:, mask]
                tar_lane_actor_vectors = tar_lane_actor_vectors[mask]

                vec_ao = data_local_origin_modes - target_hat.reshape(self.num_modes*num_ego, -1)

                rotate_mat_ego = data.rotate_mat[data.agent_index]
                rotate_mat_ego = rotate_mat_ego.repeat(self.num_modes, 1, 1)

                theta_now = torch.atan2(target_hat.reshape(self.num_modes*num_ego, -1)[..., 1:2] - y_hat[:,idx[tar_id]-1,1:2],
                                        target_hat.reshape(self.num_modes*num_ego, -1)[..., 0:1] - y_hat[:,idx[tar_id]-1,:1])
                rotate_mat_tar = torch.cat(
                    (
                        torch.cat((torch.cos(theta_now), -torch.sin(theta_now)), -1).unsqueeze(-2),
                        torch.cat((torch.sin(theta_now), torch.cos(theta_now)), -1).unsqueeze(-2)
                    ),
                    -2
                )

                rotate_mat_ego = rotate_mat_tar.reshape(self.num_modes*num_ego, 2, 2)
                
                ego_embed = self.target_al_encoder[tar_id](x=(tar_lane_vectors, ego_embed),
                                                edge_index=tar_lane_actor_index,
                                                edge_attr=tar_lane_actor_vectors,
                                                is_intersections=tar_is_intersections,
                                                turn_directions=tar_turn_directions,
                                                traffic_controls=tar_traffic_controls,
                                                vec_ao=vec_ao,
                                                rotate_mat=rotate_mat_ego)


                refine_y_hat_delta = self.refine_decoder[0](ego_embed + self.pos_embed[refine_iter+1])
                refine_cum_sum.append(refine_y_hat_delta)

            ego_embed = ego_embed.reshape(self.num_modes, num_ego, -1)

            refine_y_hat_delta = torch.cat(refine_cum_sum, dim=-2).view(self.num_modes, num_ego, self.future_steps, 4)

            refine_pi = self.refine_pi_decoder[0](ego_embed + self.pos_embed[refine_iter+1:refine_iter+2])
            pis.append(refine_pi)

            ego_embed = ego_embed.reshape(self.num_modes*num_ego, -1)
            embeds_before = torch.stack(embeds, 0)
            score_input = torch.cat((embeds_before, ego_embed.unsqueeze(0)), 0)
            score = self.refine_score_decoder[0](score_input)[0][-1]
            score = self.refine_score_decoder[1](score.reshape(self.num_modes, num_ego, -1)+self.pos_embed[refine_iter+1:refine_iter+2])
            embeds.append(ego_embed.detach())
            ego_embed = ego_embed.reshape(self.num_modes, num_ego, -1)
            scores.append(score)

            ego_embed = ego_embed.detach()
            y_hat_delta = refine_y_hat_delta.reshape(self.num_modes*num_ego, -1, 4)[...,:2].detach()

            trajs.append(refine_y_hat_delta)

        ret_pis = pis, scores
        return trajs, ret_pis
