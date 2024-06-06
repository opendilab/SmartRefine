import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData

import pickle

class ArgoverseV1Dataset(Dataset):

    def __init__(self,
                 data_root: str,
                 p1_root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius

        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        
        self.data_root = data_root
        self.p1_root = p1_root
        self._raw_file_names = os.listdir(self.raw_dir)

        self._processed_file_names = [os.path.splitext(f)[0] + '.pkl' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]

        self._p1_paths = [os.path.join(self.p1_root, self._directory, f) for f in self._processed_file_names]

        super(ArgoverseV1Dataset, self).__init__(data_root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.data_root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            data = process_argoverse(self._split, raw_path, am, self._local_radius)
            with open(os.path.join(self.processed_dir, str(data['seq_id']) + '.pkl'), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        with open(self.processed_paths[idx], 'rb') as handle:
            data = pickle.load(handle)
            data = Data.from_dict(data)
        with open(self._p1_paths[idx], 'rb') as handle:
            p1_data = pickle.load(handle)
        return data, p1_data


def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float) -> Dict:
    df = pd.read_csv(raw_path)

    # filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)

    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])

    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]

    # make the scene centered at AV
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    # initialization
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    positions_global = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
    rotate_angles_global = torch.zeros(num_nodes, dtype=torch.float)

    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        positions_global[node_idx, node_steps] = xy
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
            heading_vector_global = positions_global[node_idx, node_historical_steps[-1]] - positions_global[node_idx, node_historical_steps[-2]]
            rotate_angles_global[node_idx] = torch.atan2(heading_vector_global[1], heading_vector_global[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True
    
    positions = x.clone()
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1: 20] - x[:, : 19])
    x[:, 0] = torch.zeros(num_nodes, 2)
    
    agent_pos = torch.tensor([agent_df[19]['X'], agent_df[19]['Y']], dtype=torch.float).reshape(1, 2)
    agent_ind =  [agent_index]
    (tar_lane_positions, tar_lane_vectors, tar_is_intersections, tar_turn_directions, tar_traffic_controls, tar_id_2_idx, tar_counts, tar_len_counts) = \
                    get_lane_features_preload(am,
                                             agent_ind,
                                             agent_pos,
                                             origin,
                                             rotate_mat,
                                             city,
                                             radius)

    y = None if split == 'test' else x[:, 20:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]
 
    return {
        'x': x[:, :20],
        'positions': positions,  # [N, 50, 2]
        'positions_global': positions_global,
        'edge_index': edge_index,
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'rotate_angles': rotate_angles,  # [N]  # av->agent
        'rotate_angles_global': rotate_angles_global, # global->agent
        
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,

        #! all in av' coord
        'tar_lane_positions': tar_lane_positions, # [L_, 2]
        'tar_lane_vectors': tar_lane_vectors,  # [L_, 2]
        'tar_is_intersections': tar_is_intersections,  # [L_]
        'tar_turn_directions': tar_turn_directions,  # [L_]
        'tar_traffic_controls': tar_traffic_controls,  # [L_]
        'tar_lane_points_num': sum(tar_counts),
    }


def get_lane_features_preload(am: ArgoverseMap,
                      node_inds: List[int], # node index: origin coord
                      node_positions: torch.Tensor, # query place: origin coord
                      origin: torch.Tensor, # origin: origin coord
                      rotate_mat: torch.Tensor, # rotate_mat
                      city: str,    # city: str
                      radius: float # radius: int
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    counts = []
    id_2_idx = {}
    for node_position in node_positions:
        # in range radius
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    # relative pos
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for i, lane_id in enumerate(lane_ids):
        id_2_idx[f'{lane_id}'] = i
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)

        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        counts.append(count)
        # braod to all point
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    
    lane_positions = torch.cat(lane_positions, dim=0)   # ok
    lane_vectors = torch.cat(lane_vectors, dim=0)       # ok
    is_intersections = torch.cat(is_intersections, dim=0)   # ok
    turn_directions = torch.cat(turn_directions, dim=0)     # ok
    traffic_controls = torch.cat(traffic_controls, dim=0)   # ok

    return lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls, id_2_idx, counts, len(counts)
