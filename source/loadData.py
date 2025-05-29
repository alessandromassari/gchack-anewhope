# Starting from Deep Learning Hackthon GitHub repository, thank you!
# then I improved some stuff as we usually did in Tor Vergata 

import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import os
import networkx as nx
from tqdm import tqdm 
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx, degree

# dimension of node_features
node_feat_size = 1

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        self.graphs = self.loadGraphs(self.raw)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path):
        print(f"Loading graphs from {path}...")
        print("This may take a few minutes, please wait...")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))
        return graphs

# new version 
def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)
    # Feature 1: degree
    node_deg = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.float).view(-1,1)

    # Feature 2: inverse degree
    # node_inv_deg = 1.0 / (node_deg + 1e-7)

    # Features 3 and 4 intentionally left blank
    # node_features = torch.cat([node_deg, node_inv_deg], dim=1)
    node_features = torch.cat([node_deg], dim=1)
    if node_features.size(1) < node_feat_size:
        pad = torch.ones(num_nodes, node_feat_size - node_features.size(1))
        node_features = torch.cat([node_features, pad], dim=1)

    data.x = node_features
    return data
