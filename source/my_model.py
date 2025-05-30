import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, NNConv, global_mean_pool, global_max_pool, global_add_pool, JumpingKnowledge


# New hope network (we cannot do just ctrl+C and ctrl+V, we're still young and hungry
class NHNmodel(nn.Module):
    def __init__(self, node_in_dim: int = 1,
                 edge_in_dim: int = 7,
                 hidden_dim: int =64, # try with 128
                 num_layers: int = 4,
                 mlp_classifier_dims=(384, 256, 128),
                 out_classes: int = 6,
                 dropout: float = 0.4):
        #
        super().__init__()
        self.num_layers = num_layers
        self.drop_p = dropout

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = node_in_dim if i == 0 else hidden_dim
            out_ch = hidden_dim
            
            #MLP for edge weights: a sort of encoder in my mind
            nn_edge_mlp = nn.Sequential(
                nn.Linear(edge_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_ch*hidden_dim)
            )
            # NNConv 
            conv = NNConv(in_ch, out_ch, nn_edge_mlp, aggr='sum')
            self.convs.append(conv)

        # pooling stage - to choose one - try and then choose
        self.pool_fns = [global_mean_pool, global_max_pool, global_add_pool]

        # final MLP classifier
        mlp_layers = []
        input_dim = hidden_dim * len(self.pool_fns)
        prev_dim = input_dim
        for dim in mlp_classifier_dims:
            mlp_layers += [
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = dim
        mlp_layers.append(nn.Linear(prev_dim, out_classes))
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(self, x, edge_index, edge_attr, batch):
        # x: [N, node_in_dim]
        # edge_attr: [E, edge_in_dim]
        h_list = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.drop_p, training=self.training)
            h_list.append(h)

        # final pooling stage: MLP classifier need a
        h_final = h_list[-1]
        pooled = []
        for pool in self.pool_fns:
            pooled.append(pool(h_final, batch)) # Pooling su h_final
        graph_repr = torch.cat(pooled, dim=-1)
        
        # classification
        out = self.classifier(graph_repr)  # [batch_size, out_classes]
        return out
