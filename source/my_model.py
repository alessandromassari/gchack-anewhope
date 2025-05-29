import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, NNConv, global_mean_pool, global_max_pool, global_add_pool, JumpingKnowledge


# GIN network section
class myGIN(nn.Module):
    def __init__(self, node_in_dim: int = 1,
                 edge_in_dim: int = 7,
                 hidden_dim: int =64, # try with 128
                 num_layers: int = 4,
                 jk_mode: str = "cat",
                 mlp_classifier_dims=(384, 256, 128),
                 out_classes: int = 6,
                 dropout: float = 0.4):
    
    super().__init__()
        self.num_layers = num_layers
        self.drop_p = dropout

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = node_in_dim if i == 0 else hidden_dim
            #MLP for edge weights
            nn_edge = nn.Sequential(
                nn.Linear(edge_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_ch*hidden_dim)
            )
            conv = GINConv(nn_edge,  train_eps=True, aggr='sum')
            self.convs.append(conv)

            #jumping knowledge aggregator
            self.jk = JumpingKnowledge(mode=jk_mode)

            # pooling stage
            self.pool_fns = [global_mean_pool, global_max_pool, global_add_pool]

            # final MLP classifier
            mlp_layers = []
            input_dim = hidden_dim * num_layers * len(self.pool_fns) if jk_mode == 'cat' else hidden_dim * len(self.pool_fns)
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

        # apply Jumping Knowledge
        h_jk = self.jk(h_list)  # [N, hidden_dim * num_layers] if cat, else [N, hidden_dim]

        # global pooling
        pooled = []
        for pool in self.pool_fns:
            pooled.append(pool(h_jk, batch))  # each [batch_size, hidden_dim * num_layers] or [batch_size, hidden_dim]
        graph_repr = torch.cat(pooled, dim=-1)  # [batch_size, hidden_dim * num_layers * 3] or [batch_size, hidden_dim * 3]

        # classification
        out = self.classifier(graph_repr)  # [batch_size, out_classes]
        return out