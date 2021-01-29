import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as gnn
from torch_geometric.nn.inits import glorot_orthogonal


class RGCNStack(nn.Module):
    def __init__(
        self,
        initial_size,
        output_size,
        middle_size,
        num_nodes,
        num_relations,
        *args,
        **kwargs
    ):
        super().__init__()
        self.emb = nn.parameter.Parameter(
            torch.ones((num_nodes, initial_size)), requires_grad=True
        )
        glorot_orthogonal(self.emb, 1)
        self.conv1 = gnn.RGCNConv(
            initial_size, middle_size, num_relations, num_bases=12
        )
        self.conv2 = gnn.RGCNConv(
            middle_size,
            output_size - middle_size - initial_size,
            num_relations,
            num_bases=12,
        )
        self.drop = nn.Dropout(0.2)

    def forward(self, adj_t, edge_types=None):
        """Calculates embeddings"""
        x1 = F.relu(self.conv1(self.emb, adj_t, edge_types))
        x2 = F.relu(self.conv2(x1, adj_t, edge_types))
        x2 = torch.cat((x2, x1, self.emb), 1)
        x2 = self.drop(x2)
        return x2


class DistMult(nn.Module):
    def __init__(self, input_size, num_relations):
        super().__init__()
        self.rel = nn.parameter.Parameter(
            torch.ones((num_relations, input_size)), requires_grad=True
        )
        glorot_orthogonal(self.rel, 1)

    def forward(self, z, edge_index, relation_id, sigmoid=True):
        res = (
            (z[edge_index[0]] * self.rel[relation_id]) * z[edge_index[1]]
        ).sum(dim=1)
        if not sigmoid:
            return res
        return torch.sigmoid(res)
