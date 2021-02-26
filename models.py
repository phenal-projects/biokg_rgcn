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
        middle_size_1,
        middle_size_2,
        num_nodes,
        num_relations,
        device1,
        device2,
    ):
        super().__init__()
        self.device1 = device1
        self.device2 = device2
        self.emb = nn.parameter.Parameter(
            torch.ones((num_nodes, initial_size)), requires_grad=True
        ).to(device1)
        glorot_orthogonal(self.emb, 1)
        self.conv1 = gnn.RGCNConv(
            initial_size, middle_size_1, num_relations, num_bases=12
        ).to(device1)
        self.conv2 = gnn.RGCNConv(
            middle_size_1, middle_size_2, num_relations, num_bases=12
        ).to(device1)
        self.conv3 = gnn.RGCNConv(
            middle_size_2,
            output_size - middle_size_2 - middle_size_1 - initial_size,
            num_relations,
            num_bases=12,
        ).to(device1)
        self.drop = nn.Dropout(0.2)

    def forward(self, adj_t, edge_types=None):
        """Calculates embeddings"""
        adj_t = adj_t.to(self.device1)
        if edge_types is not None:
            edge_types = edge_types.to(self.device1)
        x1 = F.relu(self.conv1(self.emb, adj_t, edge_types))
        x2 = F.relu(self.conv2(x1, adj_t, edge_types))
        x3 = F.relu(self.conv3(x2, adj_t, edge_types))

        emb = self.emb.to(self.device2)
        x1 = x1.to(self.device2)
        x2 = x2.to(self.device2)
        x3 = x3.to(self.device2)
        x3 = torch.cat((x3, x2, x1, emb), 1)
        x3 = self.drop(x3)
        return x3

    def change_devices(self, device1, device2):
        self.device1 = device1
        self.device2 = device2
        self.to(device1)


class Lookup(nn.Module):
    def __init__(
        self, initial_size, num_nodes, num_relations, *args, **kwargs
    ):
        super().__init__()
        self.emb = nn.parameter.Parameter(
            torch.ones((num_nodes, initial_size)), requires_grad=True
        )
        glorot_orthogonal(self.emb, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, adj_t, edge_types=None):
        """Calculates embeddings"""
        return self.drop(self.emb)


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
