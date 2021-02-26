#!/usr/bin/env python
# coding: utf-8

import json
from collections import defaultdict
from itertools import product
from sys import argv

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_sparse import SparseTensor

import data

from fastapi import FastAPI


def set_grad(var):
    def hook(grad):
        var.grad = grad

    return hook


def safe_int(x):
    try:
        return int(x)
    except ValueError:
        return -1


class WholeModel(nn.Module):
    """A class for representing whole model to get saliencies"""

    def __init__(self, model, head1, head2, adj_t, proteins, diseases):
        super().__init__()
        self.model = model
        self.head1 = head1
        self.head2 = head2
        self.adj_t = adj_t.to(self.model.encoder.device1)
        self.proteins = proteins
        self.diseases = diseases

    def forward(self, embs):
        a = F.relu(self.model.encoder.conv1(embs, self.adj_t, None))
        b = F.relu(self.model.encoder.conv2(a, self.adj_t, None))
        c = F.relu(self.model.encoder.conv3(b, self.adj_t, None))

        emb = embs.to(self.model.encoder.device2)
        x1 = a.to(self.model.encoder.device2)
        x2 = b.to(self.model.encoder.device2)
        x3 = c.to(self.model.encoder.device2)
        z = torch.cat((x3, x2, x1, emb), 1)
        embs_protein = torch.zeros((1, z.shape[1])).to(
            self.model.encoder.device2
        )
        embs_disease = torch.zeros((1, z.shape[1])).to(
            self.model.encoder.device2
        )
        min_mean_max = torch.zeros((1, 3)).to(self.model.encoder.device2)

        embs_protein[0] = z[self.proteins].mean(0)
        embs_disease[0] = z[self.diseases].mean(0)
        prod = torch.LongTensor(list(product(self.proteins, self.diseases))).T
        d = self.model.decoder(z, prod, 0, sigmoid=False)
        min_mean_max[0, 0] = d.min()
        min_mean_max[0, 1] = d.mean()
        min_mean_max[0, 2] = d.max()
        z1 = self.head1(torch.cat((embs_protein, embs_disease), 1))
        probas = self.head2(torch.cat((z1, min_mean_max), 1))
        a.register_hook(set_grad(a))
        b.register_hook(set_grad(b))
        probas.backward()
        g3, g2, g1 = (
            embs.grad.cpu().numpy(),
            a.grad.cpu().numpy(),
            b.grad.cpu().numpy(),
        )
        embs.grad.zero_()
        b.grad.zero_()
        a.grad.zero_()
        return probas, g3, g2, g1


def get_saliencies(
    model, head1, head2, adj_t, proteins, diseases, node_human_readable
):
    m = WholeModel(model, head1, head2, adj_t, proteins, diseases)
    for p in m.parameters():
        p.requires_grad = False
    m.model.encoder.emb.requires_grad = True
    if m.model.encoder.emb.grad is not None:
        m.model.encoder.emb.grad.zero_()
    logit, g3, g2, g1 = m(m.model.encoder.emb)
    sal = pd.DataFrame(g3)
    sal["internal_id"] = sal.index
    sal.index = sal.index.map(node_human_readable)
    sal["impact_3"] = g3.sum(1)
    sal["impact_2"] = g2.sum(1)
    sal["impact_1"] = g1.sum(1)
    sal["sum"] = sal["impact_1"] + sal["impact_2"] + sal["impact_3"]
    return logit, sal


# # Model loading
def load_model(dir):
    model = torch.load(dir + "/best_auc_ft.pt")
    head1 = torch.load(dir + "/head1.pt").to("cuda:3")
    head2 = torch.load(dir + "/head2.pt").to("cuda:3")
    model.encoder.change_devices(model.encoder.device1, "cuda:3")
    model.decoder.to("cuda:3")
    return model, head1, head2


# # Data
def load_adjacency(path):
    train_edge, val_edge, test_edge, entity_type_dict = data.load_dataset(path)
    head = torch.cat((train_edge["head"], val_edge["head"]))
    tail = torch.cat((train_edge["tail"], val_edge["tail"]))

    # Some useful values
    num_relations = train_edge["relation"].max() + 1
    num_nodes = max(entity_type_dict.values())[1] + 1

    relation_to_entity = defaultdict(dict)
    for i in range(num_relations):
        relation_to_entity["head"][i] = np.array(train_edge["head_type"])[
            train_edge["relation"] == i
        ][0]
        relation_to_entity["tail"][i] = np.array(train_edge["tail_type"])[
            train_edge["relation"] == i
        ][0]

    # Prepare training data
    adj_t = SparseTensor(
        row=head,
        col=tail,
        value=torch.cat((train_edge["relation"], val_edge["relation"])),
        sparse_sizes=(num_nodes, num_nodes),
    )
    return adj_t


# ## Mappings
def load_mappings(index_path, dict_path):

    with open(index_path) as fin:
        node_index = json.load(fin)
        node_decoder = {v: k for k, v in node_index.items()}

    genes = pd.read_csv(
        dict_path + "/CTD_genes.csv.gz", skiprows=29, header=None
    )
    genes.columns = [
        "GeneSymbol",
        "GeneName",
        "GeneID",
        "AltGeneIDs",
        "Synonyms",
        "BioGRIDIDs",
        "PharmGKBIDs",
        "UniProtIDs",
    ]
    genes["BioGRIDIDs"] = genes["BioGRIDIDs"].str.split("|")
    genes = genes.explode("BioGRIDIDs")
    geneid_biogrid = genes[["GeneSymbol", "BioGRIDIDs"]].dropna()
    geneid_biogrid.index = geneid_biogrid["BioGRIDIDs"]
    geneid_biogrid = geneid_biogrid["GeneSymbol"]
    geneid_biogrid.index = geneid_biogrid.index.astype(int)

    ddi = pd.read_csv(
        dict_path + "/CTD_diseases.csv.gz", skiprows=29, header=None,
    )
    dis_desc = ddi[[0, 1]]
    dis_desc.index = dis_desc[1]
    dis_desc = dis_desc[0]

    chem = pd.read_csv(
        dict_path + "/CTD_chemicals.csv.gz", skiprows=29, header=None,
    )
    chem.index = chem[1].str[5:]
    chem = chem[0]

    path = pd.read_csv(
        dict_path + "/CTD_pathways.csv.gz", skiprows=29, header=None,
    )
    path.index = path[1]
    path = path[0]

    node_human_readable_path = {
        k: path[v] for k, v in node_decoder.items() if v in path
    }
    node_human_readable_gene = {
        k: geneid_biogrid[int(v)]
        for k, v in node_decoder.items()
        if safe_int(v) in geneid_biogrid
    }
    node_human_readable_dis = {
        k: dis_desc[v] for k, v in node_decoder.items() if v in dis_desc
    }
    node_human_readable_chem = {
        k: chem[v] for k, v in node_decoder.items() if v in chem
    }

    node_human_readable = node_decoder.copy()
    node_human_readable.update(node_human_readable_chem)
    node_human_readable.update(node_human_readable_dis)
    node_human_readable.update(node_human_readable_gene)
    node_human_readable.update(node_human_readable_path)

    return node_index, node_decoder, node_human_readable


if __name__ == "__main__":
    model, head1, head2 = load_model(argv[1])
    adj_t = load_adjacency(argv[2])
    node_index, node_decoder, node_human_readable = load_mappings(
        argv[3], argv[4]
    )
    node_human_readable_rev = {v: k for k, v in node_human_readable.items()}

    graph = nx.read_edgelist(argv[5], create_using=nx.MultiDiGraph,)

    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.get("/items/{protein_ids}/{disease_ids}")
    async def read_item(protein_ids, disease_ids):
        proteins = [node_human_readable_rev[k] for k in protein_ids.split(",")]
        diseases = [node_human_readable_rev[k] for k in disease_ids.split(",")]
        logit, sal = get_saliencies(
            model, head1, head2, adj_t, proteins, diseases, node_human_readable
        )
        return {"data": sal.to_json(), "logit": logit.item()}

