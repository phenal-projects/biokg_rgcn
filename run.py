import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.optim as opt
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric import nn as gnn
from torch_sparse import SparseTensor

import models
from training_utils import train_step

# Setup parser
parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="set a seed for PRNG", type=int, default=0)
parser.add_argument(
    "--size1",
    help="set the size of the initial embeddings",
    type=int,
    default=64,
)
parser.add_argument(
    "--size2",
    help="set the size of the middle embeddings",
    type=int,
    default=64,
)
parser.add_argument(
    "--size3",
    help="set the size of the last part of the embeddings",
    type=int,
    default=64,
)
parser.add_argument(
    "--negsize", help="negsize/possize", type=float, default=1,
)
parser.add_argument(
    "--adv",
    help="set the adversarial temperature for the negative part of the loss",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--lr", help="set the learning rate", type=float, default=0.005,
)
parser.add_argument(
    "--epochs",
    help="set the number of epochs to train",
    type=int,
    default=400,
)
parser.add_argument(
    "--device", help="the device to train on", type=str, default="cpu",
)
args = parser.parse_args()


# Reproducibility
torch.set_deterministic(True)
torch.manual_seed(0)

# Load the dataset and split edges
biokg = PygLinkPropPredDataset(name="ogbl-biokg", root="./datasets")
split_edge = biokg.get_edge_split()
train_edge, valid_edge, test_edge = (
    split_edge["train"],
    split_edge["valid"],
    split_edge["test"],
)

# Some useful values
num_relations = train_edge["relation"].max() + 1
num_nodes = sum(biokg[0]["num_nodes_dict"].values())

entity_type_dict = dict()
cur_idx = 0
for key in biokg[0]["num_nodes_dict"]:
    entity_type_dict[key] = (
        cur_idx,
        cur_idx + biokg[0]["num_nodes_dict"][key],
    )
    cur_idx += biokg[0]["num_nodes_dict"][key]

relation_to_entity = defaultdict(dict)
for i in range(num_relations):
    relation_to_entity["head"][i] = np.array(train_edge["head_type"])[
        train_edge["relation"] == i
    ][0]
    relation_to_entity["tail"][i] = np.array(train_edge["tail_type"])[
        train_edge["relation"] == i
    ][0]

# Prepare training data
head = (
    torch.tensor([entity_type_dict[x][0] for x in train_edge["head_type"]])
    + train_edge["head"]
)
tail = (
    torch.tensor([entity_type_dict[x][0] for x in train_edge["tail_type"]])
    + train_edge["tail"]
)
train_adj_t = SparseTensor(
    row=head,
    col=tail,
    value=train_edge["relation"],
    sparse_sizes=(num_nodes, num_nodes),
)

# Prepare validation data (only for relation == 0, entailment)
head_offset = torch.tensor(
    [entity_type_dict[x][0] for x in valid_edge["head_type"]]
)
valid_head = head_offset + valid_edge["head"]
tail_offset = torch.tensor(
    [entity_type_dict[x][0] for x in valid_edge["tail_type"]]
)
valid_tail = tail_offset + valid_edge["tail"]
valid_edge["head_neg"] = valid_edge["head_neg"] + head_offset.view(-1, 1)
valid_edge["tail_neg"] = valid_edge["tail_neg"] + tail_offset.view(-1, 1)
pos_val = torch.stack((valid_head, valid_tail))[:, valid_edge["relation"] == 0]
neg_val = torch.stack(
    (pos_val[0], valid_edge["tail_neg"][valid_edge["relation"] == 0, 0],)
)

# Model
encoder = models.RGCNStack(
    args.size1,
    args.size1 + args.size2 + args.size3,
    args.size2,
    num_nodes,
    num_relations,
)
decoder = models.DistMult(args.size1 + args.size2 + args.size3, num_relations)
model = gnn.GAE(encoder, decoder).to(args.device)
optimizer = opt.Adam(model.parameters(), args.lr)

best_loss = 0.5
best_auc = 0.6

for epoch in range(args.epochs):
    model, auc, ap, loss = train_step(
        model,
        optimizer,
        args.device,
        train_adj_t,
        pos_val,
        neg_val,
        entity_type_dict,
        relation_to_entity,
        list(range(num_relations)),
        args.negsize,
    )
    if auc > best_auc:
        torch.save(model, "best_auc.pt")
        best_auc = auc
    if loss < best_loss:
        torch.save(model, "best_loss.pt")
        best_loss = loss
    print(
        "Epoch {}/{}, AUC {:.3f}, AP {:.3f}, Loss {:.4f}".format(
            epoch + 1, args.epochs, auc, ap, loss
        )
    )
torch.save(model, "final_loss.pt")
