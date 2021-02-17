import argparse
from collections import defaultdict
from itertools import chain, product

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as opt
from ogb.linkproppred import Evaluator
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric import nn as gnn
from torch_sparse import SparseTensor

import models
from data import load_biokg, load_dataset
from training_utils import train_step

# Setup parser
parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="set a seed for PRNG", type=int, default=0)
parser.add_argument(
    "--size1",
    help="set the size of the initial embeddings",
    type=int,
    default=52,
)
parser.add_argument(
    "--size2",
    help="set the size of the middle embeddings",
    type=int,
    default=52,
)
parser.add_argument(
    "--size3",
    help="set the size of the last part of the embeddings",
    type=int,
    default=52,
)
parser.add_argument(
    "--size4",
    help="set the size of the last part of the embeddings",
    type=int,
    default=52,
)
parser.add_argument("--negsize", help="negsize/possize", type=float, default=1)
parser.add_argument(
    "--adv",
    help="set the adversarial temperature for the negative part of the loss",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--lr", help="set the learning rate", type=float, default=0.005
)
parser.add_argument(
    "--wd", help="set the weight decay", type=float, default=0.0001
)
parser.add_argument(
    "--epochs", help="set the number of epochs to train", type=int, default=400
)
parser.add_argument(
    "--device", help="the device to train on", type=str, default="cpu"
)
parser.add_argument(
    "--data",
    help="'biokg' or a path to directory with datasets",
    type=str,
    default="biokg",
)
parser.add_argument(
    "--target_relation",
    help="an id of target relation. Increases its weight in the loss",
    type=int,
    default=0,
)
parser.add_argument(
    "--mlflow",
    help="URI of the mlflow instance for logging",
    type=str,
    default="http://localhost:12345",
)
parser.add_argument(
    "--finetuning_dataset",
    help="a path to a hdf file with disease-target pairs for CTOP finetuning",
    type=str,
    default="None",
)
args = parser.parse_args()


# Reproducibility
torch.set_deterministic(True)
torch.manual_seed(args.seed)
mlflow.set_tracking_uri(args.mlflow)

# Load the dataset and split edges
if args.data == "biokg":
    train_edge, valid_edge, test_edge, entity_type_dict = load_biokg()
else:
    train_edge, valid_edge, test_edge, entity_type_dict = load_dataset(
        args.data
    )

head = train_edge["head"]
tail = train_edge["tail"]

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
train_adj_t = SparseTensor(
    row=head,
    col=tail,
    value=train_edge["relation"],
    sparse_sizes=(num_nodes, num_nodes),
)

# Prepare validation data (only for relation == 0, entailment)
pos_val = torch.stack((valid_edge["head"], valid_edge["tail"]))[
    :, valid_edge["relation"] == args.target_relation
]
neg_val = torch.stack(
    (
        pos_val[0],
        valid_edge["tail_neg"][
            valid_edge["relation"] == args.target_relation, 0
        ],
    )
)

# Model
encoder = models.RGCNStack(
    args.size1,
    args.size1 + args.size2 + args.size3 + args.size4,
    args.size2,
    args.size3,
    num_nodes,
    num_relations,
)
decoder = models.DistMult(
    args.size1 + args.size2 + args.size3 + args.size4, num_relations
)
model = gnn.GAE(encoder, decoder).to(args.device)
optimizer = opt.Adam(model.parameters(), args.lr, weight_decay=args.wd)

best_loss = 0.5
best_auc = 0.0

with mlflow.start_run():
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
        mlflow.log_metric(key="balanced_roc_auc", value=auc, step=epoch)
        mlflow.log_metric(key="balanced_ap", value=ap, step=epoch)
        mlflow.log_metric(key="loss", value=loss, step=epoch)
    model = torch.load("best_auc.pt")
    mlflow.log_artifact("best_auc.pt")

    # Link-prediction validation
    evaluator = Evaluator(name="ogbl-biokg")
    with torch.no_grad():
        z = model.encode(train_adj_t.to(args.device))
        results = []
        for et in range(num_relations):
            subresults = []
            pos_val = torch.stack(
                (
                    test_edge["head"][test_edge["relation"] == et],
                    test_edge["tail"][test_edge["relation"] == et],
                )
            )
            subresults.append(
                model.decoder(z, pos_val.to(args.device), et)
                .detach()
                .cpu()
                .numpy()
            )
            for i in range(500):
                tail_neg = test_edge["tail_neg"][
                    test_edge["relation"] == et, i
                ]
                subresults.append(
                    model.decoder(
                        z,
                        torch.stack((pos_val[0], tail_neg)).to(args.device),
                        et,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            results.append(np.stack(subresults))
            scores = np.concatenate(results, 1).T
            eval_results = evaluator.eval(
                {"y_pred_pos": scores[:, 0], "y_pred_neg": scores[:, 1:]}
            )
            mlflow.log_metric(
                key="test_lp_mrr", value=eval_results["mrr_list"].mean()
            )

    # CTOP validation
    ctop_ds = pd.read_hdf(args.finetuning_dataset, "ctop")
    train = ctop_ds[ctop_ds["subset"] == "train"]
    train_y = torch.tensor(train["result"].values).reshape(-1, 1)
    cl_head_1 = nn.Sequential(
        nn.Linear(
            2 * (args.size1 + args.size2 + args.size3 + args.size4), 128
        ),
        nn.ReLU(),
        nn.Linear(128, 13),
        nn.ReLU(),
    ).to(args.device)
    cl_head_2 = nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1),
    ).to(args.device)
    optimizer = opt.Adam(
        chain(
            model.parameters(), cl_head_1.parameters(), cl_head_2.parameters()
        ),
        args.lr,
    )
    ls = nn.BCEWithLogitsLoss()
    for epoch in range(1000):
        optimizer.zero_grad()
        z = model.encode(train_adj_t.to(args.device))
        embs_protein = torch.zeros(
            (len(train), args.size1 + args.size2 + args.size3 + args.size4)
        ).to(args.device)
        embs_disease = torch.zeros(
            (len(train), args.size1 + args.size2 + args.size3 + args.size4)
        ).to(args.device)
        min_mean_max = torch.zeros((len(train)), 3).to(args.device)
        if args.data == "biokg":
            neutral_protein = z[
                entity_type_dict["protein"][0] : entity_type_dict["protein"][1]
            ].mean(0)
            neutral_disease = z[
                entity_type_dict["disease"][0] : entity_type_dict["disease"][1]
            ].mean(0)
        else:
            neutral_protein = z[
                entity_type_dict[0][0] : entity_type_dict[0][1]
            ].mean(0)
            neutral_disease = z[
                entity_type_dict[1][0] : entity_type_dict[1][1]
            ].mean(0)
        for i, (_, idx) in enumerate(train.iterrows()):
            if (len(idx["protein"]) > 0) and (len(idx["disease"]) > 0):
                embs_protein[i] = z[idx["protein"]].mean(0)
                embs_disease[i] = z[idx["disease"]].mean(0)
                prod = torch.LongTensor(
                    list(product(idx["protein"], idx["disease"]))
                ).T
                d = model.decoder(z, prod, 0, sigmoid=False)
                min_mean_max[i, 0] = d.min()
                min_mean_max[i, 1] = d.mean()
                min_mean_max[i, 2] = d.max()
            else:
                embs_protein[i] = neutral_protein
                embs_disease[i] = neutral_disease
                min_mean_max[i, 0] = 0.0
                min_mean_max[i, 1] = 0.0
                min_mean_max[i, 2] = 0.0
        z1 = cl_head_1(torch.cat((embs_protein, embs_disease), 1))
        probas = cl_head_2(torch.cat((z1, min_mean_max), 1))
        loss = ls(probas, train_y.to(args.device))
        loss.backward()
        mlflow.log_metric(
            key="ft_loss", value=loss.item(), step=epoch + args.epochs
        )
        optimizer.step()

    # Testing
    with torch.no_grad():
        z = model.encode(train_adj_t.to(args.device))
        for subset in ctop_ds["subset"].unique():
            if subset != "train":
                test = ctop_ds[ctop_ds["subset"] == subset]
                embs_protein = torch.zeros(
                    (
                        len(test),
                        args.size1 + args.size2 + args.size3 + args.size4,
                    )
                ).to(args.device)
                embs_disease = torch.zeros(
                    (
                        len(test),
                        args.size1 + args.size2 + args.size3 + args.size4,
                    )
                ).to(args.device)
                min_mean_max = torch.zeros((len(test)), 3).to(args.device)
                if args.data == "biokg":
                    neutral_protein = z[
                        entity_type_dict["protein"][0] : entity_type_dict[
                            "protein"
                        ][1]
                    ].mean(0)
                    neutral_disease = z[
                        entity_type_dict["disease"][0] : entity_type_dict[
                            "disease"
                        ][1]
                    ].mean(0)
                else:
                    neutral_protein = z[
                        entity_type_dict[0][0] : entity_type_dict[0][1]
                    ].mean(0)
                    neutral_disease = z[
                        entity_type_dict[1][0] : entity_type_dict[1][1]
                    ].mean(0)
                for i, (_, idx) in enumerate(test.iterrows()):
                    if (len(idx["protein"]) > 0) and (len(idx["disease"]) > 0):
                        embs_protein[i] = z[idx["protein"]].mean(0)
                        embs_disease[i] = z[idx["disease"]].mean(0)
                        prod = torch.LongTensor(
                            list(product(idx["protein"], idx["disease"]))
                        ).T
                        d = model.decoder(z, prod, 0, sigmoid=False)
                        min_mean_max[i, 0] = d.min()
                        min_mean_max[i, 1] = d.mean()
                        min_mean_max[i, 2] = d.max()
                    else:
                        embs_protein[i] = neutral_protein
                        embs_disease[i] = neutral_disease
                        min_mean_max[i, 0] = 0.0
                        min_mean_max[i, 1] = 0.0
                        min_mean_max[i, 2] = 0.0
                z1 = cl_head_1(torch.cat((embs_protein, embs_disease), 1))
                probas = cl_head_2(torch.cat((z1, min_mean_max), 1))
                if len(test["result"][~test["result"].isna()]) > 0:
                    auc, ap = (
                        roc_auc_score(
                            test["result"][~test["result"].isna()],
                            probas.cpu()
                            .numpy()[~test["result"].isna()]
                            .reshape(-1),
                        ),
                        average_precision_score(
                            test["result"][~test["result"].isna()],
                            probas.cpu()
                            .numpy()[~test["result"].isna()]
                            .reshape(-1),
                        ),
                    )
                    mlflow.log_metric(
                        key="ft_auc_{}".format(subset), value=auc
                    )
                    mlflow.log_metric(key="ft_ap_{}".format(subset), value=ap)
    torch.save(model, "best_auc_ft.pt")
    torch.save(cl_head_1, "head1.pt")
    torch.save(cl_head_2, "head2.pt")
    mlflow.log_artifact("best_auc_ft.pt")
    mlflow.log_artifact("head1.pt")
    mlflow.log_artifact("head2.pt")
