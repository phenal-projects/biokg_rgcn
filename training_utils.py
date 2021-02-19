from itertools import product
import torch
from torch import nn
from torch.nn import functional as F
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, average_precision_score


def drop_edges(mat, p=0.3):
    mask = torch.rand((mat.storage.row().shape[0],)) > p
    matr = SparseTensor(
        row=mat.storage.row()[mask],
        col=mat.storage.col()[mask],
        value=mat.storage.value()[mask],
        sparse_sizes=mat.storage.sparse_sizes(),
    )
    return matr, mask


def test(z, decoder, entity_types, pos_edge_index, neg_edge_index):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)
    pos_pred = decoder(z, pos_edge_index, entity_types)
    neg_pred = decoder(z, neg_edge_index, entity_types)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred)


def negative_sample(
    positive_sample,
    start_head_index,
    stop_head_index,
    start_tail_index,
    stop_tail_index,
    size,
):
    heads, tails = positive_sample
    heads = heads[torch.randint(0, len(heads), size=(size // 2,))]
    tails = tails[torch.randint(0, len(tails), size=(size // 2,))]
    neg_heads = torch.randint(
        start_head_index, stop_head_index, size=(size // 2,)
    )
    neg_tails = torch.randint(
        start_tail_index, stop_tail_index, size=(size // 2,)
    )
    return torch.stack(
        (torch.cat((heads, neg_heads)), torch.cat((neg_tails, tails)))
    )


def logloss(pos_scores, neg_scores, adversarial_temperature=1.0):
    pos_loss = -F.logsigmoid(pos_scores).sum()
    neg_loss = -(
        F.softmax(neg_scores * adversarial_temperature, dim=0).detach()
        * F.logsigmoid(-neg_scores)
    ).sum()
    return (pos_loss + neg_loss), float(len(pos_scores) + len(neg_scores))


def train_step(
    model,
    optimizer,
    train_adj_t,
    pos_val,
    neg_val,
    entity_type_dict,
    relation_to_entity,
    edge_types_to_train,
    neg_sample_size,
    device,
):
    train_pos_adj, dropmask = drop_edges(train_adj_t)

    model.train()
    optimizer.zero_grad()
    z = model.encode(train_pos_adj)

    pos_scores = list()
    neg_scores = list()
    for edge_type in edge_types_to_train:
        pos_edges = torch.stack(
            (
                train_adj_t.storage.row()[~dropmask][
                    train_adj_t.storage.value()[~dropmask] == edge_type
                ],
                train_adj_t.storage.col()[~dropmask][
                    train_adj_t.storage.value()[~dropmask] == edge_type
                ],
            )
        )
        if pos_edges.shape[-1] != 0:
            pos_scores.append(
                model.decoder(
                    z, pos_edges.to(device), edge_type, sigmoid=False
                )
            )
            possible_tail_nodes = entity_type_dict[
                relation_to_entity["tail"][edge_type]
            ]
            possible_head_nodes = entity_type_dict[
                relation_to_entity["head"][edge_type]
            ]
            for _ in range(neg_sample_size):
                neg_edges = negative_sample(
                    pos_edges,
                    *possible_head_nodes,
                    *possible_tail_nodes,
                    int(len(pos_edges[0]))
                )
                neg_scores.append(
                    model.decoder(
                        z, neg_edges.to(device), edge_type, sigmoid=False
                    )
                )
    l, w = logloss(torch.cat(pos_scores), torch.cat(neg_scores))
    l.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        auc, ap = test(
            z, model.decoder, 0, pos_val.to(device), neg_val.to(device),
        )
    return model, auc, ap, l.item() / w


def ft_inference(
    model,
    cl_head_1,
    cl_head_2,
    train_adj_t,
    protein_bounds,
    disease_bounds,
    df,
    device,
):
    z = model.encode(train_adj_t)
    embs_protein = torch.zeros((len(df), z.shape[1])).to(device)
    embs_disease = torch.zeros((len(df), z.shape[1])).to(device)
    min_mean_max = torch.zeros((len(df)), 3).to(device)
    neutral_protein = z[protein_bounds[0] : protein_bounds[1]].mean(0)
    neutral_disease = z[disease_bounds[0] : disease_bounds[1]].mean(0)
    for i, (_, idx) in enumerate(df.iterrows()):
        if len(idx["protein"]) > 0:
            embs_protein[i] = z[idx["protein"]].mean(0)
        else:
            embs_protein[i] = neutral_protein
        if len(idx["disease"]) > 0:
            embs_disease[i] = z[idx["disease"]].mean(0)
        else:
            embs_disease[i] = neutral_disease
        if (len(idx["protein"]) > 0) and (len(idx["disease"]) > 0):
            prod = torch.LongTensor(
                list(product(idx["protein"], idx["disease"]))
            ).T
            d = model.decoder(z, prod, 0, sigmoid=False)
            min_mean_max[i, 0] = d.min()
            min_mean_max[i, 1] = d.mean()
            min_mean_max[i, 2] = d.max()
        else:
            min_mean_max[i, 0] = 0.0
            min_mean_max[i, 1] = 0.0
            min_mean_max[i, 2] = 0.0
    z1 = cl_head_1(torch.cat((embs_protein, embs_disease), 1))
    probas = cl_head_2(torch.cat((z1, min_mean_max), 1))
    return probas
