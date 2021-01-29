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
    return (pos_loss + neg_loss) / float(len(pos_scores) + len(neg_scores))


def train_step(
    model,
    optimizer,
    device,
    train_adj_t,
    pos_val,
    neg_val,
    entity_type_dict,
    relation_to_entity,
    edge_types_to_train,
    neg_sample_size,
):
    train_pos_adj, dropmask = drop_edges(train_adj_t)
    train_pos_adj = train_pos_adj.to(device)

    model.train()
    optimizer.zero_grad()
    z = model.encode(train_pos_adj)

    loss = 0
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
            pos_scores = model.decoder(
                z, pos_edges.to(device), edge_type, sigmoid=False
            )
            possible_tail_nodes = entity_type_dict[
                relation_to_entity["tail"][edge_type]
            ]
            possible_head_nodes = entity_type_dict[
                relation_to_entity["head"][edge_type]
            ]
            neg_edges = negative_sample(
                pos_edges,
                *possible_head_nodes,
                *possible_tail_nodes,
                neg_sample_size
            )
            neg_scores = model.decoder(
                z, neg_edges.to(device), edge_type, sigmoid=False
            )

            loss += logloss(pos_scores, neg_scores)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        auc, ap = test(
            z, model.decoder, 0, pos_val.to(device), neg_val.to(device),
        )
    return model, auc, ap, loss.item() / len(edge_types_to_train)
