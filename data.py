from ogb.linkproppred.dataset_pyg import PygLinkPropPredDataset
import numpy as np
import torch


def load_biokg():
    biokg = PygLinkPropPredDataset(name="ogbl-biokg", root="./datasets")
    split_edge = biokg.get_edge_split()
    train_edge, valid_edge, test_edge = (
        split_edge["train"],
        split_edge["valid"],
        split_edge["test"],
    )
    entity_type_dict = dict()
    cur_idx = 0
    for key in biokg[0]["num_nodes_dict"]:
        entity_type_dict[key] = (
            cur_idx,
            cur_idx + biokg[0]["num_nodes_dict"][key],
        )
        cur_idx += biokg[0]["num_nodes_dict"][key]
    train_edge["head"] = (
        torch.tensor([entity_type_dict[x][0] for x in train_edge["head_type"]])
        + train_edge["head"]
    )
    train_edge["tail"] = (
        torch.tensor([entity_type_dict[x][0] for x in train_edge["tail_type"]])
        + train_edge["tail"]
    )

    valid_edge["head"] = (
        torch.tensor([entity_type_dict[x][0] for x in valid_edge["head_type"]])
        + valid_edge["head"]
    )
    valid_edge["tail"] = (
        torch.tensor([entity_type_dict[x][0] for x in valid_edge["tail_type"]])
        + valid_edge["tail"]
    )
    valid_edge["head_neg"] = (
        torch.tensor(
            [entity_type_dict[x][0] for x in valid_edge["head_type"]]
        ).reshape(-1, 1)
        + valid_edge["head_neg"]
    )
    valid_edge["tail_neg"] = (
        torch.tensor(
            [entity_type_dict[x][0] for x in valid_edge["tail_type"]]
        ).reshape(-1, 1)
        + valid_edge["tail_neg"]
    )

    test_edge["head"] = (
        torch.tensor([entity_type_dict[x][0] for x in test_edge["head_type"]])
        + test_edge["head"]
    )
    test_edge["tail"] = (
        torch.tensor([entity_type_dict[x][0] for x in test_edge["tail_type"]])
        + test_edge["tail"]
    )
    test_edge["head_neg"] = (
        torch.tensor(
            [entity_type_dict[x][0] for x in test_edge["head_type"]]
        ).reshape(-1, 1)
        + test_edge["head_neg"]
    )
    test_edge["tail_neg"] = (
        torch.tensor(
            [entity_type_dict[x][0] for x in test_edge["tail_type"]]
        ).reshape(-1, 1)
        + test_edge["tail_neg"]
    )

    return train_edge, valid_edge, test_edge, entity_type_dict


def load_dataset(path):
    train_edge = dict()
    valid_edge = dict()
    test_edge = dict()

    # four row, (s, p, o, s_type, o_type, train/val/test)
    triples = np.load(path)
    # nodes of the same type should have idx within one continuous interval

    entity_type_dict = dict()
    entity_types = set(triples[3].unique()) | set(triples[4].unique())
    for e in entity_types:
        entity_type_dict[e] = (
            min(
                triples[0, triples[3] == e].min(),
                triples[2, triples[4] == e].min(),
            ),
            max(
                triples[0, triples[3] == e].max(),
                triples[2, triples[4] == e].max(),
            ),
        )

    train_edge["head"] = torch.tensor(triples[0, triples[5] == 0])
    train_edge["relation"] = torch.tensor(triples[1, triples[5] == 0])
    train_edge["tail"] = torch.tensor(triples[2, triples[5] == 0])
    train_edge["head_type"] = torch.tensor(triples[3, triples[5] == 0])
    train_edge["tail_type"] = torch.tensor(triples[4, triples[5] == 0])

    valid_edge["head"] = torch.tensor(triples[0, triples[5] == 1])
    valid_edge["relation"] = torch.tensor(triples[1, triples[5] == 1])
    valid_edge["tail"] = torch.tensor(triples[2, triples[5] == 1])
    valid_edge["head_type"] = torch.tensor(triples[3, triples[5] == 1])
    valid_edge["tail_type"] = torch.tensor(triples[4, triples[5] == 1])

    test_edge["head"] = torch.tensor(triples[0, triples[5] == 2])
    test_edge["relation"] = torch.tensor(triples[1, triples[5] == 2])
    test_edge["tail"] = torch.tensor(triples[2, triples[5] == 2])
    test_edge["head_type"] = torch.tensor(triples[3, triples[5] == 2])
    test_edge["tail_type"] = torch.tensor(triples[4, triples[5] == 2])

    valid_edge["head_neg"] = torch.stack(
        [
            torch.randint(
                entity_type_dict[x][0], entity_type_dict[x][1], size=(500,)
            )
            for x in valid_edge["head_type"]
        ]
    )
    valid_edge["tail_neg"] = torch.stack(
        [
            torch.randint(
                entity_type_dict[x][0], entity_type_dict[x][1], size=(500,)
            )
            for x in valid_edge["tail_neg"]
        ]
    )

    test_edge["head_neg"] = torch.stack(
        [
            torch.randint(
                entity_type_dict[x][0], entity_type_dict[x][1], size=(500,)
            )
            for x in test_edge["head_type"]
        ]
    )
    test_edge["tail_neg"] = torch.stack(
        [
            torch.randint(
                entity_type_dict[x][0], entity_type_dict[x][1], size=(500,)
            )
            for x in test_edge["tail_neg"]
        ]
    )
    return train_edge, valid_edge, test_edge, entity_type_dict
