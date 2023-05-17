from typing import Optional

import torch as th
from torch.utils.data import DataLoader

from datautils.datamodule import DataModuleBase
from datautils.kfold import get_idx_split
from datautils.tu.dataset import MyTUDataset


class TU_DataModule(DataModuleBase):
    def __init__(
        self, name: str,
        fold_idx: int, seed: int,
        batch_size: int, num_workers: int,
        transform: Optional[str] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False,
        train_shuffle: bool = True,
    ):

        self._name = name
        super(TU_DataModule, self).__init__(transform, transform_fn_kwargs)

        dataset = MyTUDataset(name, self._transform_fn,
                              self._transform_fn_kwargs, reload)
        self.dataset_sizes = dataset.sizes

        kfold_dir = f"{dataset.root}/{name}/10fold_idx"
        train_idx, valid_idx = get_idx_split(
            fold_idx, kfold_dir, seed,
            dataset.graph_label,
            test_set=False)

        self.train_dataset = dataset[train_idx]
        self.valid_dataset = dataset[valid_idx]

        drop_last = 0 < len(self.train_dataset) % batch_size < 8
        self.train_loader = DataLoader(
            self.train_dataset, batch_size, shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate,
            drop_last=drop_last,
            pin_memory=True)

        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.collate,
            pin_memory=True)

        self.test_loader = None

    @property
    def metric_name(self) -> str:
        return ["ACC"]

    def evaluator(self, predicts: th.Tensor, targets: th.Tensor) -> float:
        if predicts.size(1) > 1:
            pred_labels = th.argmax(predicts, 1)
        else:
            pred_labels = predicts > 0.
        correct = (pred_labels == targets).sum().item()
        accuracy = correct / predicts.shape[0]
        return {'ACC': accuracy}

    def whether_improve(self, valid_score_dict, best_valid_score_dict):
        if best_valid_score_dict is None:
            return True
        valid_score = valid_score_dict[self.metric_name[0]]
        best_valid_score = best_valid_score_dict[self.metric_name[0]]
        return valid_score > best_valid_score

    @property
    def num_node_attributes(self):
        return self.dataset_sizes['num_node_attributes']

    @property
    def num_node_labels(self):
        return self.dataset_sizes['num_node_labels']

    @property
    def num_edge_attributes(self):
        return self.dataset_sizes['num_edge_attributes']

    @property
    def num_edge_labels(self):
        return self.dataset_sizes['num_edge_labels']

    @property
    def num_graph_attributes(self):
        return self.dataset_sizes['num_graph_attributes']

    @property
    def num_graph_labels(self):
        return self.dataset_sizes['num_graph_labels']

    def __repr__(self) -> str:
        return f"TU/{self._name}"
