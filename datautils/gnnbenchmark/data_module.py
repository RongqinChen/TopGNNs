from typing import Optional

import torch as th
from torch.utils.data import DataLoader

from datautils.datamodule import DataModuleBase
from .dataset import MyGNNBenchmarkDataset


class GNNBenchmark_DataModule(DataModuleBase):
    def __init__(
        self, name: str,
        batch_size: int, num_workers: int,
        transform: Optional[str] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False,
        train_shuffle: bool = True,
    ):

        self._name = name
        super(GNNBenchmark_DataModule, self).__init__(
            transform, transform_fn_kwargs)

        train_dataset = MyGNNBenchmarkDataset(
            name, 'train', self._transform_fn,
            self._transform_fn_kwargs, reload)
        self.dataset_sizes = train_dataset.data_sizes

        valid_dataset = MyGNNBenchmarkDataset(
            name, 'valid', self._transform_fn,
            self._transform_fn_kwargs, reload)

        test_dataset = MyGNNBenchmarkDataset(
            name, 'test', self._transform_fn,
            self._transform_fn_kwargs, reload)

        collate = train_dataset.collate
        drop_last = 0 < len(train_dataset) % batch_size < 8
        self.train_loader = DataLoader(
            train_dataset, batch_size, shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=collate,
            drop_last=drop_last,
            pin_memory=True)

        self.valid_loader = DataLoader(
            valid_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True)

        self.test_loader = DataLoader(
            test_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True)

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
        return f"GNNBenchmark/{self._name}"
