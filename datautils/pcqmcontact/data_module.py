from typing import Optional

import torch as th
from torch.utils.data import DataLoader

from datautils.datamodule import DataModuleBase
from .dataset import PCQMContactDataset
from sklearn.metrics import label_ranking_average_precision_score


class PCQMContactDataModule(DataModuleBase):
    def __init__(
        self, batch_size: int, num_workers: int,
        transform: Optional[str] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False,
        train_shuffle: bool = True,
    ):
        super(PCQMContactDataModule, self).__init__(
            transform, transform_fn_kwargs)

        def couple_collate(couple_list):
            input_graph_list = [couple[0] for couple in couple_list]
            label_graph_list = [couple[1] for couple in couple_list]
            input_graph_batch = input_graph_list[0].collate(input_graph_list)
            label_graph_batch = label_graph_list[0].collate(label_graph_list)
            label_graph_batch.edge_label = label_graph_batch.edge_label.float()
            input_graph_batch.pin_memory()
            label_graph_batch.pin_memory()
            return input_graph_batch, label_graph_batch

        test_dataset = PCQMContactDataset(
            'test', self._transform_fn,
            self._transform_fn_kwargs, reload)
        self.test_loader = DataLoader(
            test_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=couple_collate,
            pin_memory=True)

        val_dataset = PCQMContactDataset(
            'val', self._transform_fn,
            self._transform_fn_kwargs, reload)
        self.valid_loader = DataLoader(
            val_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=couple_collate,
            pin_memory=True)

        train_dataset = PCQMContactDataset(
            'train', self._transform_fn,
            self._transform_fn_kwargs, reload)
        self.dataset_sizes = train_dataset.data_sizes
        drop_last = 0 < len(train_dataset) % batch_size < 8
        self.train_loader = DataLoader(
            train_dataset, batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=couple_collate,
            drop_last=drop_last,
            pin_memory=True)

    @property
    def metric_name(self) -> str:
        return ["MRR"]

    def evaluator(self, predicts: th.Tensor, targets: th.Tensor) -> float:
        MRR = label_ranking_average_precision_score(targets, predicts)
        return {'MRR': MRR}

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
        return "PCQMContact"
