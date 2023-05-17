from typing import Optional

import torch as th
from torch.utils.data import DataLoader
from datautils.datamodule import DataModuleBase
from sklearn.metrics import average_precision_score
from .dataset import PeptidesDataset


class PeptidesDataModule(DataModuleBase):
    def __init__(
        self, name: str, batch_size: int, num_workers: int,
        transform: Optional[str] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False,
        train_shuffle: bool = True,
    ):
        self.name = name
        super(PeptidesDataModule, self).__init__(
            transform, transform_fn_kwargs)
        dataset = PeptidesDataset(name, self._transform_fn,
                                  self._transform_fn_kwargs, reload)
        self.dataset_sizes = dataset.sizes
        split_dict = dataset.get_idx_split()
        self.train_dataset = dataset[split_dict['train']]
        drop_last = 0 < len(self.train_dataset) % batch_size < (batch_size//2)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size, shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=dataset[0].collate,
            drop_last=drop_last
        )
        self.valid_dataset = dataset[split_dict['val']]
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset[0].collate,
        )
        self.test_dataset = dataset[split_dict['test']]
        self.test_loader = DataLoader(
            self.test_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset[0].collate,
        )

    @property
    def metric_name(self) -> str:
        return ["AP"] if self.name == 'functional' else ['MAE']

    def whether_improve(self, valid_score_dict, best_valid_score_dict):
        if best_valid_score_dict is None:
            return True
        valid_score = valid_score_dict[self.metric_name[0]]
        best_valid_score = best_valid_score_dict[self.metric_name[0]]
        if self.name == 'functional':
            return valid_score > best_valid_score
        else:
            return valid_score < best_valid_score

    def evaluator(self, predicts: th.Tensor, targets: th.Tensor) -> float:
        if self.name == 'functional':
            ap = average_precision_score(targets.numpy(), predicts.numpy())
            return {'AP': ap}
        else:
            mae = th.mean(th.abs(predicts-targets)).item()
            return {'MAE': mae}

    def __repr__(self) -> str:
        return f"Pep-{self.name}"

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
