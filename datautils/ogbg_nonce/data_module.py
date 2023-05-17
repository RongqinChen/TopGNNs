import os
import os.path as osp
from typing import Optional

import pandas as pd
import torch
from ogb.graphproppred import Evaluator
from torch.utils.data import DataLoader

from datautils.datamodule import DataModuleBase
from datautils.ogbg_nonce.dataset import OGBG_Nonce_Dataset


class OGBG_Nonce_DataModule(DataModuleBase):
    def __init__(
        self, name: str, batch_size: int, num_workers: int,
        transform: Optional[str] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False,
    ):
        super().__init__(transform, transform_fn_kwargs)
        self._dataset = OGBG_Nonce_Dataset(
            name, self._transform_fn, self._transform_fn_kwargs, reload)

        self._sizes = self._dataset.sizes
        self._evaluator = Evaluator(name=name)
        self._name = name

        split_dict = self._get_split()
        train_dataset = self._dataset[split_dict['train']]
        valid_dataset = self._dataset[split_dict['valid']]
        test_dataset = self._dataset[split_dict['test']]

        drop_last = 0 < len(train_dataset) % batch_size < (batch_size//2)
        self.train_loader = DataLoader(
            train_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=self._dataset[0].collate,
            drop_last=drop_last
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=self._dataset[0].collate
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=self._dataset[0].collate
        )

    @property
    def metric_name(self) -> str:
        return self._evaluator.eval_metric

    def evaluator(self, predicts, targets) -> float:
        # mean, std = self._graph_stat
        input_dict = {"y_true": targets,
                      "y_pred": predicts}
        result_dict = self._evaluator.eval(input_dict)
        result = result_dict[self.metric_name]
        return result

    def __repr__(self) -> str:
        return f"OGBG/{self._name.split('-')[-1]}"

    def _get_split(self, split_type=None):
        if split_type is None:
            split_type = self._dataset.meta_info['split']
        path = osp.join(self._dataset.root, 'split', split_type)
        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(
            osp.join(path, 'train.csv.gz'),
            compression='gzip', header=None).values.T[0]
        valid_idx = pd.read_csv(
            osp.join(path, 'valid.csv.gz'),
            compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(
            osp.join(path, 'test.csv.gz'),
            compression='gzip', header=None).values.T[0]
        split_dict = {
            'train': train_idx.tolist(),
            'valid': valid_idx.tolist(),
            'test': test_idx.tolist()
        }
        return split_dict

    @property
    def num_node_attributes(self):
        return self._sizes['num_node_attributes']

    @property
    def num_node_labels(self):
        return self._sizes['num_node_labels']

    @property
    def num_edge_attributes(self):
        return self._sizes['num_edge_attributes']

    @property
    def num_edge_labels(self):
        return self._sizes['num_edge_labels']

    @property
    def num_graph_attributes(self):
        return self._sizes['num_graph_attributes']

    @property
    def num_graph_labels(self):
        return self._sizes['num_graph_labels']
