from ogb.graphproppred import Evaluator
import os
import os.path as osp
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datautils.datamodule import DataModuleBase
from datautils.ogbg.dataset import OGBG_Dataset


class OGBG_DataModule(DataModuleBase):
    def __init__(
        self, name: str,
        batch_size: int, num_workers: int,
        transform: Optional[str] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False,
        train_shuffle: bool = True,
    ):
        super(OGBG_DataModule, self).__init__(transform, transform_fn_kwargs)
        self._dataset = OGBG_Dataset(name, self._transform_fn,
                                     self._transform_fn_kwargs, reload)
        self.dataset_sizes = self._dataset.sizes
        self._name = name
        self._evaluator = Evaluator(name=name)

        split_dict = self._get_split()
        self.train_dataset = self._dataset[split_dict['train']]
        self.valid_dataset = self._dataset[split_dict['valid']]
        self.test_dataset = self._dataset[split_dict['test']]

        drop_last = 0 < len(self.train_dataset) % batch_size < (batch_size//2)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size, shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=self._dataset._data_list[0].collate,
            drop_last=drop_last
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=self._dataset._data_list[0].collate
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=self._dataset._data_list[0].collate
        )

        fpath = 'datautils/ogbg/data_info.csv'
        data_pf = pd.read_csv(fpath, index_col=[0])
        dname = name.replace('_', '-')
        self.num_tasks = int(data_pf.loc['num tasks', dname])
        self.task_type = data_pf.loc['task type', dname]
        self.eval_metric = data_pf.loc['eval metric', dname]

    @property
    def metric_name(self) -> str:
        return [self._evaluator.eval_metric]

    def evaluator(self, predicts, targets) -> float:
        input_dict = {"y_true": targets, "y_pred": predicts}
        result_dict = self._evaluator.eval(input_dict)
        return result_dict

    def whether_improve(self, valid_score_dict, best_valid_score_dict):
        if best_valid_score_dict is None:
            return True
        valid_score = valid_score_dict[self.metric_name[0]]
        best_valid_score = best_valid_score_dict[self.metric_name[0]]
        return valid_score > best_valid_score

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
