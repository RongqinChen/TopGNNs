import os
import os.path as osp
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from ogb.lsc import PCQM4Mv2Dataset
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from datautils.graph import Graph

from .my_smiles2graph import my_smiles2graph


class MyPCQM4Mv2Dataset(Dataset):
    def __init__(
        self,
        transform_fn: Optional[Callable[[Graph, Any], Graph]] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False
    ):
        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._reload = reload
        self.config_path()
        self._raw_dataset = PCQM4Mv2Dataset(self.root_dir, only_smiles=True)
        self.graph_list, self.data_sizes = self.transform()
        split_dict = self._raw_dataset.get_idx_split()
        self.train_dataset = Subset(self, split_dict['train'])
        self.valid_dataset = Subset(self, split_dict['valid'])
        self.testdev_dataset = Subset(self, split_dict['test-dev'])
        self.testchallenge_dataset = Subset(self, split_dict['test-challenge'])

        data_df = pd.read_csv(osp.join(
            self._raw_dataset.folder, 'raw/data.csv.gz'))
        yarray = data_df['homolumogap'].values
        train_yarray = yarray[split_dict['train']]
        self.y_mean = np.mean(train_yarray)
        self.y_std = np.std(train_yarray)
        print('y_mean, y_std', self.y_mean, self.y_std)

    def config_path(self):
        self.root_dir = 'datasets/OGBG'
        processed_dir = f"{self.root_dir}/pcqm4m-v2/processed"
        self.vanilla_path = f"{processed_dir}/vanilla.pkl"
        if self._transform_fn is not None:
            transed_fname = f"{self._transform_fn.__name__}"
            if len(self._transform_fn_kwargs):
                t_arg_list = [f"{key}={val}" for key, val in
                              self._transform_fn_kwargs.items()]
                t_args = ".".join(t_arg_list)
                transed_fname = f"{transed_fname}.{t_args}"
            self.transed_fpath = f"{processed_dir}/{transed_fname}.pkl"
        else:
            self.transed_fpath = None

    def process_vanilla_graph(self):
        if os.path.exists(self.vanilla_path):
            print('Loading vanilla graphs ...')
            vanilla_graph_batch, data_sizes = torch.load(self.vanilla_path)
            graph_list = vanilla_graph_batch.uncollate()
        else:
            smiles_list = self._raw_dataset.graphs
            attr_list = self._raw_dataset.labels
            graph_list = []
            for smiles, attr in tqdm(
                    zip(smiles_list, attr_list),
                    'processing vanilla graphs', len(attr_list)):

                attr = torch.tensor(((attr,),), dtype=torch.float32)
                graph = my_smiles2graph(smiles, attr)
                graph_list.append(graph)

            vanilla_graph_batch = graph_list[0].collate(graph_list)
            data_sizes = vanilla_graph_batch.get_sizes()
            print(f"Saving cache to `{self.vanilla_path}`")
            torch.save((vanilla_graph_batch, data_sizes), self.vanilla_path)

        return graph_list, data_sizes

    def transform(self):
        if os.path.exists(self.transed_fpath):
            print('Loading transformed graphs ...')
            transed_graph_batch, data_sizes = torch.load(self.transed_fpath)
            transed_graph_list = transed_graph_batch.uncollate()
            return transed_graph_list, data_sizes
        else:
            vanilla_graph_list, data_sizes = self.process_vanilla_graph()
            if self._transform_fn is None:
                return vanilla_graph_list, data_sizes

            def _transform_fn(graph):
                transed = self._transform_fn(
                    graph, **self._transform_fn_kwargs)
                del graph
                return transed

            transed_graph_list = [
                _transform_fn(graph)
                for graph in tqdm(vanilla_graph_list, 'Transforming')
            ]
            del vanilla_graph_list
            transed_graph_batch = \
                transed_graph_list[0].collate(transed_graph_list)

            print(f"Saving cache to `{self.transed_fpath}`")
            torch.save((transed_graph_batch, data_sizes), self.transed_fpath)
            return transed_graph_list, data_sizes

    def __getitem__(self, idx: int) -> Graph:
        graph = self.graph_list[idx]
        graph.graph_attr = (graph.graph_attr - self.y_mean) / self.y_std
        return graph


if __name__ == "__main__":
    MyPCQM4Mv2Dataset()
