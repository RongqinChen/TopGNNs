import os
import os.path as osp
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch import from_numpy
from torch.utils.data import Subset

from datautils.graph import Graph
from datautils.ogbg.reader import read_graph_dict


class OGBG_Nonce_Dataset(PygGraphPropPredDataset):
    def __init__(
        self, name: str,
        transform_fn: Optional[Callable[[Graph, Any], Graph]] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False
    ):
        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._reload = reload
        self._write_frequency = 1024
        self.config_path(name)
        super().__init__(name, 'datasets/OGBG')
        self.sizes = self.slices
        if self._transform_fn is None:
            self._data_list = self.data.uncollate()
        else:
            graphbatch, _ = torch.load(self.vanilla_path)
            self._data_list = graphbatch.uncollate()

    def process_vanilla_graph(self):
        # read pyg graph list
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = \
                self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = \
                self.meta_info['additional edge files'].split(',')

        gdict_list = read_graph_dict(
            self.raw_dir, add_inverse_edge=add_inverse_edge,
            additional_node_files=additional_node_files,
            additional_edge_files=additional_edge_files,
            binary=self.binary)

        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(
                osp.join(self.raw_dir, 'graph-label.csv.gz'),
                compression='gzip', header=None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ')
                           for i in range(len(graph_label_notparsed))]

            for idx, g in enumerate(gdict_list):
                g['graph_label'] = graph_label[idx]

        else:
            if self.binary:
                graph_label = np.load(
                    osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(
                    osp.join(self.raw_dir, 'graph-label.csv.gz'),
                    compression='gzip', header=None).values

            graph_label = torch.from_numpy(graph_label)
            has_nan = np.isnan(graph_label).any()
            if 'classification' in self.task_type:
                if has_nan:
                    graph_label = graph_label.to(torch.float32)
                else:
                    graph_label = graph_label.to(torch.long)
            else:
                graph_label = graph_label.to(torch.float32)

            for i, g in enumerate(gdict_list):
                targets = graph_label[i].view(1, -1)
                if 'classification' in self.task_type:
                    g['graph_label'] = targets
                else:
                    g['graph_attr'] = targets

        def sorted_eindex(edge_index, num_nodes):
            hash = edge_index[0, :] * num_nodes + edge_index[1, :]
            sorted = np.sort(hash)
            edge_index = np.stack((sorted // num_nodes, sorted % num_nodes))
            return edge_index

        graph_list = [
            Graph(g['num_nodes'],
                  from_numpy(sorted_eindex(g['edge_index'], g['num_nodes'])),
                  None, from_numpy(g['node_feat']),
                  None, from_numpy(g['edge_feat']),
                  g.get('graph_attr', None),
                  g.get('graph_label', None))
            for g in gdict_list
        ]
        return graph_list

    @property
    def processed_paths(self) -> List[str]:
        return [self.vanilla_path]

    def process(self):
        if not os.path.exists(self.vanilla_path):
            # process vanilla graphs
            vanilla_graphs = self.process_vanilla_graph()
            graph_batch = vanilla_graphs[0].collate(vanilla_graphs)
            print(f"Saving cache to `{self.vanilla_path}`")
            sizes = graph_batch.get_sizes()
            torch.save((graph_batch, sizes), self.vanilla_path)

    def config_path(self, name):
        name = name.replace('-', '_')
        processed_dir = f"datasets/OGBG/{name}/processed"
        self.vanilla_path = f"{processed_dir}/vanilla.pkl"

    def __getitem__(self, idx: Union[int, list]) \
            -> Union[Graph, List[Graph]]:

        if isinstance(idx, list):
            return Subset(self, idx)

        graph = self._data_list[idx]
        if self._transform_fn is None:
            return graph
        else:
            graph = self._data_list[idx]
            trans = self._transform_fn(graph, **self._transform_fn_kwargs)
            return trans
