import os
import os.path as osp
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch import from_numpy
from torch.utils.data import Subset
from tqdm import tqdm

from datautils.graph import Graph

from .reader import read_graph_dict


def sorted_eindex(edge_index, num_nodes, edge_feat):
    # edge_index = edge_index[:, edge_index[0, :] != edge_index[1, :]]
    hash = edge_index[0, :] * num_nodes + edge_index[1, :]
    indices = np.argsort(hash)
    edge_index = edge_index[:, indices]
    edge_feat = edge_feat[indices, :]
    return edge_index, edge_feat


class OGBG_Dataset(PygGraphPropPredDataset):
    def __init__(
        self,
        name: str,
        transform_fn: Optional[Callable[[Graph, Any], Graph]] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False,
    ):
        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._reload = reload
        self.config_path(name)
        super().__init__(name, "datasets/OGBG")
        self._data_list = self.data.uncollate()
        self.sizes = self.slices

    def process_vanilla_graph(self):
        gdict_list = read_graph_dict(
            self.raw_dir, add_inverse_edge=True, binary=self.binary
        )

        if self.task_type == "subtoken prediction":
            graph_label_notparsed = pd.read_csv(
                osp.join(self.raw_dir, "graph-label.csv.gz"),
                compression="gzip",
                header=None,
            ).values
            graph_label = [
                str(graph_label_notparsed[i][0]).split(" ")
                for i in range(len(graph_label_notparsed))
            ]

            for idx, g in enumerate(gdict_list):
                g["graph_label"] = graph_label[idx]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir,
                                               "graph-label.npz"))[
                    "graph_label"
                ]
            else:
                graph_label = pd.read_csv(
                    osp.join(self.raw_dir, "graph-label.csv.gz"),
                    compression="gzip",
                    header=None,
                ).values

            graph_label = torch.from_numpy(graph_label)
            has_nan = np.isnan(graph_label).any()
            if "classification" in self.task_type:
                if has_nan:
                    graph_label = graph_label.to(torch.float32)
                else:
                    graph_label = graph_label.to(torch.long)
            else:
                graph_label = graph_label.to(torch.float32)

            for i, g in enumerate(gdict_list):
                targets = graph_label[i].view(1, -1)
                if "classification" in self.task_type:
                    g["graph_label"] = targets
                else:
                    g["graph_attr"] = targets

        graph_list = []
        for g in gdict_list:
            edge_index, edge_feat = sorted_eindex(
                g["edge_index"], g["num_nodes"], g["edge_feat"]
            )
            graph_list.append(
                Graph(
                    g["num_nodes"],
                    from_numpy(edge_index),
                    None,
                    from_numpy(g["node_feat"]),
                    None,
                    from_numpy(edge_feat),
                    g.get("graph_attr", None),
                    g.get("graph_label", None),
                )
            )

        return graph_list

    @property
    def processed_paths(self) -> List[str]:
        return [self._processed_paths]

    def process(self):
        if (
            not self._reload
            and self.transed_fpath is not None
            and osp.exists(self.transed_fpath)
        ):
            print(f"Loading cached `{self.transed_fpath}` dataset ...")
        else:
            if os.path.exists(self.vanilla_path):
                print(f"Loading cached `{self.vanilla_path}` dataset ...")
                graph_batch, sizes = torch.load(self.vanilla_path)
                vanilla_graphs = graph_batch.uncollate()
            else:
                # process vanilla graphs
                vanilla_graphs = self.process_vanilla_graph()
                graph_batch = vanilla_graphs[0].collate(vanilla_graphs)
                sizes = graph_batch.get_sizes()
                print(f"Saving cache to `{self.vanilla_path}`")
                torch.save((graph_batch, sizes), self.vanilla_path)

            if self._transform_fn is None:
                print(f"Loading cached dataset from `{self.vanilla_path}` ...")
            else:
                tran_graph_list: List[Graph] = [
                    self._transform_fn(graph, **self._transform_fn_kwargs)
                    for graph in tqdm(
                        vanilla_graphs,
                        dynamic_ncols=True,
                        desc="Transforming graphs ...",
                    )
                ]
                tran_graph_batch = tran_graph_list[0].collate(tran_graph_list)
                torch.save((tran_graph_batch, sizes), self.transed_fpath)

    def config_path(self, name):
        name = name.replace("-", "_")
        processed_dir = f"datasets/OGBG/{name}/processed"
        self.vanilla_path = f"{processed_dir}/vanilla.pkl"
        if self._transform_fn is not None:
            transed_fname = f"{self._transform_fn.__name__}"
            if len(self._transform_fn_kwargs):
                t_arg_list = [
                    f"{key}={val}" 
                    for key, val in self._transform_fn_kwargs.items()
                ]
                t_args = ".".join(t_arg_list)
                transed_fname = f"{transed_fname}.{t_args}"
            self.transed_fpath = f"{processed_dir}/{transed_fname}.pkl"
            self._processed_paths = self.transed_fpath
        else:
            self.transed_fpath = None
            self._processed_paths = self.vanilla_path

    def __getitem__(self, idx: Union[int, list]) -> Union[Graph, List[Graph]]:
        if isinstance(idx, list):
            return Subset(self, idx)

        return self._data_list[idx]
