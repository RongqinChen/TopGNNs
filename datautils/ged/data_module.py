from itertools import combinations
from random import shuffle
from typing import Optional
import torch as th
from torch.utils.data import DataLoader

from datautils.datamodule import DataModuleBase

from .graph_process import GED_Process
from .graphpair_dataset import GED_GraphPairCollator, GED_GraphPairDataset


class GED_DataModule(DataModuleBase):
    def __init__(
        self, name: str, batch_size: int, num_workers: int,
        transform: Optional[str] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False, norm_ged: bool = True
    ):
        super(GED_DataModule, self).__init__(
            transform, transform_fn_kwargs)

        self._name = name
        process = GED_Process(
            name, self._transform_fn, self._transform_fn_kwargs,
            norm_ged, reload
        )

        graph_dict = process.graph_dict
        train_graphs = process.graph_names_list[0]
        nodel_label_max = graph_dict[train_graphs[0]].node_label.max(0)[0]
        for gname in train_graphs:
            graph = graph_dict[gname]
            nodel_label_max2 = graph.node_label.max(0)[0]
            nodel_label_max = th.max(nodel_label_max, nodel_label_max2)
        test_graphs = process.graph_names_list[1]
        for gname in test_graphs:
            graph = graph_dict[gname]
            nodel_label_max2 = graph.node_label.max(0)[0]
            nodel_label_max = th.max(nodel_label_max, nodel_label_max2)
        self._nodel_label_size = (nodel_label_max+1).tolist()

        pair_list = list(combinations(train_graphs, 2))
        shuffle(pair_list)
        ten_percent = int(len(pair_list) * 0.1)
        train_pair_list = pair_list[ten_percent:]
        val_pair_list = pair_list[:ten_percent]
        test_pair_list = [
            (name1, name2)
            for name1 in test_graphs for name2 in train_graphs]

        get_ged = process.get_ged
        train_ged_list = list(map(get_ged, train_pair_list))
        val_ged_list = list(map(get_ged, val_pair_list))
        test_ged_list = list(map(get_ged, test_pair_list))

        train_pair_dataset = GED_GraphPairDataset(
            train_pair_list, train_ged_list)
        val_pair_dataset = GED_GraphPairDataset(
            val_pair_list, val_ged_list)
        test_pair_dataset = GED_GraphPairDataset(
            test_pair_list, test_ged_list)

        collator = GED_GraphPairCollator(process.graph_dict)

        self.train_loader = DataLoader(
            train_pair_dataset, batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collator.collate)
        self.valid_loader = DataLoader(
            val_pair_dataset, batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collator.collate)
        self.test_loader = DataLoader(
            test_pair_dataset, batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collator.collate)

    @property
    def metric_name(self) -> str:
        return "mae"

    def evaluator(self, predicts, targets) -> float:
        return th.mean(th.abs(predicts - targets)).item()

    def __repr__(self) -> str:
        return f"GED/{self._name}"

    @property
    def num_node_attributes(self):
        return None

    @property
    def num_node_labels(self):
        return self._nodel_label_size

    @property
    def num_edge_attributes(self):
        return None

    @property
    def num_edge_labels(self):
        return None

    @property
    def num_graph_attributes(self):
        return 1

    @property
    def num_graph_labels(self):
        return None
