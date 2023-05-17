from typing import Optional
from torch.utils.data import DataLoader

from ogb.lsc import PCQM4Mv2Evaluator
from datautils.datamodule import DataModuleBase
from datautils.pcqm4mv2.dataset import MyPCQM4Mv2Dataset


class PCQM4Mv2_DataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int, num_workers: int,
        transform: Optional[str] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False,
        train_shuffle: bool = True,
    ):
        super(PCQM4Mv2_DataModule, self).__init__(
            transform, transform_fn_kwargs)
        dataset = MyPCQM4Mv2Dataset(self._transform_fn,
                                    self._transform_fn_kwargs, reload)
        self.y_mean, self.y_std = dataset.y_mean, dataset.y_std
        self.dataset_sizes = dataset.data_sizes
        self._evaluator = PCQM4Mv2Evaluator()

        drop_last = len(dataset.train_dataset) % batch_size < (batch_size//2)
        self.train_loader = DataLoader(
            dataset.train_dataset, batch_size, shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=dataset[0].collate,
            drop_last=drop_last
        )
        self.valid_loader = DataLoader(
            dataset.valid_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset[0].collate,
            drop_last=drop_last
        )
        self.dev_loader = DataLoader(
            dataset.testdev_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset[0].collate,
            drop_last=drop_last
        )
        self.chall_loader = DataLoader(
            dataset.testchallenge_dataset, batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset[0].collate,
            drop_last=drop_last
        )
        self.test_loader = None

    @property
    def metric_name(self) -> str:
        return ['mae']

    def evaluator(self, predicts, targets) -> float:
        predicts = predicts.squeeze(1) * self.y_std + self.y_mean
        targets = targets.squeeze(1) * self.y_std + self.y_mean
        input_dict = {'y_pred': predicts, 'y_true': targets}
        result_dict = self._evaluator.eval(input_dict)
        return result_dict

    def whether_improve(self, valid_score_dict, best_valid_score_dict):
        if best_valid_score_dict is None:
            return True
        valid_score = valid_score_dict['mae']
        best_valid_score = best_valid_score_dict['mae']
        return valid_score > best_valid_score

    def __repr__(self) -> str:
        return "OGBG/PCQM4Mv2"

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
