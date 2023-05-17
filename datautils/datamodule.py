from typing import Optional

import transform_fns


class DataModuleBase:
    def __init__(self, transform: Optional[str] = None,
                 transform_fn_kwargs: Optional[dict] = None,
                 ) -> None:

        super().__init__()

        if transform is not None:
            try:
                transform_fn = getattr(transform_fns, transform)
            except Exception:
                raise KeyError(f"Invalid transform {transform}")

            if transform_fn_kwargs is None:
                transform_fn_kwargs = {}
            else:
                transform_fn_kwargs = {
                    key: transform_fn_kwargs[key]
                    for key in sorted(transform_fn_kwargs.keys())
                }
        else:
            transform_fn = None
            transform_fn_kwargs = {}

        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self.dataset_sizes = {}
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    @property
    def num_node_attributes(self):
        raise NotImplementedError

    @property
    def num_node_labels(self):
        raise NotImplementedError

    @property
    def num_edge_attributes(self):
        raise NotImplementedError

    @property
    def num_edge_labels(self):
        raise NotImplementedError

    @property
    def num_graph_attributes(self):
        raise NotImplementedError

    @property
    def num_graph_labels(self):
        raise NotImplementedError

    @property
    def metric_name(self) -> list[str]:
        raise NotImplementedError

    def evaluator(self, predicts, targets) -> dict:
        raise NotImplementedError

    def whether_improve(self, valid_score_dict, best_valid_score_dict):
        raise NotImplementedError
