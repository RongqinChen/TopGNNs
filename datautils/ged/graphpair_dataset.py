from typing import Dict, List, Tuple

from torch import FloatTensor, LongTensor
from torch.utils.data.dataset import Dataset

from datautils.graph import Graph


class GED_GraphPairDataset(Dataset):

    def __init__(self, pair_name_list: List[Tuple[int, int]],
                 ged_list: List[float]) -> None:

        super().__init__()
        self._pair_name_list = pair_name_list
        self._ged_list = ged_list

    def __len__(self):
        return len(self._pair_name_list)

    def __getitem__(self, index: int) -> Tuple[Tuple[int, int], float]:
        pair_name = self._pair_name_list[index]
        ged = self._ged_list[index]
        return pair_name, ged


class GED_GraphPairCollator:
    def __init__(
        self,
        graph_dict: Dict[int, Graph],
    ) -> None:
        self.graph_dict = graph_dict

    def collate(
        self,
        tuple_list: List[Tuple[Tuple[int, int], float]]
    ):
        ged_list = [[gtuple[1]] for gtuple in tuple_list]
        ged_ts = FloatTensor(ged_list)
        gidx_list = list({gidx for gtuple in tuple_list for gidx in gtuple[0]})
        gobj_list = [self.graph_dict[gidx] for gidx in gidx_list]

        gbatch = gobj_list[0].collate(gobj_list)
        gidxs_tup = ([
            gidx_list.index(gtuple[0][0])
            for gtuple in tuple_list
        ], [
            gidx_list.index(gtuple[0][1])
            for gtuple in tuple_list
        ])
        gidxs = LongTensor(gidxs_tup)
        return ged_ts, gbatch, gidxs
