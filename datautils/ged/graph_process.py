# Derived from torch_geometric.datasets.GEDDataset
import glob
import os
import os.path as osp
import pickle as pkl
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch as th
from datautils.graph import Graph
from networkx import read_gexf, relabel_nodes
from torch_geometric.data import download_url, extract_tar, extract_zip
from torch_geometric.data.makedirs import makedirs
from tqdm import tqdm


class GED_Process:

    url = 'https://drive.google.com/uc?export=download&id={}'
    datasets = {
        'AIDS700nef': {
            'id': '10czBPJDEzEDI2tq7Z7mkBjLhj55F-a2z',
            'extract': extract_zip,
            'pickle': '1OpV4bCHjBkdpqI6H5Mg0-BqlA2ee2eBW',
        },
        'LINUX': {
            'id': '1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI',
            'extract': extract_tar,
            'pickle': '14FDm3NSnrBvB7eNpLeGy5Bz6FjuCSF5v',
        },
        'ALKANE': {
            'id': '1-LmxaWW3KulLh00YqscVEflbqr0g4cXt',
            'extract': extract_tar,
            'pickle': '15BpvMuHx77-yUGYgM27_sQett02HQNYu',
        },
        'IMDBMulti': {
            'id': '12QxZ7EhYA7pJiF4cO-HuE8szhSOWcfST',
            'extract': extract_zip,
            'pickle': '1wy9VbZvZodkixxVIOuRllC-Lp-0zdoYZ',
        },
    }
    # List of atoms contained in the AIDS700nef dataset:
    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te'
    ]

    def __init__(
        self, name: str,
        transform_fn: Optional[Callable] = None,
        transform_fn_kwargs: Optional[dict] = None,
        norm_ged: bool = True, reload: bool = False
    ):
        assert name in self.datasets.keys()
        self.name = name
        self.root = f"datasets/GED/{name}"

        self._norm_ged = norm_ged
        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._reload = reload
        super(GED_Process, self).__init__()
        self.config_path(transform_fn, transform_fn_kwargs)
        self.download()
        ged_path = osp.join(self.raw_dir, 'ged.pickle')
        with open(ged_path, 'rb') as f:
            self._ged = pkl.load(f)

        if transform_fn is not None and osp.exists(self.transed_fpath):
            graph_names_list, graph_dict, Ns_dict = \
                self.transform_graphs(
                    transform_fn, transform_fn_kwargs,
                    None, None, None)
        else:
            graph_names_list, graph_dict, Ns_dict = \
                self.generate_graphs()
            if transform_fn is not None and not osp.exists(self.transed_fpath):
                graph_names_list, graph_dict, Ns_dict = \
                    self.transform_graphs(
                        transform_fn, transform_fn_kwargs,
                        graph_names_list, graph_dict, Ns_dict)

        self.graph_names_list = graph_names_list
        self.graph_dict = graph_dict
        self.Ns_dict = Ns_dict

    def get_ged(self, idx_tup: Tuple[int, int]):
        ged = self._ged[idx_tup]
        if self._norm_ged:
            ns = self.Ns_dict
            Ns0 = ns[idx_tup[0]]
            Ns1 = ns[idx_tup[1]]
            ged = ged / (0.5 * (Ns0 + Ns1))

        return ged

    def config_path(
        self, transform_fn: Union[Callable, None],
        transform_fn_kwargs: Union[Dict, None]
    ):
        self.raw_dir = f"{self.root}/raw"
        self.raw_file_names = ["train", "test"]
        self.raw_paths = [f"{self.raw_dir}/{f}"
                          for f in self.raw_file_names]

        self.processed_dir = f"{self.root}/processed"
        self.vanilla_path = f"{self.processed_dir}/vanilla.pkl"

        if transform_fn is not None:
            transed_fname = f"{transform_fn.__name__}"
            if len(transform_fn_kwargs) > 0:
                t_arg_list = [f"{key}={val}" for key, val in
                              transform_fn_kwargs.items()]
                t_args = ".".join(t_arg_list)
                transed_fname = f"{transed_fname}.{t_args}"
            self.transed_fpath = f"{self.processed_dir}/{transed_fname}.pkl"
        else:
            self.transed_fpath = self.vanilla_path

    def download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.root)

        # Downloads the .tar/.zip file of the graphs and extracts them:
        name = self.datasets[self.name]['id']
        path = download_url(self.url.format(name), self.root)
        self.datasets[self.name]['extract'](path, self.root)
        os.rename(f"{self.root}/{self.name}", f"{self.raw_dir}")
        os.unlink(path)

        # Downloads the pickle file containing pre-computed GEDs:
        name = self.datasets[self.name]['pickle']
        path = download_url(self.url.format(name), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'ged.pickle'))

    def generate_graphs(self):
        if osp.exists(self.vanilla_path):
            if self._reload:
                option = input("Will you reload the data now? (y/N)\n").lower()
                if option == 'y':
                    os.system(f"rm -rf {self.vanilla_path}")
                else:
                    return
            else:
                print("Loading cached vanilla dataset")
                with open(self.vanilla_path, 'rb') as rbfile:
                    graph_names_list, graph_dict, Ns_dict = \
                        pkl.load(rbfile)
                return graph_names_list, graph_dict, Ns_dict

        graph_names_list = []
        graph_dict = {}
        Ns_dict = {}
        for r_path in self.raw_paths:
            # Find the paths of all raw graphs:
            names = glob.glob(osp.join(r_path, '*.gexf'))
            # Get sorted graph IDs given filename: 123.gexf -> 123
            names = [int(name.rsplit(os.sep, 1)[-1][:-5]) for name in names]
            graph_names_list.append(names)

        for split_idx, split in enumerate(['train', 'test']):
            split_dir = osp.join(self.raw_dir, split)
            for name in graph_names_list[split_idx]:
                # Reading the raw `*.gexf` graph:
                G = read_gexf(osp.join(split_dir, f'{name}.gexf'))
                # Mapping of nodes in `G` to a contiguous number:
                mapping = {name: j for j, name in enumerate(G.nodes())}
                G = relabel_nodes(G, mapping)
                Ns_dict[name] = G.number_of_nodes()

                edge_list = list(sorted(G.to_directed().edges))
                if len(edge_list) < 1:
                    continue
                src_nodes = [edge[0] for edge in edge_list]
                dst_nodes = [edge[1] for edge in edge_list]
                degree_dict = G.degree()
                if self.name == 'AIDS700nef':
                    label_list = [
                        (
                            self.types.index(info['type']),
                            1, degree_dict[node]
                        )
                        for node, info in G.nodes(data=True)
                    ]
                else:
                    label_list = [
                        (1, degree_dict[node])
                        for node, info in G.nodes(data=True)
                    ]

                node_label = th.LongTensor(label_list)
                edge_index = th.tensor([src_nodes, dst_nodes])
                graph = Graph(Ns_dict[name], edge_index,
                              node_attr=None,
                              node_label=node_label,
                              edge_attr=None,
                              edge_label=None,
                              graph_attr=None,
                              graph_label=None)
                graph_dict[name] = graph

        makedirs(self.processed_dir)
        with open(self.vanilla_path, 'wb') as wbfile:
            pkl.dump((graph_names_list, graph_dict, Ns_dict), wbfile)

        return graph_names_list, graph_dict, Ns_dict

    def transform_graphs(
        self, transform_fn, transform_fn_kwargs,
        graph_names_list, graph_dict, Ns_dict
    ):
        if not self._reload and osp.exists(self.transed_fpath):
            print(f"Loading cached `{self.transed_fpath}` dataset ...")
            with open(self.transed_fpath, 'rb') as rbfile:
                graph_names_list, graph_dict, Ns_dict =\
                    pkl.load(rbfile)
            return graph_names_list, graph_dict, Ns_dict

        trang_dict: Dict[int, Graph] = {
            idx: transform_fn(gdata, **transform_fn_kwargs)
            for idx, gdata in tqdm(
                graph_dict.items(), dynamic_ncols=True,
                desc=f"Transforming {self.transed_fpath} graphs",
            )
        }
        print(f"Caching `{self.transed_fpath}` dataset ...")
        data_to_save = (graph_names_list, trang_dict, Ns_dict)
        with open(self.transed_fpath, 'wb') as wbfile:
            pkl.dump(data_to_save, wbfile)
        return graph_names_list, trang_dict, Ns_dict

    def vanilla_graph_dict(self):
        with open(self.vanilla_path, 'rb') as rbfile:
            _, graph_dict, _ = pkl.load(rbfile)
        return graph_dict


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])
