# derived from https://raw.githubusercontent.com/rampasek/GraphGPS/main/graphgps/loader/dataset/peptides_functional.py  # noqa

import hashlib
import os
import os.path as osp
import pickle as pkl
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch as th
from ogb.utils import smiles2graph
from ogb.utils.url import decide_download
from rdkit import rdBase
from torch.utils.data import Subset
from torch_geometric.data import download_url
from tqdm import tqdm

from datautils.graph import Graph

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')


class PeptidesDataset():
    def __init__(
        self, name: str, transform_fn: Optional[Callable] = None,
        transform_fn_kwargs: Optional[dict] = None, reload: bool = True,
    ):
        """
        functional dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        ----------------------------------------------------------------------------

        structural dataset of 15,535 small peptides represented as their molecular
        graph (SMILES) with 11 regression targets derived from the peptide's
        3D structure.

        The original amino acid sequence representation is provided in
        'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
        of the dataset file, but not used here as any part of the input.

        The 11 regression targets were precomputed from molecule XYZ:
            Inertia_mass_[a-c]: The principal component of the inertia of the
                mass, with some normalizations. Sorted
            Inertia_valence_[a-c]: The principal component of the inertia of the
                Hydrogen atoms. This is basically a measure of the 3D
                distribution of hydrogens. Sorted
            length_[a-c]: The length around the 3 main geometric axis of
                the 3D objects (without considering atom types). Sorted
            Spherocity: SpherocityIndex descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
            Plane_best_fit: Plane of best fit (PBF) descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcPBF

        """  # noqa
        super(PeptidesDataset, self).__init__()
        assert name in {'functional', 'structural'}
        self._name = name
        if name == "functional":
            self.folder = 'datasets/Peptides/functional'
            self.url = 'https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1'  # noqa
            self.version = '701eb743e899f4d793f0e13c8fa5a1b4'  # MD5 hash of the intended dataset file  # noqa
            self.url_stratified_split = 'https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1'  # noqa
            self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'  # noqa
            self.raw_file_path = f"{self.folder}/peptide_multi_class_dataset.csv.gz"  # noqa
            self.split_file_path = f"{self.folder}/splits_random_stratified_peptide.pickle"  # noqa
        else:
            self.folder = 'datasets/Peptides/structural'
            self.url = 'https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1'  # noqa
            self.version = '9786061a34298a0684150f2e4ff13f47'  # MD5 hash of the intended dataset file # noqa
            self.url_stratified_split = 'https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1'  # noqa
            self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'  # noqa
            self.raw_file_path = f"{self.folder}/peptide_structure_dataset.csv.gz"  # noqa
            self.split_file_path = f"{self.folder}/splits_random_stratified_peptide_structure.pickle"  # noqa

        self._reload = reload
        self.download()
        self.processed_dir = f"{self.folder}/processed"
        os.makedirs(self.processed_dir, exist_ok=True)
        self.vanilla_path = f"{self.processed_dir}/vanilla_graphs.pt"
        self.graph_list, self.sizes = self.generate_vanilla_graphs()

        if transform_fn is not None:
            self._transform_fn = transform_fn
            self._transform_fn_kwargs = transform_fn_kwargs
            transed_name = f"{transform_fn.__name__}"
            if len(transform_fn_kwargs) > 0:
                t_arg_list = [f"{key}={val}" for key, val in
                              transform_fn_kwargs.items()]
                t_args = ".".join(t_arg_list)
                transed_name = f"{transed_name}.{t_args}"
            self.transed_path = f"{self.processed_dir}/{transed_name}.pt"
            self.graph_list, self.sizes = self.transform_graphs(self.graph_list)  # noqa

    def download(self):
        if osp.exists(self.raw_file_path) and osp.exists(self.split_file_path):
            return

        if decide_download(self.url):
            print('Downloading ...')
            path = download_url(self.url, self.folder)
            # Save to disk the MD5 hash of the downloaded file.
            hash_md5 = hashlib.md5()
            with open(path, 'rb') as f:
                buffer = f.read()
                hash_md5.update(buffer)
            hash = hash_md5.hexdigest()
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.folder)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print('Stop download.')
            exit(-1)

    def get_whole_targets(self) -> th.Tensor:
        data_df = pd.read_csv(self.raw_file_path)
        if self._name == 'functional':
            labels = data_df['labels']
            labels = labels.apply(eval)
            labels = labels.apply(np.array)
            labels = np.stack(labels.to_list())
            labels = th.from_numpy(labels).to(th.float32)
            return labels
        else:
            target_names = [
                'Inertia_mass_a', 'Inertia_mass_b', 'Inertia_mass_c',
                'Inertia_valence_a', 'Inertia_valence_b', 'Inertia_valence_c',
                'length_a', 'length_b', 'length_c',
                'Spherocity', 'Plane_best_fit'
            ]
            targets = data_df[target_names].to_numpy().astype(np.float32)
            targets = th.from_numpy(targets)
            means = th.mean(targets, axis=0, keepdims=True)
            stds = th.std(targets, axis=0, keepdims=True)
            targets = (targets - means) / stds
            return targets

    def generate_vanilla_graphs(self):
        if osp.exists(self.vanilla_path) and self._reload:
            option = input("Will you reload vanilla data now? (y/N)\n").lower()
            if option == 'y':
                os.system(f"rm -rf {self.vanilla_path}")

        if osp.exists(self.vanilla_path):
            graphbatch, sizes = torch.load(self.vanilla_path)
            graph_list = graphbatch.uncollate()
            return graph_list, sizes

        def sorted_eindex(edge_index, num_nodes):
            hash = edge_index[0, :] * num_nodes + edge_index[1, :]
            argsort = np.argsort(hash)
            sorted_edge_index = edge_index[:, argsort]
            return sorted_edge_index, argsort

        data_df = pd.read_csv(self.raw_file_path)
        smiles_list = data_df['smiles']
        # Normalize to zero mean and unit standard deviation.
        whole_targets = self.get_whole_targets()
        graph_list = []
        node_attr = None
        edge_attr = None
        graph_attr = graph_label = None
        for mol_idx in tqdm(
                range(len(smiles_list)), dynamic_ncols=True,
                desc='Converting SMILES strings into graphs...'):
            smiles = smiles_list[mol_idx]
            graph = smiles2graph(smiles)
            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])
            num_nodes = graph['num_nodes']
            node_label = th.from_numpy(graph['node_feat']).to(th.long)
            sorted_edge_index, argsort = sorted_eindex(
                graph['edge_index'], num_nodes)
            edge_index = th.from_numpy(sorted_edge_index).to(th.long)
            edge_feat = graph['edge_feat'][argsort, :]
            edge_label = th.from_numpy(edge_feat).to(th.long)
            graph_tgts = whole_targets[mol_idx: mol_idx+1, :]
            if self._name == 'functional':
                graph_label = graph_tgts
            else:
                graph_attr = graph_tgts
            graph = Graph(num_nodes, edge_index, node_attr, node_label,
                          edge_attr, edge_label, graph_attr, graph_label)
            graph_list.append(graph)

        graphbatch = graph.collate(graph_list)
        sizes = graphbatch.get_sizes()
        torch.save((graphbatch, sizes), self.vanilla_path)
        return graph_list, sizes

    def transform_graphs(self, vanilla_graph_list):
        if osp.exists(self.transed_path):
            if self._reload:
                option = input(f"Will you reload `{self.transed_path}`"
                               " data now? (y/N)\n").lower()
                if option == 'y':
                    os.system(f"rm -rf {self.transed_path}")
                else:
                    graphbatch, sizes = torch.load(self.transed_path)
                    graph_list = graphbatch.uncollate()
                    return graph_list, sizes
            else:
                graphbatch, sizes = torch.load(self.transed_path)
                graph_list = graphbatch.uncollate()
                return graph_list, sizes

        num_graphs = len(vanilla_graph_list)
        transed_graph_list = []
        saving_file = self.transed_path.rsplit('/', 1)[-1]
        for g_idx in tqdm(
            range(num_graphs), total=num_graphs,
            desc=f'Transforming graphs ({saving_file})',
            dynamic_ncols=True
        ):
            graph = vanilla_graph_list[g_idx]
            transed_graph = self._transform_fn(
                graph, **self._transform_fn_kwargs)
            transed_graph_list.append(transed_graph)

        transed_graphbatch = transed_graph.collate(transed_graph_list)
        sizes = transed_graphbatch.get_sizes()
        torch.save((transed_graphbatch, sizes), self.transed_path)
        return transed_graph_list, sizes

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        with open(self.split_file_path, 'rb') as f:
            split_dict = pkl.load(f)

        split_dict = {
            key: val.tolist()
            for key, val in split_dict.items()
        }
        return split_dict

    def __getitem__(self, idx: Union[int, list]) \
            -> Union[Graph, List[Graph]]:
        if isinstance(idx, list):
            return Subset(self, idx)

        return self.graph_list[idx]

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()


if __name__ == "__main__":
    PeptidesDataset('structural')
