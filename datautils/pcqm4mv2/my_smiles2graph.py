from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector) 
from rdkit import Chem
import numpy as np
import torch
from datautils.graph import Graph


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def my_smiles2graph(smiles_string, graph_attr) -> Graph:
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    node_label_list = []
    for atom in mol.GetAtoms():
        node_label_list.append(atom_to_feature_vector(atom))
    node_label_arr = np.array(node_label_list, dtype=np.int64)
    node_label = torch.from_numpy(node_label_arr)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    bonds = list(mol.GetBonds())
    if len(bonds) > 0:  # mol has bonds
        edges_list = []
        edge_label_list = []
        for bond in bonds:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_label_list.append(edge_feature)
            edges_list.append((j, i))
            edge_label_list.append(edge_feature)

        edge_argsort = argsort(edges_list)
        edges_list = [edges_list[idx] for idx in edge_argsort]
        edge_label_list = [edge_label_list[idx] for idx in edge_argsort]

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_index = torch.from_numpy(edge_index)
        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_label = np.array(edge_label_list, dtype=np.int64)
        edge_label = torch.from_numpy(edge_label)

    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        edge_label = torch.empty((0, num_bond_features), dtype=torch.int64)

    graph = Graph(len(node_label_list), edge_index, None,
                  node_label, None, edge_label, graph_attr, None)
    return graph


if __name__ == '__main__':
    graph_attr = torch.tensor((1))
    graph = my_smiles2graph(
        'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5', graph_attr)
    print(graph)
