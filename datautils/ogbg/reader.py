from ogb.io.read_graph_raw import read_csv_graph_raw, read_binary_graph_raw


def read_graph_dict(
        raw_dir, add_inverse_edge=False,
        additional_node_files=[],
        additional_edge_files=[], binary=False):

    if binary:
        # npz
        graph_list = read_binary_graph_raw(raw_dir, add_inverse_edge)
    else:
        # csv
        graph_list = read_csv_graph_raw(
            raw_dir, add_inverse_edge,
            additional_node_files, additional_edge_files)

    return graph_list
