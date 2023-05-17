# distutils: language=c++
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

cpdef pyx_make_TPF(
    int tree_height,
    int num_nodes,
    vector[int] n2n_src_vec, # ordered
    vector[int] n2n_dst_vec,
    bool epath,
):
    """
    dst:
        edge_k
        n2e_dst

    source:
        node_k
        edge_k
        edge

    num_nodes:
        node_k
        edge
        node

    """

    #### dicts to return
    cdef map[string, vector[int]] dst_dict = map[string, vector[int]]()
    cdef map[string, vector[int]] source_dict = map[string, vector[int]]()
    cdef map[string, int] num_nodes_dict = map[string, int]()

    cdef unsigned int num_n2n = n2n_src_vec.size()
    
    # compute the relation between edge and n2n
    cdef vector[int] n2n_2_edge_vec = vector[int]()
    cdef vector[int] edge_2_n2n_vec = vector[int]()
    cdef map[int, int] nnhash_2_edge = map[int, int]()
    cdef unsigned int n2n_idx, n2n_src, n2n_dst, nnhash, edge
    for n2n_idx in range(num_n2n):
        n2n_src = n2n_src_vec[n2n_idx]
        n2n_dst = n2n_dst_vec[n2n_idx]
        if n2n_src < n2n_dst:
            edge = nnhash_2_edge.size()
            nnhash = n2n_src * num_nodes + n2n_dst
            nnhash_2_edge[nnhash] = edge
            edge_2_n2n_vec.push_back(n2n_idx)
        else:
            nnhash = n2n_dst * num_nodes + n2n_src
            edge = nnhash_2_edge[nnhash]
        n2n_2_edge_vec.push_back(edge)

    # height 1
    # dst_dict[b'edge_1'] = n2n_src_vec  # ordered
    source_dict[b'edge_1'] = n2n_2_edge_vec
    source_dict[b'edge'] = edge_2_n2n_vec
    dst_dict[b'n2e'] = n2n_2_edge_vec
    # source_dict[b'node_1'] = n2n_dst_vec
    # num_nodes_dict[b'node_1'] = n2n_src_vec.size()

    #### n2n_ptrs
    cdef vector[int] n2n_ptrs = vector[int]()
    make_n2n_ptrs(num_nodes, &n2n_src_vec, &n2n_ptrs)
    
    #### preparation for higher height
    cdef vector[vector[int]] pre_pathNode_vec_vec = vector[vector[int]]()
    cdef vector[vector[int]] pathNode_vec_vec
    cdef vector[int] prePath_vec  # prefix path (i.e. upper-level-atom)
    cdef vector[int] edge_h_oidx
    cdef vector[int] node_h_oidx

    make_pathNode_vec_vec(&n2n_src_vec, &n2n_dst_vec, &pre_pathNode_vec_vec)

    #### higher height
    for h in range(2, tree_height + 1):
        pathNode_vec_vec = vector[vector[int]]()
        prePath_vec = vector[int]()
        edge_h_oidx = vector[int]()
        node_h_oidx = vector[int]()

        epath_tree_search(
            # input
            &pre_pathNode_vec_vec,
            &n2n_2_edge_vec, 
            &n2n_dst_vec,
            &n2n_ptrs,
            epath,
            # output
            &pathNode_vec_vec,
            &prePath_vec,
            &edge_h_oidx,
            &node_h_oidx,
        )
        
        dst_dict[f'edge_{h}'.encode('ascii')] = prePath_vec  # ordered
        source_dict[f'edge_{h}'.encode('ascii')] = edge_h_oidx
        source_dict[f'node_{h}'.encode('ascii')] = node_h_oidx
        num_nodes_dict[f'node_{h}'.encode('ascii')] = node_h_oidx.size()
        pre_pathNode_vec_vec = pathNode_vec_vec

    return dst_dict, source_dict, num_nodes_dict


cdef make_n2n_ptrs(
    unsigned int num_nodes, 
    vector[int] * n2n_src_vec,
    # output
    vector[int] * n2n_ptrs,
):
    # make node-to-node pointers
    cdef unsigned int num_n2n = n2n_src_vec.size()
    cdef int next_node = 0

    for idx in range(num_n2n):
        src = n2n_src_vec.at(idx)
        while src >= next_node:
            n2n_ptrs.push_back(idx)
            next_node += 1

    while n2n_ptrs.size() < num_nodes + 1:
        n2n_ptrs.push_back(num_n2n)


cdef make_pathNode_vec_vec(
    vector[int] * n2n_src_vec, 
    vector[int] * n2n_dst_vec,
    # output
    vector[vector[int]] * pathNode_vec_vec,
):
    cdef vector[int] nodes
    for idx in range(n2n_src_vec.size()):
        nodes = vector[int]()
        nodes.push_back(n2n_src_vec.at(idx))
        nodes.push_back(n2n_dst_vec.at(idx))
        pathNode_vec_vec.push_back(nodes)


cdef epath_tree_search(
    # input
    vector[vector[int]] * pre_pathNode_vec_vec,
    vector[int] * n2n_2_edge_vec, 
    vector[int] * n2n_dst_vec,
    vector[int] * n2n_ptrs,
    bool epath,
    # output
    vector[vector[int]] * pathNode_vec_vec,
    vector[int] * prePath_vec,
    vector[int] * edge_h_oidx,
    vector[int] * node_h_oidx,
):
    
    cdef vector[int] pathNode_vec
    cdef int ptr_start, ptr_stop
    cdef int n_oidx, e_oidx

    for prePathIdx in range(pre_pathNode_vec_vec.size()):
        pre_pathNode_vec = pre_pathNode_vec_vec.at(prePathIdx)
        relay_node = pre_pathNode_vec.back()
        if relay_node == pre_pathNode_vec.front():  # formed a ring
            continue

        ptr_start = n2n_ptrs.at(relay_node)
        ptr_stop = n2n_ptrs.at(relay_node + 1)
        for ptr in range(ptr_start, ptr_stop):
            n_oidx = n2n_dst_vec.at(ptr)
            if can_form_epath(&pre_pathNode_vec, n_oidx, epath) == 1:
                pathNode_vec = vector[int](pre_pathNode_vec)
                pathNode_vec.push_back(n_oidx)
                pathNode_vec_vec.push_back(pathNode_vec)
                prePath_vec.push_back(prePathIdx)
                e_oidx = n2n_2_edge_vec.at(ptr)
                edge_h_oidx.push_back(e_oidx)
                node_h_oidx.push_back(n_oidx)


cdef int can_form_epath(vector[int] * vec, int val, bool epath):
    cdef int num = vec.size()
    if epath:
        if vec.size() == 2:
            for idx in range(num):
                if vec.at(idx) == val:
                    return -1
        else:
            for idx in range(1, num):
                if vec.at(idx) == val:
                    return -1
        return 1
    else:
        for idx in range(num):
            if vec.at(idx) == val:
                return -1
        return 1
