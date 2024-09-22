import torch
import numpy as np
import pandas as pd
import math

from src.data.utils.geometry import calc_dist_mat


def get_noise(input, var=0.005, dim=-1):
    return (var**0.5)*torch.randn(input.shape)

def get_inter_intra_edges(N, num_atoms):
    num_res = N // num_atoms
    #print(N, num_res, num_atoms)
    edges_all = torch.zeros((N, N)).long()
    # edge 1: intra-resides edges
    intra_edge_indices_rows = torch.tensor([t for t in range(N)]).repeat_interleave(num_atoms)
    intra_edge_indices_cols = torch.tensor([t for t in range(N)]).view(num_res, num_atoms)
    intra_edge_indices_cols = torch.repeat_interleave(intra_edge_indices_cols,
                                                        num_atoms, dim=0).flatten()
    intra_index_tensor = torch.stack([intra_edge_indices_rows, intra_edge_indices_cols],
                                dim=0).permute(1, 0).long()
    intra_edge_token = torch.ones(intra_index_tensor.shape[0]).long()
    edges_all.index_put_(tuple(intra_index_tensor.t()), intra_edge_token)
    return edges_all


def get_fragmented_partner_pos_indices(indices_partner_1, indices_partner_2,
                                       num_atoms, offset=100):
    res_pos_indices = torch.cat([indices_partner_1, indices_partner_2 + offset
                                ], dim=0)
    pos_indices = res_pos_indices.repeat_interleave(num_atoms)
    return pos_indices


def get_node_and_edge_positional_embeddings(node_indices,
                                            frag_positions,
                                            n_pe_nodes,
                                            n_pe_edges,
                                            edge_distance,
                                            distance_threshold=None):

    frag_positions = frag_positions.view(-1, 1)
    # Map selected node indices (in case of topk neighbors) to
    #  sequence positions (frag_positions)

    nn_positions = torch.gather(
        frag_positions.expand(-1, node_indices.shape[1]), 0, node_indices)

    pe_edges = get_positional_embeddings(nn_positions,
                                         frag_positions,
                                         num_embeddings=n_pe_edges)

    if distance_threshold is not None:
        pe_edges = \
            pe_edges[(edge_distance <=
                      distance_threshold).unsqueeze_(-1).expand(pe_edges.size())
                    ]

    pe_nodes = get_positional_embeddings_nodes(frag_positions,
                                               num_embeddings=n_pe_nodes)

    return pe_nodes, pe_edges


def get_src_dst_nodes(node_indices_p0, node_indices_p1, p0_contact_indices,
                      p1_contact_indices, edist_p0, edist_p1,
                      distance_threshold):
    def get_intra_src_dst(node_indices, edist):
        src = []
        dst = []
        for i in range(node_indices.shape[0]):
            for j in range(node_indices.shape[1]):
                #Message passing from neighbors TO node
                if edist[i, j] <= distance_threshold:
                    dst.append(i)
                    src.append(node_indices[i, j].item())
        return (src, dst)

    src_dst_p0 = get_intra_src_dst(node_indices_p0, edist_p0)

    src_dst_p1 = get_intra_src_dst(node_indices_p1, edist_p1)

    src_p0p1 = []
    dst_p0p1 = []
    for i in list(p0_contact_indices):
        for j in list(p1_contact_indices):
            src_p0p1.append(i.item())
            dst_p0p1.append(j.item())

    return src_dst_p0, src_dst_p1, (src_p0p1, dst_p0p1)


def make_simple_knn_graph(p0, topk_p0, distance_threshold, d_count, real_dist,
                          real_valued_inverted_distance, include_orientations,
                          exclude_self_edge):
    dist_mat_p0_inputs = p0.dist_angle_mat_inputs[0, :, :].squeeze_(0)

    if exclude_self_edge:
        (edist_p0, node_indices_p0) = \
            torch.topk(dist_mat_p0_inputs,
                        min(dist_mat_p0_inputs.shape[0],topk_p0+1),
                        dim=1, largest=False)
        edist_p0 = edist_p0[:, 1:]
        node_indices_p0 = node_indices_p0[:, 1:]

    else:
        (edist_p0, node_indices_p0) = \
            torch.topk(dist_mat_p0_inputs,\
                        min(dist_mat_p0_inputs.shape[0],\
                            topk_p0), dim=1, largest=False)

    node_indices_p0.unsqueeze_(0)
    node_indices_p0 = node_indices_p0.expand(p0.dist_angle_mat_inputs.shape[0],
                                             -1, -1)
    edges_p0_topk = torch.gather(p0.dist_angle_mat_inputs, 2, node_indices_p0)
    #makes the tensor shape "uneven" - so not using
    #edges_p0_topk = filter_dist_threshold(edges_p0_topk,
    #                                      threshold=distance_threshold)

    return edges_p0_topk, node_indices_p0[0, :, :]


def get_node_feats_for_prot(prot,
                            n_pe_nodes=16,
                            n_pe_edges=16,
                            masked=True):
            
    edges_p0 = prot.dist_angle_mat
    node_indices_p0 = torch.tensor([i for i in range(0, prot.seq_len)])

    # each node is connected to all nodes
    #[[0, ... , len_prot], [0, ..., len_prot] ... len_prot]
    node_indices_p0 = node_indices_p0.unsqueeze(0).expand(prot.seq_len, -1)

    pe_nodes_p0, _ = \
        get_node_and_edge_positional_embeddings(node_indices_p0,
                                                prot.fragment_indices,
                                                n_pe_nodes,
                                                n_pe_edges,
                                                edges_p0)
    if masked:
        seq_p0 = prot.prim_masked
    else:
        seq_p0 = prot.prim

    nfeats_p0 = seq_p0.float()

    nfeats_p0_pe = pe_nodes_p0.view(-1, n_pe_nodes).float()
    nfeats_p0_pe.requires_grad = False

    return nfeats_p0, nfeats_p0_pe


def select_contact_indices(dist_angle_mat_int, dist_angle_mat_int_asymm,
                           p0_contact_indices, p1_contact_indices):

    dist_angle_mat_int_p0_select = \
            torch.index_select(dist_angle_mat_int,
                               1, p0_contact_indices)
    dist_angle_mat_int_p0_p1_select = \
        torch.index_select(dist_angle_mat_int_p0_select,
                            2, p1_contact_indices)

    dist_angle_mat_int_p0_asym_select = \
        torch.index_select(dist_angle_mat_int_asymm,
                            2, p0_contact_indices)
    dist_angle_mat_int_p0_p1_asym_select = \
        torch.index_select(dist_angle_mat_int_p0_asym_select,
                            1, p1_contact_indices)

    return dist_angle_mat_int_p0_p1_select,\
        dist_angle_mat_int_p0_p1_asym_select


def get_positional_embeddings(E_idx,
                              N_idx,
                              num_embeddings=16,
                              max_length=10000):
    ii = N_idx.view((1, -1, 1))
    d = (E_idx.float() - ii).unsqueeze(-1)
    # Original Transformer frequencies
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32) *
        -(np.log(max_length) / num_embeddings))
    angles = d * frequency.view((1, 1, -1))
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E.squeeze_(0)


def get_positional_embeddings_nodes(N_idx,
                                    num_embeddings=16,
                                    max_length=10000):
    ii = (N_idx - N_idx[0]).view((1, -1, 1))  #start from zero
    d = (ii).unsqueeze(-1)
    # Original Transformer frequencies
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32) *
        -(np.log(max_length) / num_embeddings))
    angles = d * frequency.view((1, 1, -1))
    NE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return NE.squeeze_(0).squeeze_(1)


def filter_dist_threshold(edges_in, threshold=12.0):
    dist = edges_in[0, :, :]
    d = edges_in[0, :, :][dist <= threshold].view(1, -1)
    o = edges_in[1, :, :][dist <= threshold].view(1, -1)
    t = edges_in[2, :, :][dist <= threshold].view(1, -1)
    p = edges_in[3, :, :][dist <= threshold].view(1, -1)
    return torch.cat([d, o, t, p], dim=0)


def get_edge_features(edges_in, d_min=0.0, d_max=16.0, d_count=10,\
                    real_dist=False, threshold=12.0,\
                    include_orientations=False,
                    inverted_dist=True):
    dist = edges_in[0, :]
    distances = dist[dist <= threshold]

    if not real_dist:
        d_mu = torch.linspace(d_min, d_max, d_count)
        d_sigma = (d_max - d_min) / d_count
        d_mu = d_mu.view(1, -1)
        distances.unsqueeze_(1)
        d = torch.exp(-((distances - d_mu) / d_sigma)**2)
    else:
        if inverted_dist:
            d_inv = torch.Tensor(1.0 / (1.0 + np.exp(0.5 * distances - 6.5)))
            d = d_inv.unsqueeze_(1).expand(-1, 1)
        else:
            d = distances.unsqueeze_(1).expand(-1, 1)
    if include_orientations:
        o = edges_in[1, :][dist <= threshold]
        t = edges_in[2, :][dist <= threshold]
        p = edges_in[3, :][dist <= threshold]
        o = o.unsqueeze_(1).expand(-1, 1)
        t = t.unsqueeze_(1).expand(-1, 1)
        p = p.unsqueeze_(1).expand(-1, 1)
        return torch.cat([d, o, t, p], dim=1)
    else:
        return d


def gather_nn_data(node_indices):
    s1, s2, s3 = node_indices.shape[1], node_indices.shape[2],\
                    node_indices.shape[0]

    #convert neighbor node data into edge data
    node_indices = node_indices.reshape(s1 * s2, s3)
    #node_indices = node_indices.expand(-1,20)
    #This operation moved to forward
    #x = torch.gather(node_data,0,node_indices)
    return node_indices


def plot_contact_data(contact_dist, contact_loop_nodes,\
                      contact_paratope, contact_epitope,\
                      name='tmp', d=12, save=True):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12 * 4, 12))
    ax = plt.subplot(1, 4, 1)
    im = ax.imshow(contact_dist, vmin=2, vmax=25)
    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax = plt.subplot(1, 4, 2)
    im = ax.imshow(contact_loop_nodes)
    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax = plt.subplot(1, 4, 3)
    im = ax.imshow(contact_paratope.unsqueeze_(1).expand(-1, 1))
    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax.axes.get_xaxis().set_visible(False)
    ax = plt.subplot(1, 4, 4)
    im = ax.imshow(contact_epitope.unsqueeze_(0).expand(1, -1))
    ax.axes.get_yaxis().set_visible(False)
    fig.colorbar(im, ax=ax, orientation='horizontal')
    #plt.tight_layout()
    if save:
        name = str(name)
        cl_name = name.replace("(_'", '')
        print(cl_name)
        plt.savefig('.{}_{}.png'.format(cl_name, d), transparent=True)
    plt.show()
