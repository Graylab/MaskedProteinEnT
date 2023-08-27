import torch
import numpy as np
import pandas as pd
import math


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

