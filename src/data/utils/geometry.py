import math
import random
import torch


def calc_dist_mat(a_coords, b_coords):
    assert a_coords.shape == b_coords.shape
    mat_shape = (len(a_coords), len(a_coords), 3)

    a_coords = a_coords.unsqueeze(0).expand(mat_shape)
    b_coords = b_coords.unsqueeze(1).expand(mat_shape)

    dist_mat = (a_coords - b_coords).norm(dim=-1)

    return dist_mat


def calc_dihedral(a_coords,
                  b_coords,
                  c_coords,
                  d_coords,
                  convert_to_degree=False):
    b1 = a_coords - b_coords
    b2 = b_coords - c_coords
    b3 = c_coords - d_coords

    n1 = torch.cross(b1, b2)
    n1 = torch.div(n1, n1.norm(dim=-1, keepdim=True))
    n2 = torch.cross(b2, b3)
    n2 = torch.div(n2, n2.norm(dim=-1, keepdim=True))
    m1 = torch.cross(n1, torch.div(b2, b2.norm(dim=-1, keepdim=True)))

    dihedral = torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    if convert_to_degree:
        dihedral = dihedral * 180 / math.pi

    return dihedral

def get_masked_mat(input_mat, mask, mask_fill_value=-999, device=None):
    out_mat = torch.ones(input_mat.shape).type_as(input_mat)
    if device is not None:
        mask = mask.to(device)
        out_mat = out_mat.to(device)

    out_mat[mask == 0] = mask_fill_value
    out_mat[mask == 1] = input_mat[mask == 1]

    return out_mat
