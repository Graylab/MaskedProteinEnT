import torch
import numpy as np
from einops import rearrange, repeat
from matplotlib import pyplot as plt
import numpy 
import math

def get_num_bkbone_atoms(num_atoms):
    bkb_atoms = 3
    if num_atoms == 14:
        bkb_atoms = 4
    return bkb_atoms


def mask_nfeats_cb(masked_seq_label, nfeats, nfeats_atoms):
    #print(masked_seq_label.shape, nfeats.shape, nfeats_atoms.shape)
    num_atoms = nfeats_atoms.shape[1]
    bkb_atoms = get_num_bkbone_atoms(num_atoms)
    nfeats_atom_inputs = torch.ones((nfeats_atoms.shape[0], bkb_atoms))
    nfeats_atom_masked = torch.ones((nfeats_atoms.shape[0], num_atoms - bkb_atoms))
    sequence_label_exp = torch.tensor(masked_seq_label).clone().detach()
    sequence_label_exp[sequence_label_exp == 20] = -1 # 
    sequence_label_exp[sequence_label_exp != -1] = 1 # mask these sidechain feats
    sequence_label_atoms = \
        torch.tensor(sequence_label_exp).unsqueeze(1).expand(-1, num_atoms).reshape(masked_seq_label.shape[0]*num_atoms)
    sequence_label_exp_2 = sequence_label_atoms.unsqueeze(1).expand(-1, nfeats_atom_masked.shape[1])
    nfeats_atom_masked[sequence_label_exp_2 == 1] = 0
    nfeats_atom_masked = torch.cat([nfeats_atom_inputs, nfeats_atom_masked], dim=1)
    nfeats_atoms[nfeats_atom_masked == 0] = 0
    nfeats = torch.cat([nfeats, nfeats_atoms], dim=1)
    return nfeats
    

def mask_nfeats_aa_and_cb(masked_seq_label, nfeats_res, nfeats_atoms):
    #print(masked_seq_label.shape, nfeats.shape, nfeats_atoms.shape)
    num_atoms = nfeats_atoms.shape[1]
    bkb_atoms = get_num_bkbone_atoms(num_atoms)
    nfeats_atom_inputs = torch.ones((nfeats_atoms.shape[0], bkb_atoms))
    nfeats_atom_masked = torch.ones((nfeats_atoms.shape[0], num_atoms - bkb_atoms))
    sequence_label_exp = torch.tensor(masked_seq_label).clone().detach()
    sequence_label_exp[sequence_label_exp == 20] = -1 # 
    sequence_label_exp[sequence_label_exp != -1] = 1 # mask these sidechain feats
    sequence_label_atoms = \
        torch.tensor(sequence_label_exp).unsqueeze(1).expand(-1, num_atoms).reshape(masked_seq_label.shape[0]*num_atoms)
    sequence_label_exp_2 = sequence_label_atoms.unsqueeze(1).expand(-1, nfeats_atom_masked.shape[1])
    nfeats_atom_masked[sequence_label_exp_2 == 1] = 0
    nfeats_atom_masked = torch.cat([nfeats_atom_inputs, nfeats_atom_masked], dim=1)
    nfeats_atoms[nfeats_atom_masked == 0] = 0

    #AA
    num_aa = nfeats_res.shape[-1]
    sequence_label_exp_3 = sequence_label_atoms.unsqueeze(1).expand(-1, num_aa)
    #print(nfeats_res.shape, masked_seq_label.shape, sequence_label_exp_3.shape)
    #mask these residue aa identities
    nfeats_res[sequence_label_exp_3==1] = 0
    #import matplotlib.pyplot as plt
    #plt.imshow(nfeats_res.reshape(-1, num_atoms, num_aa)[:, 0, :].cpu().numpy())
    #plt.colorbar()
    #plt.savefig('./nfeats_res_masked.png')
    #plt.close()
    #combine atoms and res
    nfeats = torch.cat([nfeats_res, nfeats_atoms], dim=1)
    return nfeats

