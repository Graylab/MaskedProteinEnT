import numpy as np
import torch
from src.data.preprocess.parsing_utils import get_cdr_indices,\
                                            get_chain_seqs_from_pdb

from src.data.utils.pdb import protein_dist_coords_matrix

from src.data.preprocess.ppi_parser import get_dist_angle_mat, get_partner_masks


ag_chains_ids = ['X','Y','Z','Q','R','S','T']
ag_chains_ids_mod = [':'+t for t in ag_chains_ids]

def get_ab_ag_lengths(chain_seqs):
    ag_len = 0
    ab_len = 0
    for _ in chain_seqs:
        if _ in ['H','L']:
            ab_len += len(chain_seqs[_])
        else:
            ag_len += len(chain_seqs[_])
    return ab_len, ag_len

def get_ab_ag_lengths_from_info(info):
    ag_len = 0
    ab_len = 0
    for _ in info:
        if _ in ['H','L']:
            ab_len += len(info[_])
        elif _ in info['antigen_chains']:
            ag_len += len(info[_])
    return ab_len, ag_len



def _get_contact_residues(dist_mat_np,ab_len,distance_cutoff,\
                            loops_only_interface = False, cdr_indices = None,\
                            flanks = 2,
                            cdr_names=['h1','h2','h3','l1','l2','l3']):
    indices = np.argwhere((dist_mat_np <= distance_cutoff) & (dist_mat_np > 0))
    if loops_only_interface:
        print(list(cdr_indices.keys()))
        cdrs = list(cdr_indices.keys()) #[t for t in cdr_names if len(cdr_indices)>0]
        clean_dict = {}
        for k in cdrs:
            val_list = cdr_indices[k]
            if len(val_list)==1:
                clean_dict[k] = [val_list[0],val_list[0]]
            else:
                clean_dict[k] = val_list
        cdr_indices_list = [u for k in cdrs \
            for u in range(clean_dict[k][0]-flanks,clean_dict[k][-1]+flanks+1)]
        indices_ag = [indices[t,0] for t in range(indices.shape[0]) \
                    if (indices[t,0] >= ab_len) and (indices[t,1] in cdr_indices_list)]
    else:
        indices_ag = [indices[t,0] for t in range(indices.shape[0]) \
                    if (indices[t,0] >= ab_len)]
    uniq_indices_ag = list(set(indices_ag))
    uniq_indices_ag.sort()
    return uniq_indices_ag


def get_chain_lengths(seq, chains=['']):
    lengths = {}
    for key in chains:
        lengths[key] = 0
    for chain_str in chains:
        for letter in chain_str:
            lengths[chain_str] += len(seq[letter])
    return lengths


def _get_ppi_contact_residues(dist_mat,
                              chain_seqs,
                              partners,
                              distance_cutoff=10.0):

    # select residue indices that form "epitope"
    #ppi - both sides are epitopes
    p0, p1 = partners[0], partners[1]
    lengths = get_chain_lengths(chain_seqs, partners)
    assert(len(lengths.keys())==2)
    mask_non_int = get_partner_masks(lengths[p0],lengths[p1],\
                    s_1=0, s_2=1, s_3=1, s_4=0)
    
    dist_mat_int = np.ones((dist_mat.shape)) * 1000 #large distance
    #print(dist_mat_int.shape, mask_non_int.shape)
    dist_mat_int[mask_non_int == 1] = dist_mat[mask_non_int == 1]
    dist_mat_np = dist_mat_int[:, :]
    
    indices = np.argwhere((dist_mat_np <= distance_cutoff) \
                         & (dist_mat_np > 0))
    
    uniq_epi_indices = {}
    indices_cutoff_p = [indices[t,0] \
                        for t in range(indices.shape[0]) \
                            if 0 < indices[t,0] < lengths[partners[0]]]
    uniq_epi_indices[partners[0]] = list(set(indices_cutoff_p))
    uniq_epi_indices[partners[0]].sort()

    indices_cutoff_p = [indices[t,0] \
                        for t in range(indices.shape[0]) \
                            if (lengths[partners[0]] \
                                 <= indices[t,0]) and \
                            (indices[t,0] < lengths[partners[0]] + lengths[partners[1]]) ]

    uniq_epi_indices[partners[1]] = list(set(indices_cutoff_p))
    uniq_epi_indices[partners[1]].sort()
    
    return uniq_epi_indices


def _get_contact_pairs(dist_mat_np,ab_len,distance_cutoff,\
                            loops_only_interface = False, cdr_indices = None,\
                            flanks = 2,
                            cdr_names=['h1','h2','h3','l1','l2','l3']):
    indices = np.argwhere((dist_mat_np <= distance_cutoff) & (dist_mat_np > 0))
    if loops_only_interface:
        cdrs = [t for t in cdr_names if len(cdr_indices[t])>0]
        clean_dict = {}
        for k in cdrs:
            val_list = cdr_indices[k]
            if len(val_list)==1:
                clean_dict[k] = [val_list[0],val_list[0]]
            else:
                clean_dict[k] = val_list
        cdr_indices_list = [u for k in cdrs \
            for u in range(clean_dict[k][0]-flanks,clean_dict[k][-1]+flanks+1)]
        indices_ag = [indices[t,0] for t in range(indices.shape[0]) \
                    if (indices[t,0] >= ab_len) and (indices[t,1] in cdr_indices_list)]
    else:
        indices_ag = [indices[t,0] for t in range(indices.shape[0]) \
                    if (indices[t,0] >= ab_len)]
    uniq_indices_ag = list(set(indices_ag))
    uniq_indices_ag.sort()
    return uniq_indices_ag


def _get_epitope_neighbors(chain_seqs,dist_mat, indices_epitope,distance_cutoff_epitope):

    ab_len, ag_len = get_ab_ag_lengths(chain_seqs)
    mask_non_ag = get_partner_masks(ab_len,ag_len,s_1=0,s_2=0,s_3=0,s_4=1)

    dist_mat_ag = np.ones((dist_mat.shape)) * 1000 #large distance
    dist_mat_ag[mask_non_ag == 1] = dist_mat[mask_non_ag == 1]
    dist_mat = dist_mat_ag[:,:]
    
    dist_mat_epi = np.take(dist_mat,indices_epitope,0)
    indices_epi_nn = np.argwhere((dist_mat_epi <= distance_cutoff_epitope) & (dist_mat_epi > 0))
    uniq_indices_nn = list(set(list(indices_epi_nn[:,1])))
    uniq_indices_nn.sort()
    return uniq_indices_nn


def filter_antigen_noninterface(chain_seqs,dist_mat,
                                distance_cutoff=14, loops_only_interface=False,\
                                cdr_indices=None, include_epitope_neighbors=False,\
                                distance_cutoff_epitope=12.0, max_len=200,
                                flanks=2,
                                make_contiguous=False,
                                lower_bound_contact_distance=8.0):
    '''
    remove antigen residues not at the interface
    lower_bound_contact_distance: There should be atleast 2 residues within
    this distance
    '''
    chain_lengths = {}
    for key in chain_seqs:
        if not key in ['H','L']:
            chain_lengths[key] = len(chain_seqs[key])
    ab_len, ag_len = get_ab_ag_lengths(chain_seqs)

    mask_non_int = get_partner_masks(ab_len,ag_len,s_1=0,s_2=1,s_3=1,s_4=0)

    dist_mat_int = np.ones((dist_mat.shape)) * 1000 #large distance
    dist_mat_int[mask_non_int == 1] = dist_mat[mask_non_int == 1]
    dist_mat_np = dist_mat_int[:,:]
    
    uniq_indices_ag = _get_contact_residues(dist_mat_np,ab_len,distance_cutoff,\
                                            loops_only_interface = loops_only_interface,\
                                            cdr_indices = cdr_indices,flanks=flanks)

    uniq_indices_ag_lb = _get_contact_residues(dist_mat_np,ab_len,lower_bound_contact_distance,\
                                            loops_only_interface = loops_only_interface,\
                                            cdr_indices = cdr_indices,flanks=flanks)
    if len(uniq_indices_ag_lb) < 2:
        return None
    print(uniq_indices_ag)
    
    ag_len = len(uniq_indices_ag)
    #reduce the cutoff if more than 200 ag-contact residues
    if ag_len>max_len:
        print('contact ag res > {}'.format(max_len))
        uniq_indices_ag = _get_contact_residues(dist_mat_np,ag_len,distance_cutoff-1.0,
                                                flanks=flanks)
        contg_ag_range = get_contiguous_fragments(uniq_indices_ag,ab_len,
                                                   chain_lengths,ag_chains_ids, 
                                                   b_insert=False)
    else:    
        contg_ag_range = get_contiguous_fragments(uniq_indices_ag,ab_len,
                                                   chain_lengths,ag_chains_ids)
            
    if len(contg_ag_range)<1:
        return None
    contg_ag_range_flat = [u for t in contg_ag_range for u in range(t[0],t[1]+1)]
    contg_ag_minus_ab = [(t[0]-ab_len,t[1]-ab_len) for t in contg_ag_range]
    #print(len(contg_ag_range_flat))
    contg_ag_nn_minus_ab = []
    
    if include_epitope_neighbors:
        
        uniq_indices_nn = _get_epitope_neighbors(chain_seqs,dist_mat,uniq_indices_ag,\
            distance_cutoff_epitope=distance_cutoff_epitope)
        contg_ag_nn_range_flat = list(set(uniq_indices_nn + contg_ag_range_flat))
        
        #print(len(contg_ag_nn_range_flat))
        if len(contg_ag_nn_range_flat) > max_len:
            uniq_indices_nn = _get_epitope_neighbors(chain_seqs,dist_mat,uniq_indices_ag,\
                distance_cutoff_epitope=max(4.0,distance_cutoff_epitope-4.0))
            contg_ag_nn_range_flat = list(set(uniq_indices_nn + contg_ag_range_flat))
        
        #print(len(contg_ag_nn_range_flat))
        contg_ag_nn_range = get_contiguous_fragments(contg_ag_nn_range_flat,ab_len,
                                                      chain_lengths, ag_chains_ids)
        contg_ag_range_flat = [u for t in contg_ag_nn_range for u in range(t[0],t[1]+1)]
        contg_ag_nn_minus_ab = [(t[0]-ab_len,t[1]-ab_len) for t in contg_ag_nn_range]
        #print('epi: {}\nepi+nn: {}'.format(contg_ag_minus_ab,contg_ag_nn_minus_ab))
        
    keep_indices = torch.tensor([t for t in range(0,ab_len)] + [t for t in contg_ag_range_flat])
    #print('keep ',keep_indices)
    
    return (keep_indices,
            contg_ag_minus_ab,
            contg_ag_nn_minus_ab)


def get_antigen_interface(chain_seqs,dist_mat,
                        cdr_indices,
                        distance_cutoff=12.0,\
                        loops_only_interface=True,\
                        include_epitope_neighbors=False,\
                        distance_cutoff_epitope=12.0,
                        flanks=2):
    filtered_contacts = \
        filter_antigen_noninterface(chain_seqs,dist_mat,
                                    distance_cutoff=distance_cutoff,
                                    loops_only_interface=loops_only_interface,\
                                    cdr_indices = cdr_indices,\
                                    include_epitope_neighbors=include_epitope_neighbors,\
                                    distance_cutoff_epitope=distance_cutoff_epitope,
                                    flanks=flanks)
    if filtered_contacts is None:
        return None
    
    keep_indices = filtered_contacts[0]
    fragment_ag_indices = filtered_contacts[1]
    fragment_ag_nn_indices = filtered_contacts[2]

    contact_frags, all_frags, _ = \
        get_epitope_and_neighbor_fragments(ag_chains_ids,
                                           chain_seqs,
                                           fragment_ag_indices,
                                           fragment_ag_nn_indices)
    
    return keep_indices, contact_frags, all_frags


def get_antigen_contact_fragments(pdb_file,distance_cutoff=12.0,\
                                    loops_only_interface=True,\
                                    antigen_info=True,\
                                    include_epitope_neighbors=False,\
                                    distance_cutoff_epitope=12.0,
                                    get_rosetta_contact_pairs=True,
                                    flanks=2):
    
    chain_seqs = get_chain_seqs_from_pdb(pdb_file,\
                                         ag_chains_ids_mod,
                                         antigen_info=antigen_info)
    antigen_chains = []
    for chain_id in ag_chains_ids:
        if chain_id in chain_seqs:
            antigen_chains.append(chain_id)
    
    try:
        protein_geometry_mats = protein_dist_coords_matrix(pdb_file)
    except:
        return None
    
    dist_mat = protein_geometry_mats[0]
    
    cdr_indices = get_cdr_indices(pdb_file)
    print(cdr_indices)
    _, contact_frags, _ =  get_antigen_interface(chain_seqs,dist_mat,
                                    cdr_indices,distance_cutoff=distance_cutoff,\
                                    loops_only_interface=loops_only_interface,\
                                    include_epitope_neighbors=include_epitope_neighbors,\
                                    distance_cutoff_epitope=distance_cutoff_epitope,
                                    flanks=flanks)
    return contact_frags


def add_protein_info(info,pdb_file,chain_seqs):
    protein_chains=[]
    #always loop over chain_ids to maintain the sequence in which chains are
    for chain_id in chain_seqs:
        protein_chains.append(chain_id)
    info.update(dict(protein_chains=protein_chains))

    #try:
    protein_geometry_mats = protein_dist_coords_matrix(pdb_file)
    #except:
    #    return None
    
    dist_mat = protein_geometry_mats[0]
    coords = protein_geometry_mats[1]
    
    info.update(dict(bk_and_cb_coords=coords))
    
    info.update(
    dict(dist_mat=dist_mat.unsqueeze(0)))

    return info

def add_antigen_info_full(info,pdb_file,chain_seqs,
                          cdr_indices,
                          distance_cutoff=12.0,\
                          loops_only_interface=True,\
                          include_epitope_neighbors=False,\
                          distance_cutoff_epitope=12.0,
                          write_trunc_pdbs=False,
                        ):
    antigen_chains=[]
    #always loop over ag_chain_ids to maintain the sequence in which chains are
    for chain_id in ag_chains_ids:
        if chain_id in chain_seqs:
            antigen_chains.append(chain_id)
    info.update(dict(antigen_chains=antigen_chains))

    print(cdr_indices)

    try:
        protein_geometry_mats = protein_dist_coords_matrix(pdb_file)
    except:
        return None
    
    dist_mat = protein_geometry_mats[0]
    coords = protein_geometry_mats[1]
    
    info.update(dict(bk_and_cb_coords=coords))
    
    info.update(
    dict(dist_mat=dist_mat.unsqueeze(0)))

    data_antigen_interface = \
                    get_antigen_interface(chain_seqs,dist_mat,
                                         cdr_indices,
                                         distance_cutoff=distance_cutoff,\
                                         loops_only_interface=loops_only_interface,\
                                         include_epitope_neighbors=include_epitope_neighbors,\
                                         distance_cutoff_epitope=distance_cutoff_epitope
                                         )
    if data_antigen_interface is None:
        print('No antigen interacting residues found')
        return None
    else:
        ag_contact_frags = data_antigen_interface[1]

    
    info.update(dict(chain_contact_indices=ag_contact_frags))

    ab_len, ag_len = get_ab_ag_lengths(chain_seqs)
    ag_len = dist_angle_mat.shape[1] - ab_len #ag seq may have been trimmed
    
    pdb_residue_ids_dict = get_residue_numbering_for_pdb(pdb_file)
    
    frag_indices = []
    pdb_indices = []
    chain_fragments ={}
    prev_len = 0
    for chain_num, chain in enumerate(ag_chain_ids):
        frag_indices += [i+prev_len for i in range(len(chain_seqs[chain]))]
        offset = 100 if chain_num > 0 else 0
        pdb_indices += [t+offset for t in pdb_residue_ids_dict[chain]]
        prev_len += len(chain_seqs[chain])
        chain_fragments[chain] = [i for i in range(len(chain_seqs[chain]))]

     
    info.update(dict(chain_fragments=chain_fragments))

    rel_contact_indices = [frag_indices.index(ind)
                                    for ind in uniq_epi_indices_contg_flat]
    frag_indices_pdb = [pdb_indices[t] for t in frag_indices]

    info.update(dict(frag_indices=frag_indices_pdb))
    info.update(dict(antigen_prim=antigen_prim))

    return info

