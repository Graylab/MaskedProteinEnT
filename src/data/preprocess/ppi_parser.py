import logging
import numpy as np
import torch


def get_chain_lengths(seq, chains=['']):
    lengths = {}
    for key in chains:
        lengths[key] = 0
    for chain_str in chains:
        for letter in chain_str:
            lengths[chain_str] += len(seq[letter])
    return lengths


def get_partner_masks(l_1, l_2, s_1=0, s_2=0, s_3=0, s_4=0):

    
    if s_1 == 1:
        m_1 = np.ones((l_1, l_1))
    else:
        m_1 = np.zeros((l_1, l_1))

    if s_2 == 1:
        m_2 = np.ones((l_1, l_2))
    else:
        m_2 = np.zeros((l_1, l_2))

    if s_3 == 1:
        m_3 = np.ones((l_2, l_1))
    else:
        m_3 = np.zeros((l_2, l_1))

    if s_4 == 1:
        m_4 = np.ones((l_2, l_2))
    else:
        m_4 = np.zeros((l_2, l_2))

    temp_1 = np.concatenate((m_1, m_2), axis=1)
    temp_2 = np.concatenate((m_3, m_4), axis=1)
    mask = np.expand_dims(np.concatenate((temp_1, temp_2),
                                          axis=0), 0)
    return mask.squeeze(0)

def get_masked_matrices_for_partners(dist_angle_mat, p0_len, p1_len,
                                     mask_fill_value=-999):
    f = dist_angle_mat.shape[0]
    #print(ab_len,ag_len,dist_angle_mat.shape)
    mask_ag_all = get_partner_masks(p0_len, p1_len, s_1=1, s_2=0, s_3=0, s_4=0, f=f)
    dist_angle_mat_ab = torch.ones((dist_angle_mat.shape)) * mask_fill_value
    dist_angle_mat_ab[mask_ag_all == 1] = dist_angle_mat[mask_ag_all == 1]
    
    mask_non_int = get_partner_masks(p0_len, p1_len, s_1=0, s_2=1, s_3=1, s_4=0, f=f)
    dist_angle_mat_int = torch.ones((dist_angle_mat.shape)) * mask_fill_value
    dist_angle_mat_int[mask_non_int == 1] = dist_angle_mat[mask_non_int == 1]

    mask_ab_all = get_partner_masks(p0_len, p1_len, s_1=0, s_2=0, s_3=0, s_4=1, f=f)
    dist_angle_mat_ag = torch.ones((dist_angle_mat.shape)) * mask_fill_value
    dist_angle_mat_ag[mask_ab_all == 1] = dist_angle_mat[mask_ab_all == 1]

    assert(not torch.equal(dist_angle_mat_ab, dist_angle_mat_ag))
    assert(not torch.equal(dist_angle_mat_ab, dist_angle_mat_int))

    return dist_angle_mat_ab, dist_angle_mat_int, dist_angle_mat_ag


def _get_neighbors_for_partner(dist_mat, indices_epitope,
                               length_p0, length_p1,
                               distance_cutoff_epitope,
                               s_1=0, s_4=1):
    '''
    Abstraction of _get_epitope_neighbors
    '''
    mask_non_part = get_partner_masks(length_p0, length_p1, 
                                      s_1=s_1, s_2=0, s_3=0, s_4=s_4)
    dist_mat_part = np.ones((dist_mat.shape)) * 1000 #large distance

    dist_mat_part[mask_non_part == 1] = dist_mat[mask_non_part == 1]
    dist_mat = dist_mat_part[:, :]
    dist_mat_epi = np.take(dist_mat, indices_epitope, 0)
    indices_epi_nn = np.argwhere((dist_mat_epi <= distance_cutoff_epitope)\
                                  & (dist_mat_epi > 0))
    uniq_indices_nn = list(set(list(indices_epi_nn[:,1])))
    uniq_indices_nn.sort()
    return uniq_indices_nn


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

def _get_ppi_epitope_neighbors(dist_mat,
                               chain_seqs,
                               uniq_epi_indices,
                               partners,
                               distance_cutoff_epitope=8.0):

    p0, p1 = partners[0], partners[1]
    lengths = get_chain_lengths(chain_seqs, partners)
    
    uniq_p_indices = {}
    uniq_p_indices[p0] = _get_neighbors_for_partner(\
                                        dist_mat, uniq_epi_indices[p0],\
                                        lengths[p0], lengths[p1],
                                        distance_cutoff_epitope,\
                                        s_1=1,s_4=0)
    uniq_p_indices[p0] = list(set(uniq_p_indices[p0] + uniq_epi_indices[p0]))
    uniq_p_indices[p0].sort()

    uniq_p_indices[p1] = _get_neighbors_for_partner(
                                        dist_mat, uniq_epi_indices[p1],\
                                        lengths[p0], lengths[p1],
                                        distance_cutoff_epitope,\
                                        s_1=0,s_4=1)
    uniq_p_indices[p1] = list(set(uniq_p_indices[p1] + uniq_epi_indices[p1]))
    uniq_p_indices[p1].sort()

    return uniq_p_indices


def get_dist_angle_mat(pdb_file, component_chains_partners=None,
                       use_pairwise_geometry=False,
                       get_cb_coords=False,
                       get_bk_and_cb_coords=False,
                       get_fullatom=False):
    if not use_pairwise_geometry:
        dist_angle_mat = \
            protein_dist_angle_matrix(pdb_file,
                                      chains=component_chains_partners)
        dist_mat = dist_angle_mat[0, :, :]
    else:
        dist_angle_mat = \
            protein_pairwise_geometry_matrix(pdb_file,
                                             chains=component_chains_partners)
        dist_mat = dist_angle_mat[1, :, :]
    
    if get_cb_coords:
        coords = protein_coords_matrix(pdb_file, 
                                       chains=component_chains_partners,
                                       b_cb_coords=get_cb_coords)
        return (dist_angle_mat, dist_mat, coords)
    elif get_bk_and_cb_coords:
        coords = protein_coords_matrix(pdb_file,
                                       chains=component_chains_partners,
                                       b_bk_and_cb_coords=get_bk_and_cb_coords)
        return (dist_angle_mat, dist_mat, coords)
    elif get_fullatom:
        coords = protein_full_coords_matrix(pdb_file,
                                       chains=component_chains_partners)
        return (dist_angle_mat, dist_mat, coords)
    else:
        return (dist_angle_mat, dist_mat)


def add_ppi_info(info,pdb_file,chain_seqs,
                 distance_cutoff=12.0,
                 include_epitope_neighbors=True,
                 distance_cutoff_epitope=12.0,
                 write_trunc_pdbs=False,
                 include_cl=False,
                 flanks=2,
                 partners=[],
                 max_length=350,
                 min_interface_dist=False,
                 average_ha_dist=True,
                 use_pairwise_geometry=False,
                 get_cb_coords=False,
                 get_bk_and_cb_coords=False,
                 get_fullatom=False,
                 add_per_res=False,
                 do_not_truncate=False):
    
    logger = logging.getLogger(__name__ + 'add_ppi_info')
    logger.debug('Adding ppi info ...')

    if do_not_truncate:
        return add_ppi_info_full(info,
                                pdb_file,
                                chain_seqs,
                                distance_cutoff=distance_cutoff,
                                partners=partners,
                                use_pairwise_geometry=use_pairwise_geometry,
                                get_cb_coords=get_cb_coords,
                                get_bk_and_cb_coords=get_bk_and_cb_coords)
    
    for pi in partners:
        for ch in pi:
            if ch not in chain_seqs.keys():
                return None

    p0, p1 = partners[0], partners[1]
    component_chains_partners = [t for t in p0] + [t for t in p1]
    component_chain_lengths = get_chain_lengths(chain_seqs,
                                                component_chains_partners)

    try:
        protein_geometry_mats = get_dist_angle_mat(pdb_file,
                                               component_chains_partners,
                                               use_pairwise_geometry=
                                               use_pairwise_geometry,
                                               get_cb_coords=get_cb_coords,
                                               get_bk_and_cb_coords=get_bk_and_cb_coords,
                                               get_fullatom=get_fullatom)
    except:
        return None
    dist_angle_mat = protein_geometry_mats[0]
    dist_mat = protein_geometry_mats[1]
    if get_cb_coords or get_bk_and_cb_coords or get_fullatom:
        coords = protein_geometry_mats[2]
    
    if add_per_res:
        per_res_prop = read_bfactor_value(pdb_file)
    
    phi_psi_mat = protein_bb_phi_psi_matrix(pdb_file,
                                            chains=component_chains_partners)
    
    uniq_epi_indices = _get_ppi_contact_residues(dist_mat,
                                                 chain_seqs,
                                                 partners,
                                                 distance_cutoff=distance_cutoff)

    logger.debug('epi indices {}: {}'.format(p0, uniq_epi_indices[p0]))
    logger.debug('epi indices {}: {}'.format(p1, uniq_epi_indices[p1]))
    for pi in partners:
        if len(uniq_epi_indices[pi]) == 0:
            return None
    #Convert contacts to contiguous ranges
    # [1,2,3,7,8,9,11] -> [(1,3), (7-11)] etc.
    lengths = get_chain_lengths(chain_seqs, partners)
    
    logger.debug('chain lengths: {}\n'.format(lengths))
    uniq_epi_indices_contg_range = {}
    uniq_epi_indices_contg_range[p0] = get_contiguous_fragments(uniq_epi_indices[p0],
                                                           0,
                                                           component_chain_lengths,
                                                           [t for t in p0])
    
    uniq_epi_indices_contg_range[p1] = get_contiguous_fragments(uniq_epi_indices[p1],
                                                           lengths[p0],
                                                           component_chain_lengths,
                                                           [t for t in p1])

    logger.debug('contg {} epi {}'.format(p0, uniq_epi_indices_contg_range[p0]))
    logger.debug('contg {} epi {}\n'.format(p1, uniq_epi_indices_contg_range[p1]))
    
    uniq_epi_indices_contg_flat = {}
    uniq_epi_indices_contg_flat[p0] = [t for u in uniq_epi_indices_contg_range[p0]\
                                         for t in range(u[0], u[1]+1)]
    uniq_epi_indices_contg_flat[p1] = [t for u in uniq_epi_indices_contg_range[p1]\
                                         for t in range(u[0], u[1]+1)]

    #Add Neighbor "context" for each partner
    uniq_p_indices = _get_ppi_epitope_neighbors(dist_mat,
                                                chain_seqs,
                                                uniq_epi_indices_contg_flat,
                                                partners,
                                                distance_cutoff_epitope=\
                                                    distance_cutoff_epitope)
    logger.debug('nn {}: {}'.format(p0, uniq_p_indices[p0]))                                                    
    logger.debug('nn {}: {}\n'.format(p1, uniq_p_indices[p1]))
    #Make contiguous
    uniq_p_indices_contg_range = {}
    uniq_p_indices_contg_range[p0] = get_contiguous_fragments(uniq_p_indices[p0],
                                                                0,
                                                                component_chain_lengths,
                                                                [t for t in p0])
    
    uniq_p_indices_contg_range[p1] = get_contiguous_fragments(uniq_p_indices[p1],
                                                               lengths[p0],
                                                               component_chain_lengths,
                                                               [t for t in p1])
    
    logger.debug('contg nn {}: {}'.format(p0, uniq_p_indices_contg_range[p0]))
    logger.debug('contg nn {}: {}\n'.format(p1, uniq_p_indices_contg_range[p1]))

    uniq_p_indices_contg_flat_p0 = [t for u in uniq_p_indices_contg_range[p0]\
                                         for t in range(u[0], u[1]+1)]
    uniq_p_indices_contg_flat_p1 = [t for u in uniq_p_indices_contg_range[p1]\
                                         for t in range(u[0], u[1]+1)]

    # Filter distance,angle matrices to keep only contact+context frags
    keep_indices = torch.tensor([t for t in uniq_p_indices_contg_flat_p0] \
                        + [t for t in uniq_p_indices_contg_flat_p1])

    if not use_pairwise_geometry:
        dist_angle_mat_f1 = torch.index_select(dist_angle_mat,1,keep_indices)
        dist_angle_mat_filtered = torch.index_select(dist_angle_mat_f1,2,keep_indices)
    else:
        dist_angle_mat_f1 = torch.index_select(dist_angle_mat,1,keep_indices)
        dist_angle_mat_f1 = torch.index_select(dist_angle_mat_f1,2,keep_indices)

        #original d-d mat
        dist_angle_mat_filtered = torch.index_select(dist_angle_mat_f1, 0, torch.tensor([1,3,4,5]))
        
        #ca, no
        dist_add_mat_filtered = torch.index_select(dist_angle_mat_f1, 0, torch.tensor([0,2]))
        info.update(
        dict(dist_add_mat=dist_add_mat_filtered.contiguous()))

    phi_psi_mat_filtered = torch.index_select(phi_psi_mat,1,keep_indices)
    info.update(
    dict(dist_angle_mat=dist_angle_mat_filtered.contiguous(),
        phi_psi_mat=phi_psi_mat_filtered))

    if min_interface_dist:
        dist_min_ha_mat = protein_all_heavy_min_dist_matrix(pdb_file, average=average_ha_dist)
        dist_min_ha_mat_f1 = torch.index_select(dist_min_ha_mat,0,keep_indices)
        dist_min_ha_mat_cutoff = torch.index_select(dist_min_ha_mat_f1,1,keep_indices)
        info.update(dict(dist_min_ha_mat=dist_min_ha_mat_cutoff))

    if get_cb_coords:
        cb_coords_filtered = torch.index_select(coords, 0, keep_indices)
        info.update(dict(cb_coords=cb_coords_filtered))

    if get_bk_and_cb_coords:
        bk_and_cb_coords_filtered = torch.index_select(coords, 1, keep_indices)
        info.update(dict(bk_and_cb_coords=bk_and_cb_coords_filtered))

    if get_fullatom:
        coords_filtered = torch.index_select(coords, 1, keep_indices)
        info.update(dict(all_coords=coords_filtered))
    
    if add_per_res:
        per_res_prop_filtered = torch.index_select(per_res_prop, 0, keep_indices)
        info.update(dict(per_res_prop=per_res_prop_filtered))
    
    #Get final contact and contact+context fragment ranges for each partner; chainwise.
    # first remove partner 0's length from p1's indices
    uniq_epi_indices_range_p1 = [(t[0]-lengths[p0], t[1]-lengths[p0]) \
                                            for t in uniq_epi_indices_contg_range[p1]]
    uniq_epi_indices_contg_range[p1] = uniq_epi_indices_range_p1
    
    uniq_p_indices_contg_range_p1 = [(t[0]-lengths[p0], t[1]-lengths[p0]) \
                                            for t in uniq_p_indices_contg_range[p1]]
    uniq_p_indices_contg_range[p1] = uniq_p_indices_contg_range_p1
    
    contact_frags, all_frags, rel_contact_indices = \
        {}, {}, {}
    
    total_contact_length = 0
    for pi in partners:
        contact_frags[pi], all_frags[pi], contact_length = \
                                            get_epitope_and_neighbor_fragments([t for t in pi],
                                                                    chain_seqs,
                                                                    uniq_epi_indices_contg_range[pi],
                                                                    uniq_p_indices_contg_range[pi])
        if contact_length > 350:
            return None
        total_contact_length += contact_length
        rel_contact_indices[pi] = get_relative_positions(all_frags[pi], contact_frags[pi])
        delete_chains = []
        for ch in all_frags[pi]:
            if len(all_frags[pi][ch]) == 0:
                delete_chains.append(ch)
        #print(pi, delete_chains)
        for ch in delete_chains:
            del all_frags[pi][ch]
        
    #print(all_frags)
    
    if total_contact_length > 350:
        return None

    # Get sequence of fragments and contact fragment indices for positional encoding
    pi_prim, frag_indices, pi_chain_begs, pi_c_terms, orig_frags, orig_contact_frags = \
         {}, {}, {}, {}, {}, {}
    
    for pi in partners:
        frag_indices[pi], orig_frags[pi], orig_contact_frags[pi] = \
            map_indices_to_pdbs(pdb_file, all_frags[pi], contact_frags[pi])
        
        pi_prim[pi], pi_chain_begs[pi], pi_c_terms[pi] = \
                             get_information_from_fragments(all_frags[pi],
                                                            [t for t in pi],
                                                            chain_seqs)
        #print(frag_indices[pi], orig_frags[pi])
    logger.debug(frag_indices)
    
    if write_trunc_pdbs:
        combined_contact_frags = {}
        combined_contact_frags.update(orig_contact_frags[p0])
        combined_contact_frags.update(orig_contact_frags[p1])
        
        write_pdb_for_fragments(pdb_file, combined_contact_frags,
                                component_chains_partners,
                                get_id(pdb_file),
                                distance_cutoff=distance_cutoff,suffix='_ppi_interface_epi',\
                                prefix='data/ppi_pdbs_int_trunc')
        combined_all_frags = {}
        combined_all_frags.update(orig_frags[p0])
        combined_all_frags.update(orig_frags[p1])

        write_pdb_for_fragments(pdb_file, combined_all_frags,
                                component_chains_partners,
                                get_id(pdb_file),
                                distance_cutoff=distance_cutoff,\
                                suffix='_loop_interface_epi_nn{}'.format(distance_cutoff_epitope),\
                                prefix='data/ppi_pdbs_int_trunc')
    #Update info dictionary with fragment information
    #print('frag p0', frag_indices[p0], len(frag_indices[p0]))
    #print('cont p0', rel_contact_indices[p0])
    #print('frag p1', frag_indices[p1], len(frag_indices[p1]))
    #print('cont p1', rel_contact_indices[p1])
    info.update(dict(chain_fragments_p0=all_frags[p0]))
    info.update(dict(chain_fragments_p1=all_frags[p1]))
    info.update(dict(chain_contact_indices_p0=rel_contact_indices[p0]))
    info.update(dict(chain_contact_indices_p1=rel_contact_indices[p1]))
    info.update(dict(p0_prim=pi_prim[p0]))
    info.update(dict(p1_prim=pi_prim[p1]))
    info.update(dict(frag_indices_p0=frag_indices[p0]))
    info.update(dict(frag_indices_p1=frag_indices[p1]))
    info.update(dict(p0_chain_begs=pi_chain_begs[p0],
                    p1_chain_begs=pi_chain_begs[p1],
                    p0_c_terms=pi_c_terms[p0],
                    p1_c_terms=pi_c_terms[p1]))

    return info

def add_ppi_info_full(info,pdb_file,chain_seqs,
                        distance_cutoff=12.0,
                        partners=[],
                        use_pairwise_geometry=False,
                        get_cb_coords=False,
                        get_bk_and_cb_coords=False):
    
    logger = logging.getLogger(__name__ + 'add_ppi_info')
    logger.debug('Adding ppi info ...')
    
    for pi in partners:
        for ch in pi:
            if ch not in chain_seqs.keys():
                return None

    p0, p1 = partners[0], partners[1]
    component_chains_partners = [t for t in p0] + [t for t in p1]
    component_chain_lengths = get_chain_lengths(chain_seqs,
                                                component_chains_partners)

    try:
        protein_geometry_mats = get_dist_angle_mat(pdb_file,
                                               component_chains_partners,
                                               use_pairwise_geometry=
                                               use_pairwise_geometry,
                                               get_cb_coords=get_cb_coords,
                                               get_bk_and_cb_coords=get_bk_and_cb_coords)
    except:
        return None
    dist_angle_mat = protein_geometry_mats[0]
    dist_mat = protein_geometry_mats[1]
    if get_cb_coords or get_bk_and_cb_coords:
        coords = protein_geometry_mats[2]
    
    phi_psi_mat = protein_bb_phi_psi_matrix(pdb_file,
                                            chains=component_chains_partners)
    
    uniq_epi_indices = _get_ppi_contact_residues(dist_mat,
                                                 chain_seqs,
                                                 partners,
                                                 distance_cutoff=distance_cutoff)

    logger.debug('epi indices {}: {}'.format(p0, uniq_epi_indices[p0]))
    logger.debug('epi indices {}: {}'.format(p1, uniq_epi_indices[p1]))
    for pi in partners:
        if len(uniq_epi_indices[pi]) == 0:
            return None
    #Convert contacts to contiguous ranges
    # [1,2,3,7,8,9,11] -> [(1,3), (7-11)] etc.
    lengths = get_chain_lengths(chain_seqs, partners)
    
    logger.debug('chain lengths: {}\n'.format(lengths))
    uniq_epi_indices_contg_range = {}
    uniq_epi_indices_contg_range[p0] = get_contiguous_fragments(uniq_epi_indices[p0],
                                                           0,
                                                           component_chain_lengths,
                                                           [t for t in p0])
    
    uniq_epi_indices_contg_range[p1] = get_contiguous_fragments(uniq_epi_indices[p1],
                                                           lengths[p0],
                                                           component_chain_lengths,
                                                           [t for t in p1])

    logger.debug('contg {} epi {}'.format(p0, uniq_epi_indices_contg_range[p0]))
    logger.debug('contg {} epi {}\n'.format(p1, uniq_epi_indices_contg_range[p1]))
    
    info.update(
    dict(dist_angle_mat=dist_angle_mat,
        phi_psi_mat=phi_psi_mat))

    if get_cb_coords:
        info.update(dict(cb_coords=coords))

    if get_bk_and_cb_coords:
        info.update(dict(bk_and_cb_coords=coords))

    uniq_epi_indices_range_p1 = [(t[0]-lengths[p0], t[1]-lengths[p0]) \
                                            for t in uniq_epi_indices_contg_range[p1]]
    uniq_epi_indices_contg_range[p1] = uniq_epi_indices_range_p1

    uniq_epi_indices_contg_flat = {}
    uniq_epi_indices_contg_flat[p0] = [t for u in uniq_epi_indices_contg_range[p0]\
                                         for t in range(u[0], u[1]+1)]
    uniq_epi_indices_contg_flat[p1] = [t for u in uniq_epi_indices_contg_range[p1]\
                                         for t in range(u[0], u[1]+1)]
    
    pdb_residue_ids_dict = get_residue_numbering_for_pdb(pdb_file)
    pi_prim,frag_indices, rel_contact_indices, frag_indices_pdb, pdb_indices \
        = {}, {}, {}, {}, {}

    for partner in [p0, p1]:
        pi_prim[partner] = []
        frag_indices[partner] = []
        rel_contact_indices[partner] = []
        pdb_indices[partner] = []
        frag_indices_pdb[partner] = []
        prev_len = 0
        for chain_num, chain in enumerate(partner):
            pi_prim[partner] += chain_seqs[chain]
            frag_indices[partner] += [i+prev_len for i in range(len(chain_seqs[chain]))]
            offset = 100 if chain_num > 0 else 0
            pdb_indices[partner] += [t+offset for t in pdb_residue_ids_dict[chain]]
            prev_len += len(chain_seqs[chain])
        rel_contact_indices[partner] = [frag_indices[partner].index(ind)
                                        for ind in uniq_epi_indices_contg_flat[partner]]
        frag_indices_pdb[partner] = [pdb_indices[partner][t] for t in frag_indices[partner]]
        #print(len(pi_prim[partner]), len(frag_indices_pdb[partner]))
        #print(partner, frag_indices_pdb[partner])
        #print(partner, rel_contact_indices[partner])
    
    #Update info dictionary with fragment information
    info.update(dict(chain_contact_indices_p0=rel_contact_indices[p0]))
    info.update(dict(chain_contact_indices_p1=rel_contact_indices[p1]))
    info.update(dict(p0_prim=pi_prim[p0]))
    info.update(dict(p1_prim=pi_prim[p1]))
    info.update(dict(frag_indices_p0=frag_indices_pdb[p0]))
    info.update(dict(frag_indices_p1=frag_indices_pdb[p1]))
    
    return info


def raw_residues_to_int_residues(res_ids):
    import re
    indices_insertion_codes = [i for i,t in enumerate(res_ids) if re.search('[a-zA-Z]', t)]
    res_ids_nocodes = [re.sub(r'[a-zA-Z]', r'', t) if re.search('[a-zA-Z]', t) else t for t in res_ids ]
    #now increase the value at all positions after insertion codes by 1
    res_ids_new = [int(t) + len([z for z in indices_insertion_codes if z<= i])
                    for i,t in enumerate(res_ids_nocodes)]
    return res_ids_new


def get_non_missing_residue_ids(dfs, partners):
    res_ids = {}
    for df_p, part in zip(dfs, partners):
        res_ids[part] = \
            df_p[(df_p['atom_name']=='CA') &
                (df_p['resname'].isin(list(_aa_3_1_dict.keys())))]['residue'].tolist()
    
    for df_p, part in zip(dfs, partners):
        for atom_name in ['N', 'C']:
            atom_ids = df_p[df_p['atom_name']==atom_name]['residue'].tolist()
            common_ids = [t for t in res_ids[part] if t in atom_ids]
            res_ids[part] = common_ids

    return res_ids


def get_sequence_from_df(dfs, partners, res_ids):
    seqs = {}
    for df_p, part in zip(dfs, partners):
        ca_ids = df_p[df_p['atom_name']=='CA']['residue'].tolist()
        ca_resnames = df_p[df_p['atom_name']=='CA']['resname'].tolist()
        resids_to_prim = {ca_ids[i]:(_aa_3_1_dict[t] 
                            if t in _aa_3_1_dict else 'X')
                            for i,t in enumerate(ca_resnames)}
        seqs[part] = letter_to_num(''.join([resids_to_prim[resid] 
                        for resid in resids_to_prim if resid in res_ids[part]]),
                        _aa_dict)
    return seqs



def get_coords_from_dill_df(df, residue_ids,
                            atom_names = ['N', 'CA', 'C', 'CB']):
    coords = np.zeros((len(atom_names), len(residue_ids), 3))
    #print(coords.shape)
    for i,atom_name in enumerate(atom_names):
        if atom_name == 'CB':
            coords_cb = [df[((df['atom_name']==atom_name) 
                            & (df['residue']==t))][['x', 'y', 'z']].to_numpy().flatten().tolist()
                         if not df[((df['atom_name']==atom_name) 
                            & (df['residue']==t))].empty
                         else get_pseudoatom_cb_from_backbone(coords[atom_names.index('N'), j, :],
                                                              coords[atom_names.index('CA'), j, :],
                                                              coords[atom_names.index('C'), j, :]).tolist()
                         for j,t in enumerate(residue_ids)
                         ]
            coords[i] = np.array(coords_cb)
        else:
            selection_boolean = (df['atom_name']==atom_name) & (df['residue'].isin(residue_ids))
            i_c = df[selection_boolean][['x', 'y', 'z']].to_numpy()
            #print(atom_name, i_c.shape)
            coords[i] = i_c
    return torch.tensor(coords)
    

def add_ppi_info_from_dill_file(dill_file,
                                include_epitope_neighbors=True,
                                distance_cutoff=12.0,
                                distance_cutoff_epitope=12.0,
                                get_cb_coords=False,
                                get_bk_and_cb_coords=False,
                                write_trunc_pdbs=False):
    import dill
    logger = logging.getLogger(__name__ + 'add_ppi_info')
    logger.debug('Adding ppi info ...')
    info = {}
    logger.debug('dill \n'.format(dill_file))
    print(dill_file)
    dill_info = dill.load(open(dill_file, 'rb'))
    
    pdb_id = dill_info[0][:4]
    info['id'] = pdb_id
    df_p0 = dill_info[1]
    df_p1 = dill_info[2]
    p0 = str(df_p0['chain'][0])
    p1 = str(df_p1['chain'][0])
    
    partners = [p0, p1]
    non_missing_residue_ids = get_non_missing_residue_ids([df_p0, df_p1], partners)
    len_p0 = len(non_missing_residue_ids[p0])
    len_p1 = len(non_missing_residue_ids[p1])

    chain_seqs = get_sequence_from_df([df_p0, df_p1], partners, non_missing_residue_ids)

    component_chain_lengths = {p0:len_p0, p1:len_p1}


    if get_cb_coords:
        # Use CA anyway
        atom_names = ['CA']
    elif get_bk_and_cb_coords:
        atom_names = ['N', 'CA', 'C', 'CB']
    else:
        atom_names = ['CA']
    coords_p0 = get_coords_from_dill_df(df_p0, non_missing_residue_ids[partners[0]], atom_names)
    coords_p1 = get_coords_from_dill_df(df_p1, non_missing_residue_ids[partners[1]], atom_names)

    
    coords_p0p1 = torch.cat((coords_p0, coords_p1), dim=1)
    coords_p0p1_ca = coords_p0p1[atom_names.index('CA'), :, :]
    coords_ca_masked = get_atom_coords_mask(coords_p0p1_ca)
    mask=make_square_mask(coords_ca_masked)
    dist_mat = get_masked_mat(calc_dist_mat(coords_p0p1_ca, coords_p0p1_ca).float(),
                             mask=mask)
    # Get contact residues
    # consider when partners are ['AB', 'CD']
    
    uniq_epi_indices = _get_ppi_contact_residues(dist_mat,
                                                 chain_seqs,
                                                 partners,
                                                 distance_cutoff=distance_cutoff)

    logger.debug('epi indices {}: {}'.format(p0, uniq_epi_indices[p0]))
    logger.debug('epi indices {}: {}'.format(p1, uniq_epi_indices[p1]))
    for pi in partners:
        if len(uniq_epi_indices[pi]) == 0:
            return None
    #Convert contacts to contiguous ranges
    # [1,2,3,7,8,9,11] -> [(1,3), (7-11)] etc.
    lengths = get_chain_lengths(chain_seqs, partners)
    
    logger.debug('chain lengths: {}\n'.format(lengths))
    uniq_epi_indices_contg_range = {}
    uniq_epi_indices_contg_range[p0] = get_contiguous_fragments(uniq_epi_indices[p0],
                                                           0,
                                                           component_chain_lengths,
                                                           [t for t in p0])
    
    uniq_epi_indices_contg_range[p1] = get_contiguous_fragments(uniq_epi_indices[p1],
                                                           lengths[p0],
                                                           component_chain_lengths,
                                                           [t for t in p1])

    logger.debug('contg {} epi {}'.format(p0, uniq_epi_indices_contg_range[p0]))
    logger.debug('contg {} epi {}\n'.format(p1, uniq_epi_indices_contg_range[p1]))
    
    uniq_epi_indices_contg_flat = {}
    uniq_epi_indices_contg_flat[p0] = [t for u in uniq_epi_indices_contg_range[p0]\
                                         for t in range(u[0], u[1]+1)]
    uniq_epi_indices_contg_flat[p1] = [t for u in uniq_epi_indices_contg_range[p1]\
                                         for t in range(u[0], u[1]+1)]

    #Add Neighbor "context" for each partner
    uniq_p_indices = _get_ppi_epitope_neighbors(dist_mat,
                                                chain_seqs,
                                                uniq_epi_indices_contg_flat,
                                                partners,
                                                distance_cutoff_epitope=\
                                                    distance_cutoff_epitope)
    logger.debug('nn {}: {}'.format(p0, uniq_p_indices[p0]))                                                    
    logger.debug('nn {}: {}\n'.format(p1, uniq_p_indices[p1]))
    #Make contiguous
    uniq_p_indices_contg_range = {}
    uniq_p_indices_contg_range[p0] = get_contiguous_fragments(uniq_p_indices[p0],
                                                                0,
                                                                component_chain_lengths,
                                                                [t for t in p0])
    
    uniq_p_indices_contg_range[p1] = get_contiguous_fragments(uniq_p_indices[p1],
                                                               lengths[p0],
                                                               component_chain_lengths,
                                                               [t for t in p1])
    
    logger.debug('contg nn {}: {}'.format(p0, uniq_p_indices_contg_range[p0]))
    logger.debug('contg nn {}: {}\n'.format(p1, uniq_p_indices_contg_range[p1]))

    uniq_p_indices_contg_flat_p0 = [t for u in uniq_p_indices_contg_range[p0]\
                                         for t in range(u[0], u[1]+1)]
    uniq_p_indices_contg_flat_p1 = [t for u in uniq_p_indices_contg_range[p1]\
                                         for t in range(u[0], u[1]+1)]

    # Filter distance,angle matrices to keep only contact+context frags
    keep_indices = torch.tensor([t for t in uniq_p_indices_contg_flat_p0] \
                        + [t for t in uniq_p_indices_contg_flat_p1])

    dist_mat.unsqueeze_(0)
    dist_mat_f1 = torch.index_select(dist_mat,1,keep_indices)
    dist_mat_filtered = torch.index_select(dist_mat_f1,2,keep_indices)

    info.update(
    dict(dist_angle_mat=dist_mat_filtered.contiguous()))

    
    if get_cb_coords:
        cb_coords_filtered = torch.index_select(coords_p0p1[:, atom_names.index('CB'), :], 0, keep_indices)
        info.update(dict(cb_coords=cb_coords_filtered))

    if get_bk_and_cb_coords:
        bk_and_cb_coords_filtered = torch.index_select(coords_p0p1, 1, keep_indices)
        info.update(dict(bk_and_cb_coords=bk_and_cb_coords_filtered))

    uniq_epi_indices_range_p1 = [(t[0]-lengths[p0], t[1]-lengths[p0]) \
                                            for t in uniq_epi_indices_contg_range[p1]]
    uniq_epi_indices_contg_range[p1] = uniq_epi_indices_range_p1
    
    uniq_p_indices_contg_range_p1 = [(t[0]-lengths[p0], t[1]-lengths[p0]) \
                                            for t in uniq_p_indices_contg_range[p1]]
    uniq_p_indices_contg_range[p1] = uniq_p_indices_contg_range_p1
    
    contact_frags, all_frags, rel_contact_indices = \
        {}, {}, {}
    
    total_contact_length = 0
    for pi in partners:
        contact_frags[pi], all_frags[pi], contact_length = \
                                            get_epitope_and_neighbor_fragments([t for t in pi],
                                                                    chain_seqs,
                                                                    uniq_epi_indices_contg_range[pi],
                                                                    uniq_p_indices_contg_range[pi])
        #print(contact_frags[pi])
        if contact_length > 350:
            return None
        total_contact_length += contact_length
        rel_contact_indices[pi] = get_relative_positions(all_frags[pi], contact_frags[pi])
        delete_chains = []
        for ch in all_frags[pi]:
            if len(all_frags[pi][ch]) == 0:
                delete_chains.append(ch)
        for ch in delete_chains:
            del all_frags[pi][ch]
        
    
    if total_contact_length > 350:
        return None

    # Get sequence of fragments and contact fragment indices for positional encoding
    pi_prim, frag_indices, pi_chain_begs, pi_c_terms, orig_frags, orig_contact_frags = \
         {}, {}, {}, {}, {}, {}
    
    for pi in partners:
        res_ids = {}
        res_ids[pi] = raw_residues_to_int_residues(non_missing_residue_ids[pi])
        frag_indices[pi], orig_frags[pi], orig_contact_frags[pi] = \
            map_indices_to_pdbids(res_ids, all_frags[pi], contact_frags[pi])
        
        pi_prim[pi], pi_chain_begs[pi], pi_c_terms[pi] = \
                             get_information_from_fragments(all_frags[pi],
                                                            [t for t in pi],
                                                            chain_seqs)
        #print(frag_indices[pi], orig_frags[pi])
    logger.debug(frag_indices)
    if write_trunc_pdbs:
        combined_contact_frags = {}
        combined_contact_frags.update(orig_contact_frags[p0])
        combined_contact_frags.update(orig_contact_frags[p1])
        
        write_pdb_for_fragments(None, combined_contact_frags,
                                partners,
                                pdb_id.upper(),
                                distance_cutoff=distance_cutoff,suffix='_ppi_interface_epi',\
                                prefix='data/ppi_pdbs_int_trunc')
        combined_all_frags = {}
        combined_all_frags.update(orig_frags[p0])
        combined_all_frags.update(orig_frags[p1])

        write_pdb_for_fragments(None, combined_all_frags,
                                partners,
                                pdb_id.upper(),
                                distance_cutoff=distance_cutoff,\
                                suffix='_loop_interface_epi_nn{}'.format(distance_cutoff_epitope),\
                                prefix='data/ppi_pdbs_int_trunc')
    
    #Update info dictionary with fragment information
    info.update(dict(chain_fragments_p0=all_frags[p0]))
    info.update(dict(chain_fragments_p1=all_frags[p1]))
    info.update(dict(chain_contact_indices_p0=rel_contact_indices[p0]))
    info.update(dict(chain_contact_indices_p1=rel_contact_indices[p1]))
    info.update(dict(p0_prim=pi_prim[p0]))
    info.update(dict(p1_prim=pi_prim[p1]))
    info.update(dict(frag_indices_p0=frag_indices[p0]))
    info.update(dict(frag_indices_p1=frag_indices[p1]))
    info.update(dict(p0_chain_begs=pi_chain_begs[p0],
                    p1_chain_begs=pi_chain_begs[p1],
                    p0_c_terms=pi_c_terms[p0],
                    p1_c_terms=pi_c_terms[p1]))

    return info


if __name__ == '__main__':
    print('In ',__name__)
