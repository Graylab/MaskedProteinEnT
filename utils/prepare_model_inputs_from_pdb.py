import os
import sys
import torch
import glob
import numpy as np
from tqdm import tqdm
from src.data.constants import _aa_dict, letter_to_num
from src.data.InteractingProteinDataClass \
    import AntibodyTypeInteractingProtein,\
        FragmentedMultiChainInteractingProtein, InteractingPartners
from src.data.masking_utils import mask_cb_coords, mask_nfeats_cb
from src.data.ppi_graph_dataset_utils import get_fragmented_partner_pos_indices
from src.data.utils.pdb import protein_dist_coords_matrix, get_residue_numbering_for_pdb, get_indices_dict_for_cdrs
from src.data.preprocess.antigen_parser import get_chain_lengths, _get_ppi_contact_residues, add_protein_info

from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder


def protein_data_batcher(id, feats, coords, mask, seqlabel, 
                         seq_positions, metadata, with_metadata=False):
    batched_tens = {}
    batched_tens['feats'] = [feats]
    batched_tens['coords'] = [coords]
    batched_tens['mask'] = [mask]
    batched_tens['seq_label'] = [seqlabel]
    batched_tens['seq_pos'] = [seq_positions]
    
    max_n = max([t.shape[0] for t in batched_tens['feats']])
    max_n_labels = max([t.shape[0] for t in batched_tens['seq_label']])
    #print('Max ', max_n, [t.shape[0] for t in batched_tens['feats']])        
    for key in batched_tens:
        if key == 'mask' or key == 'feats' or key == 'seq_pos':
            # 1D
            padded_tens = [
                torch.cat([t, torch.zeros((max_n - t.shape[0]))], dim=0)
                for t in batched_tens[key]
            ]
        elif key == 'seq_label':
            # 1D - pad with ignore value
            padded_tens = [
                torch.cat([
                    t,
                    torch.ones((max_n_labels - t.shape[0])) * len(_aa_dict.keys())
                ],
                            dim=0) for t in batched_tens[key]
            ]
        else:
            # 2D
            #print(key, [t.shape for t in batched_tens[key]], max_n)
            padded_tens = [
                torch.cat(
                    [t, torch.zeros((max_n - t.shape[0], t.shape[1]))],
                    dim=0) for t in batched_tens[key]
            ]
        # batched
        if key != 'seq_label' and key!= 'seq_pos':
            batched_tens[key] = torch.stack(padded_tens, dim=0)
        else:
            batched_tens[key] = torch.cat(padded_tens, dim=0)
    
    if with_metadata:
        return [id], (batched_tens['feats'].long(), batched_tens['coords'].double(),\
            None, batched_tens['mask'].bool(),\
                batched_tens['seq_label'].long(), batched_tens['seq_pos'].long()), [metadata]
    else:
        return [id], (batched_tens['feats'].long(), batched_tens['coords'].double(),\
            None, batched_tens['mask'].bool(),\
                batched_tens['seq_label'].long(), batched_tens['seq_pos'].long())


def get_protein_info_from_pdb_file(pdb_file, max_id_len=40,
                                   index=0, with_metadata=False,
                                   mask_fraction=1.0,
                                   label_seq={}):

    chain_seqs = dict()
    parser = PDBParser()
    id=os.path.basename(pdb_file)[:-4][:max_id_len]
    structure = parser.get_structure(id, pdb_file)
    
    for chain in structure.get_chains():
        seq = seq1(''.join([residue.resname for residue in chain\
                    if 'CA' in residue]))
        chain_seqs.update({chain.id: letter_to_num(seq, _aa_dict)})
    
    if label_seq != {}:
        for ch in chain_seqs:
            if ch in label_seq:
                seq_chain = label_seq[ch]
            else:
                print(f'DID not find seq for {ch} from pdb_file {pdb_file} in {label_seq}')
                print('Sequence scores may not be accurate.')

    info = {}
    info.update(chain_seqs)
    info.update(dict(id=id))
    info = add_protein_info(info,
                            pdb_file,
                            chain_seqs=chain_seqs)
    sequence_len = sum([len(chain_seqs[key]) for key in chain_seqs])
    prim = []
    fragment_indices = []
    offset = 0
    for ichain, chain in enumerate(info['protein_chains']):
        prim += info[chain]
        offset = (len(prim)+100)*ichain
        fragment_indices += [t+offset for t in range(len(info[chain]))]
    
    assert len(fragment_indices) == sequence_len
    p0 = FragmentedMultiChainInteractingProtein(
            index,
            'protein',
            id,
            contact_indices=torch.tensor([t for t in range(sequence_len)]).long(),
            fragment_indices=torch.tensor(fragment_indices).long(),
            prim=prim,
            dist_angle_mat=info['dist_mat'])
    p0.to_one_hot()
    p0.dist_angle_mat_inputs = p0.dist_angle_mat

    sequence_masked_label_p0 = \
        p0.mask_sequence(contact_percent=mask_fraction)
    nfeats = p0.prim_masked.float()
    sequence_label = sequence_masked_label_p0
    num_res = nfeats.shape[0]
    coords = torch.tensor(info['bk_and_cb_coords']).permute(1, 0, 2)
    num_atoms = coords.shape[1]
    nfeats = nfeats.unsqueeze(1).expand(-1, num_atoms, -1).reshape(-1, nfeats.shape[-1])
    nfeats_atoms = torch.eye(num_atoms).unsqueeze(0).expand(num_res, -1, -1).reshape(-1, num_atoms)
    nfeats = mask_nfeats_cb(sequence_label, nfeats, nfeats_atoms)
    nfeats = nfeats.double()
    seq_positions_res = torch.arange(nfeats.shape[0] // num_atoms)
    seq_positions = seq_positions_res.repeat_interleave(num_atoms)    
    assert coords is not None
    coords = mask_cb_coords(sequence_label, coords)
    missing_residues_mask = torch.ones((nfeats.shape[0])).long()
    batch = protein_data_batcher(p0.id, nfeats, coords, missing_residues_mask,
                        sequence_label, seq_positions, {}, with_metadata=with_metadata)
    return batch


def add_ppi_info_full(info,pdb_file,chain_seqs,
                        distance_cutoff=12.0,
                        partners=[],
                        mask_fill_value=-999):
    
    for pi in partners:
        for ch in pi:
            if ch not in chain_seqs.keys():
                return None
    
    p0, p1 = partners[0], partners[1]
    component_chains_partners = [t for t in p0] + [t for t in p1]
    component_chain_lengths = get_chain_lengths(chain_seqs,
                                                component_chains_partners)
    l0, l1 = sum([component_chain_lengths[t] for t in p0]), sum([component_chain_lengths[t] for t in p1]) 
    protein_geometry_mats = protein_dist_coords_matrix(pdb_file,
                                                       chains=component_chains_partners,
                                                       mask_fill_value=mask_fill_value)
    
    dist_mat = protein_geometry_mats[0]
    coords = protein_geometry_mats[1]
    dist_mat_p0_p1 = dist_mat[:l0, l0:]

    contact_indices_partners = {}
    contact_indices_partners[p0] = list(set((np.argwhere((dist_mat_p0_p1 <= distance_cutoff) & 
                                     (dist_mat_p0_p1 != mask_fill_value) )[0]).tolist()))
    contact_indices_partners[p1] = list(set((np.argwhere((dist_mat_p0_p1 <= distance_cutoff) & 
                                     (dist_mat_p0_p1 != mask_fill_value) )[1]).tolist()))

    #print('epi indices {}: {}'.format(p0, contact_indices_partners[p0]))
    #print('epi indices {}: {}'.format(p1, contact_indices_partners[p1]))
    for pi in partners:
        if len(contact_indices_partners[pi]) == 0:
            return None
    
    lengths = get_chain_lengths(chain_seqs, partners)
    
    info.update(
    dict(dist_mat=dist_mat.unsqueeze(0)))

    info.update(dict(bk_and_cb_coords=coords))

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
            #print(partner, len(chain_seqs[chain]))
        #print(partner, frag_indices[partner])
        #print(contact_indices_partners[partner])
        rel_contact_indices[partner] = [frag_indices[partner].index(ind)
                                        for ind in contact_indices_partners[partner]]
        frag_indices_pdb[partner] = [pdb_indices[partner][t] for t in frag_indices[partner]]
    
    info.update(dict(chain_contact_indices_p0=rel_contact_indices[p0]))
    info.update(dict(chain_contact_indices_p1=rel_contact_indices[p1]))
    info.update(dict(p0_prim=pi_prim[p0]))
    info.update(dict(p1_prim=pi_prim[p1]))
    info.update(dict(frag_indices_p0=frag_indices_pdb[p0]))
    info.update(dict(frag_indices_p1=frag_indices_pdb[p1]))
    
    return info


def get_ppi_info_from_pdb_file(pdb_file, max_id_len=40,
                               index=0, with_metadata=False,
                               partners=[],
                               mr_p0=1,
                               mr_p1=0):
    
    chain_seqs = dict()
    parser = PDBParser()
    id=os.path.basename(pdb_file)[:-4][:max_id_len]
    structure = parser.get_structure(id, pdb_file)
    partner_chains = [u for t in partners for u in t]
    
    for chain in structure.get_chains():
        skip_chain = False
        seq = seq1(''.join([residue.resname for residue in chain\
                    if 'CA' in residue]))
        for aa in seq:
            if not aa in list(_aa_dict.keys()):
                skip_chain = True
        if skip_chain:
            continue
        if chain.id in partner_chains:
            chain_seqs.update({chain.id: letter_to_num(seq, _aa_dict)})
    
    for partner in partners:
        for chain in partner:
            if not chain in chain_seqs:
                return None
    info = {}
    info.update(chain_seqs)
    info.update(dict(id=id))
    if len(chain_seqs.keys())==2:
        partners = list(chain_seqs.keys())
    
    info = add_ppi_info_full(info, pdb_file, chain_seqs,
                        distance_cutoff=12.0,
                        partners=partners)
    
    if info is None:
        return
    
    chain_breaks = []
    length = 0
    for partner in partners:
        for chain in partner:
            length += len(chain_seqs[chain])
            chain_breaks.append(length)
    
    p0_prim = info['p0_prim']
    p1_prim = info['p1_prim']
    p0_contact_indices = info['chain_contact_indices_p0']
    p1_contact_indices = info['chain_contact_indices_p1']
    p0_frag_positions = torch.tensor(info['frag_indices_p0']).long()
    p1_frag_positions = torch.tensor(info['frag_indices_p1']).long()
    p0 =\
            FragmentedMultiChainInteractingProtein(index,
                                                   'p0',
                                                   id,
                                                   p0_prim,
                                                   p0_contact_indices,
                                                   p0_frag_positions)
    p1 =\
            FragmentedMultiChainInteractingProtein(index,
                                                   'p1',
                                                   id,
                                                   p1_prim,
                                                   p1_contact_indices,
                                                   p1_frag_positions
                                                   )
    p0.to_one_hot()
    p1.to_one_hot()
    len_p0 = len(p0_prim)
    dist_angle_mat = info['dist_mat']
    p0.dist_angle_mat = dist_angle_mat[:, :len_p0, :len_p0]
    p1.dist_angle_mat = dist_angle_mat[:, len_p0:, len_p0:]
    coords = info['bk_and_cb_coords']

    sequence_masked_label_p0 = \
        p0.mask_sequence(contact_percent=mr_p0)
    sequence_masked_label_p1 = \
        p1.mask_sequence(contact_percent=mr_p1)
    
    nfeats_p0 = p0.prim_masked.float()
    nfeats_p1 = p1.prim_masked.float()

    nfeats = torch.cat([nfeats_p0, nfeats_p1], dim=0)
    sequence_label = torch.cat(
            [sequence_masked_label_p0, sequence_masked_label_p1], dim=0)

    num_res = nfeats.shape[0]
    split_coords = torch.tensor(info['bk_and_cb_coords']).permute(1, 0, 2)
    num_atoms = split_coords.shape[1]
    nfeats = nfeats.unsqueeze(1).expand(-1, num_atoms, -1).reshape(-1, nfeats.shape[-1])
    nfeats_atoms = torch.eye(num_atoms).unsqueeze(0).expand(num_res, -1, -1).reshape(-1, num_atoms)
    nfeats = mask_nfeats_cb(sequence_label, nfeats, nfeats_atoms)
    nfeats = nfeats.double()

    coords = mask_cb_coords(sequence_label, split_coords)
    p0_noncontact_mask = p0.mask_non_contact_indices(torch.ones((p0.seq_len)))
    p1_noncontact_mask = p1.mask_non_contact_indices(torch.ones((p1.seq_len)))
    noncontact_residues_mask = torch.cat([p0_noncontact_mask, p1_noncontact_mask], dim=0).long()
            
    pos_indices = get_fragmented_partner_pos_indices(p0.fragment_indices,
                                                     p1.fragment_indices,
                                                     num_atoms)
    metadata = {'p0_len': p0.seq_len, 'p1_len':p1.seq_len, 'noncontact_mask': noncontact_residues_mask,
                'chain_breaks':chain_breaks}
    assert len(coords.shape) == 2
    missing_residues_mask = torch.ones((nfeats.shape[0])).long()
    batch = protein_data_batcher(p0.id, nfeats, coords, missing_residues_mask,
                                sequence_label, pos_indices, metadata, with_metadata=with_metadata)
    return batch


def get_abag_info_from_pdb_file(pdb_file, max_id_len=40,
                               index=0, with_metadata=False,
                               partners=[],
                               mr_p0=1,
                               mr_p1=0,
                               partner_selection='Ab',
                               mask_ab_region=None,
                               mask_ab_indices=None,
                               assert_contact=False):
    
    chain_seqs = dict()
    parser = PDBParser()
    id=os.path.basename(pdb_file)[:-4][:max_id_len]
    structure = parser.get_structure(id, pdb_file)
    partner_chains = [u for t in partners for u in t ]
    for chain in structure.get_chains():
        skip_chain = False
        seq = seq1(''.join([residue.resname for residue in chain\
                    if 'CA' in residue]))
        for aa in seq:
            if not aa in list(_aa_dict.keys()):
                skip_chain = True
        if skip_chain:
            continue
        if chain.id in partner_chains:
            chain_seqs.update({chain.id: letter_to_num(seq, _aa_dict)})
    
    for partner in partners:
        for chain in partner:
            if not chain in chain_seqs:
                return None
    info = {}
    info.update(chain_seqs)
    info.update(dict(id=id))
    if len(chain_seqs.keys())==2:
        partners = list(chain_seqs.keys())
    chain_breaks = []
    length = 0
    for partner in partners:
        for chain in partner:
            length += len(chain_seqs[chain])
            chain_breaks.append(length)

    info = add_ppi_info_full(info, pdb_file, chain_seqs,
                        distance_cutoff=12.0,
                        partners=partners)
    if info is None:
        return
    
    # ab ag specific stuff
    ab_chains = partners[0]
    heavy_chain = ab_chains[0]
    heavy_prim = chain_seqs[heavy_chain]
    light_chain = ''
    light_prim = []
    if len(ab_chains)==2:
        light_chain = ab_chains[1]
        light_prim = chain_seqs[light_chain]
    
    ag_chains = partners[1]
    antigen_prim = [t for chain in ag_chains for t in chain_seqs[chain]]

    ab_len = len(heavy_prim) + len(light_prim)
    ag_len = len(antigen_prim)

    ab_contact_indices = info['chain_contact_indices_p0']
    ag_contact_indices = info['chain_contact_indices_p1']
    ab_frag_positions = torch.tensor(info['frag_indices_p0']).long()
    ag_frag_positions = torch.tensor(info['frag_indices_p1']).long()


    cdr_names = ['h1', 'h2', 'h3']
    if light_chain != '':
        cdr_names += ['l1', 'l2', 'l3']
    cdr_indices = get_indices_dict_for_cdrs(pdb_file, per_chain_dict=False)
    cdrs = [(indices[0], indices[-1]) for cdr_name, indices in cdr_indices.items()]
    
    ab = AntibodyTypeInteractingProtein(index,
                                        'ab',
                                        id,
                                        heavy_prim,
                                        light_prim,
                                        cdrs=cdrs,
                                        cdr_names=cdr_names)
    ab.set_contact_indices(ab_contact_indices)
    ag = FragmentedMultiChainInteractingProtein(index,
                                                'ag',
                                                id,
                                                antigen_prim,
                                                ag_contact_indices,
                                                ag_frag_positions)

    
    ab.to_one_hot()
    ag.to_one_hot()
    dist_angle_mat = info['dist_mat']
    ab.dist_angle_mat = dist_angle_mat[:, :ab_len, :ab_len]
    ag.dist_angle_mat = dist_angle_mat[:, ab_len:, ab_len:]
    
    
    mask_residue_selection = None
    if partner_selection == 'Ab' or partner_selection=='both':
        if not mask_ab_region is None:
            if mask_ab_region == 'non-cdr':
                all_indices = [t for t in range(ab.seq_len)]
                mask_residue_selection = [t for t in all_indices 
                                            if t not in ab.loop_indices.tolist()]

            if mask_ab_region == 'cdrs':
                all_indices = [t for t in range(ab.seq_len)]
                mask_residue_selection = [t for t in all_indices 
                                            if t in ab.loop_indices.tolist()]
            
            if mask_ab_region == 'non-contact':
                all_indices = [t for t in range(ab.seq_len)]
                mask_residue_selection = [t for t in all_indices 
                                            if t not in ab.contact_indices.tolist()]

            for region in ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']:
                if region == mask_ab_region and \
                    (region in ab.cdr_indices_dict):
                    mask_residue_selection = ab.cdr_indices_dict[region]
                    break
        if not mask_ab_indices is None:
            mask_residue_selection = mask_ab_indices

    sequence_masked_label_p0 = \
        ab.mask_sequence(contact_percent=mr_p0,
                        subset_selection_indices=mask_residue_selection,
                        select_intersection=assert_contact,
                        non_contact_percent=0)
    sequence_masked_label_p1 = \
        ag.mask_sequence(contact_percent=mr_p1,
                        non_contact_percent=0)
    
    nfeats_p0 = ab.prim_masked.float()
    nfeats_p1 = ag.prim_masked.float()

    nfeats = torch.cat([nfeats_p0, nfeats_p1], dim=0)
    sequence_label = torch.cat(
            [sequence_masked_label_p0, sequence_masked_label_p1], dim=0)

    num_res = nfeats.shape[0]
    split_coords = torch.tensor(info['bk_and_cb_coords']).permute(1, 0, 2)
    num_atoms = split_coords.shape[1]
    nfeats = nfeats.unsqueeze(1).expand(-1, num_atoms, -1).reshape(-1, nfeats.shape[-1])
    nfeats_atoms = torch.eye(num_atoms).unsqueeze(0).expand(num_res, -1, -1).reshape(-1, num_atoms)
    nfeats = mask_nfeats_cb(sequence_label, nfeats, nfeats_atoms)
    nfeats = nfeats.double()

    coords = mask_cb_coords(sequence_label, split_coords)
    p0_noncontact_mask = ab.mask_non_contact_indices(torch.ones((ab.seq_len)))
    p1_noncontact_mask = ag.mask_non_contact_indices(torch.ones((ag.seq_len)))
    noncontact_residues_mask = torch.cat([p0_noncontact_mask, p1_noncontact_mask], dim=0).long()
            
    pos_indices = get_fragmented_partner_pos_indices(ab.fragment_indices,
                                                     ag.fragment_indices,
                                                     num_atoms)
    metadata = {'cdrs':ab.cdr_indices_dict, 'Ab_len': ab.seq_len, 'Ag_len': ag.seq_len,
                'Ab_seq':ab.prim, 'h_len': ab.heavy_len, 'noncontact_mask': noncontact_residues_mask,
                'chain_breaks':chain_breaks}
    assert len(coords.shape) == 2
    missing_residues_mask = torch.ones((nfeats.shape[0])).long()
    batch = protein_data_batcher(ab.id, nfeats, coords, missing_residues_mask,
                                sequence_label, pos_indices, metadata, with_metadata=with_metadata)
    return batch


def get_antibody_info_from_pdb_file(pdb_file, max_id_len=40,
                                    mr=1.0,
                                    index=0, with_metadata=False,
                                    mask_ab_region=None,
                                    mask_ab_indices=None):

    chain_seqs = dict()
    parser = PDBParser()
    id=os.path.basename(pdb_file)[:-4][:max_id_len]
    structure = parser.get_structure(id, pdb_file)
    
    for chain in structure.get_chains():
        seq = seq1(''.join([residue.resname for residue in chain\
                    if 'CA' in residue]))
        chain_seqs.update({chain.id: letter_to_num(seq, _aa_dict)})
    info = {}
    info.update(chain_seqs)
    info.update(dict(id=id))
    info = add_protein_info(info,
                            pdb_file,
                            chain_seqs=chain_seqs)
    
    sequence_len = sum([len(chain_seqs[key]) for key in chain_seqs])
    prim = []
    fragment_indices = []
    offset = 0
    for ichain, chain in enumerate(info['protein_chains']):
        prim += info[chain]
        offset = (len(prim)+100)*ichain
        fragment_indices += [t+offset for t in range(len(info[chain]))]
    
    assert len(fragment_indices) == sequence_len

    ab_chains = list(chain_seqs.keys())
    heavy_chain = ab_chains[0]
    heavy_prim = chain_seqs[heavy_chain]
    light_chain = ''
    light_prim = []
    if len(ab_chains)==2:
        light_chain = ab_chains[1]
        light_prim = chain_seqs[light_chain]
    
    ab_len = len(heavy_prim) + len(light_prim)

    cdr_names = ['h1', 'h2', 'h3']
    if light_chain != '':
        cdr_names += ['l1', 'l2', 'l3']
    cdr_indices = get_indices_dict_for_cdrs(pdb_file, per_chain_dict=False)
    cdrs = [indices for cdr_name, indices in cdr_indices.items()]

    ab = AntibodyTypeInteractingProtein(index,
                                        'ab',
                                        id,
                                        heavy_prim,
                                        light_prim,
                                        cdrs=cdrs,
                                        cdr_names=cdr_names)
    mask_residue_selection = None
    if not mask_ab_region is None:
        if mask_ab_region == 'non-cdr':
            all_indices = [t for t in range(ab.seq_len)]
            mask_residue_selection = [t for t in all_indices 
                                        if t not in ab.loop_indices.tolist()]

        if mask_ab_region == 'cdrs':
            all_indices = [t for t in range(ab.seq_len)]
            mask_residue_selection = [t for t in all_indices 
                                        if t in ab.loop_indices.tolist()]
        
        if mask_ab_region == 'non-contact':
            all_indices = [t for t in range(ab.seq_len)]
            mask_residue_selection = [t for t in all_indices 
                                        if t not in ab.contact_indices.tolist()]

        for region in ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']:
            if region == mask_ab_region and \
                (region in ab.cdr_indices_dict):
                mask_residue_selection = ab.cdr_indices_dict[region]
                break
    if not mask_ab_indices is None:
        mask_residue_selection = mask_ab_indices

    ab.set_contact_indices(torch.tensor([t for t in range(sequence_len)]).long())
    
    ab.to_one_hot()
    ab.dist_angle_mat_inputs = ab.dist_angle_mat

    sequence_masked_label_p0 = \
        ab.mask_sequence(contact_percent=mr,
                        subset_selection_indices=mask_residue_selection,
                        non_contact_percent=0)
    nfeats = ab.prim_masked.float()
    sequence_label = sequence_masked_label_p0
    num_res = nfeats.shape[0]
    coords = torch.tensor(info['bk_and_cb_coords']).permute(1, 0, 2)
    num_atoms = coords.shape[1]
    nfeats = nfeats.unsqueeze(1).expand(-1, num_atoms, -1).reshape(-1, nfeats.shape[-1])
    nfeats_atoms = torch.eye(num_atoms).unsqueeze(0).expand(num_res, -1, -1).reshape(-1, num_atoms)
    nfeats = mask_nfeats_cb(sequence_label, nfeats, nfeats_atoms)
    nfeats = nfeats.double()
    seq_positions_res = torch.arange(nfeats.shape[0] // num_atoms)
    seq_positions = seq_positions_res.repeat_interleave(num_atoms)    
    assert coords is not None
    coords = mask_cb_coords(sequence_label, coords)
    missing_residues_mask = torch.ones((nfeats.shape[0])).long()
    batch = protein_data_batcher(ab.id, nfeats, coords, missing_residues_mask,
                        sequence_label, seq_positions, {}, with_metadata=with_metadata)
    return batch


