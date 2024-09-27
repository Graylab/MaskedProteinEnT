import h5py
import operator
import torch
import numpy as np
import os
import torch.utils.data as data

from src.datasets.ppi_graph_dataset_utils \
    import get_fragmented_partner_pos_indices, get_node_feats_for_prot
from src.datasets.InteractingProteinDataClass \
    import AntibodyTypeInteractingProtein, FragmentedMultiChainInteractingProtein,\
         InteractingPartners
from src.datasets.masking_utils import mask_cb_coords, \
        mask_nfeats_cb
from src.data.constants import _aa_dict, letter_to_num

class H5AbAgPPIMaskedMultiAtomDatasetBatched(data.Dataset):
    def __init__(
            self,
            filename,
            gmodel='egnn-trans-ma',
            max_seq_len=None,
            contact_dist_threshold=8.0,
            max_mask=0.40,
            min_mask=0.10,
            noncontact_mask=0.05,
            dataset='ppi',
            partner_selection='random',
            mask_ab_region=None,
            assert_contact=False,
            mask_indices=None
            ):
        """
        :param filename: The h5 file for the antibody data.
            Whether or not to onehot-encode the primary structure data.
        """
        super(H5AbAgPPIMaskedMultiAtomDatasetBatched, self).__init__()
        self.gmodel = gmodel # gt, egnn, egnn-trans
        self.filename = filename
        self.h5file = h5py.File(filename, 'r')
        if dataset == 'ppi':
            self.num_proteins, _ = self.h5file['p0_chain_primary'].shape
        if dataset == 'abag':
            self.num_proteins, _ = self.h5file['heavy_chain_primary'].shape
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.valid_indices = None
        if max_seq_len is not None:
            self.valid_indices = self.get_valid_indices()
            self.num_proteins = len(self.valid_indices)

        self.contact_dist_threshold = contact_dist_threshold
        self.dataset = dataset
        self.percent_mask = [max_mask, min_mask]
        self.noncontact_mask_rate = noncontact_mask
        self.partner_selection = partner_selection # options: 'Ab', 'Ag', 'random'
        self.mask_ab_region = mask_ab_region
        self.mask_ab_indices = mask_indices
        self.intersect_contact_and_region = assert_contact


    def get_item_ppi(self, index):

        id_ = self.h5file['id'][index]
        if len(str(id_)) < 4:
            exit()

        p0_seq_len = self.h5file['p0_chain_seq_len'][index]
        p1_seq_len = self.h5file['p1_chain_seq_len'][index]
        total_seq_len = p0_seq_len + p1_seq_len
        
        p0_prim = self.h5file['p0_chain_primary'][index, :p0_seq_len]
        p1_prim = self.h5file['p1_chain_primary'][index, :p1_seq_len]

        p0_contact_len = self.h5file['p0_contact_len'][index]
        p1_contact_len = self.h5file['p1_contact_len'][index]

        p0_contact_indices = \
            torch.tensor((
                self.h5file['p0_contact_indices'][index,
                                :p0_contact_len]).astype(int)).long()
        p0_frag_positions = \
            torch.tensor((
                self.h5file['p0_frag_positions'][index,
                                :p0_seq_len]).astype(int)).long()
        
        p1_contact_indices = \
            torch.tensor((
                self.h5file['p1_contact_indices'][index,
                                :p1_contact_len]).astype(int)).long()
        p1_frag_positions = \
            torch.tensor((
                self.h5file['p1_frag_positions'][index,
                                :p1_seq_len]).astype(int)).long()

        if 'bk_and_cb_coords' in self.h5file.keys():
            coords = torch.tensor(self.h5file['bk_and_cb_coords'][index][:4, :total_seq_len, :])
        else:
            coords = None
        
        #Setup individual proteins
        p0 =\
            FragmentedMultiChainInteractingProtein(index,
                                                   'p0',
                                                   id_,
                                                   p0_prim,
                                                   p0_contact_indices,
                                                   p0_frag_positions)
        p1 =\
            FragmentedMultiChainInteractingProtein(index,
                                                   'p1',
                                                   id_,
                                                   p1_prim,
                                                   p1_contact_indices,
                                                   p1_frag_positions
                                                   )

        # Try preprocessing - how large?
        p0.to_one_hot()
        p1.to_one_hot()

        # 1. Distance, angle matrix
        try:
            dist_angle_mat = self.h5file['dist_angle_mat'][
                index][:4, :total_seq_len, :total_seq_len]
            dist_angle_mat = torch.Tensor(dist_angle_mat).type(
                dtype=torch.float)
        except Exception:
            raise ValueError('Output matrix not defined')

        len_p0 = len(p0_prim)
        p0.dist_angle_mat = dist_angle_mat[:, :len_p0, :len_p0]
        p1.dist_angle_mat = dist_angle_mat[:, len_p0:, len_p0:]

        return p0, p1, dist_angle_mat, coords

    def get_item_abag(self, index):

        id_ = self.h5file['id'][index]
        #print('get item:', id_)
        if len(str(id_)) < 4:
            print("Invalid pdb id_")
            exit()

        heavy_seq_len = self.h5file['heavy_chain_seq_len'][index]
        light_seq_len = self.h5file['light_chain_seq_len'][index]
        antigen_seq_len = self.h5file['ag_chain_seq_len'][index]
        total_seq_len = heavy_seq_len + light_seq_len + antigen_seq_len
        #print(id_, total_seq_len)

        # Get the attributes from a protein and cut off zero padding
        heavy_prim = self.h5file['heavy_chain_primary'][index, :heavy_seq_len]
        #h_seq = num_to_letter(heavy_prim, _aa_dict)[:heavy_seq_len]
        light_prim = self.h5file['light_chain_primary'][index, :light_seq_len]
        antigen_prim = self.h5file['ag_chain_primary'][index, :antigen_seq_len]
        
        ag_contact_len = self.h5file['ag_contact_len'][index]

        ag_contact_indices = \
            torch.tensor((self.h5file['ag_contact_indices'][index, :ag_contact_len]).astype(int)).long()
        
        ag_frag_positions = \
            torch.tensor((self.h5file['ag_frag_positions'][index, :antigen_seq_len]).astype(int)).long()
        
        # Get CDR loops
        h3 = self.h5file['h3_range'][index]
        h2 = self.h5file['h2_range'][index]
        h1 = self.h5file['h1_range'][index]
        l3 = self.h5file['l3_range'][index]
        l2 = self.h5file['l2_range'][index]
        l1 = self.h5file['l1_range'][index]
        cdrs = [h1, h2, h3, l1, l2, l3]
        cdr_names = ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']

        ab = AntibodyTypeInteractingProtein(index,
                                            'ab',
                                            id_,
                                            heavy_prim,
                                            light_prim,
                                            cdrs=cdrs,
                                            cdr_names=cdr_names)
        ag =\
            FragmentedMultiChainInteractingProtein(index,
                                                   'ag',
                                                   id_,
                                                   antigen_prim,
                                                   ag_contact_indices,
                                                   ag_frag_positions)

        ab.to_one_hot()
        ag.to_one_hot()
        
        # 1. Distance, angle matrix
        try:
            dist_angle_mat = self.h5file['dist_angle_mat'][
                index][:4, :total_seq_len, :total_seq_len]
            dist_angle_mat = torch.Tensor(dist_angle_mat).type(
                dtype=torch.float)
        except Exception:
            raise ValueError('Output matrix not defined')

        len_ag = len(antigen_prim)
        len_ab = len(heavy_prim) + len(light_prim)

        ab.dist_angle_mat = dist_angle_mat[:, :len_ab, :len_ab]
        ag.dist_angle_mat = dist_angle_mat[:, len_ab:len_ab + len_ag,
                                           len_ab:len_ab + len_ag]

        try:
            phi_psi_mat = self.h5file['phi_psi_mat'][index][:2, :total_seq_len]
            phi_psi_mat = torch.Tensor(phi_psi_mat).type(dtype=torch.float)
        
        except Exception:
            raise ValueError('Output matrix not defined')
        # 2. Backbone dihedral matrix
        ab.phi_psi_mat = phi_psi_mat[:, :len_ab]
        ag.phi_psi_mat = phi_psi_mat[:, len_ab:]
        
        if 'bk_and_cb_coords' in self.h5file.keys():
            coords = torch.tensor(self.h5file['bk_and_cb_coords'][index][:4, :total_seq_len, :])
        else:
            coords = None
        
        return ab, ag, dist_angle_mat, coords

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise IndexError('Slicing not supported')
        
        if self.valid_indices is not None:
            index = self.valid_indices[index]
    
        if self.dataset == 'ppi':
            p0, p1, dist_angle_mat, coords = self.get_item_ppi(index)
        if self.dataset == 'abag':
            p0, p1, dist_angle_mat, coords = self.get_item_abag(index)
            # Before masking sequence, get contact residues in antibody
            abag_complex = InteractingPartners(p0, p1, dist_angle_mat)
            abag_complex.setup_paratope_epitope(
                contact_dist_threshold=self.contact_dist_threshold)
            # contact indices initialized as loop residues
            # Now replace them with loop residues in contact
            paratope_mask = abag_complex.paratope.tolist()
            p0.contact_indices = [i for i in range(len(paratope_mask)) if paratope_mask[i]==1]
            
        mask_residue_selection = None
        if self.partner_selection == 'random': # Default
            percent_index = np.random.random_integers(0, 1, 1)[0]
        elif self.partner_selection == 'Ab' or self.partner_selection=='both':
            percent_index = 0
            #print(self.mask_ab_region)
            if self.mask_ab_region != None:
                mask_residue_selection = []
                if self.mask_ab_region == 'non-cdr':
                    all_indices = [t for t in range(p0.seq_len)]
                    mask_residue_selection = [t for t in all_indices 
                                                if t not in p0.loop_indices.tolist()]

                if self.mask_ab_region == 'cdrs':
                    all_indices = [t for t in range(p0.seq_len)]
                    mask_residue_selection = [t for t in all_indices 
                                                if t in p0.loop_indices.tolist()]
                if self.mask_ab_region == 'non-contact':
                    all_indices = [t for t in range(p0.seq_len)]
                    mask_residue_selection = [t for t in all_indices 
                                            if t not in p0.contact_indices.tolist()]
                        

                for region in ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']:
                    if region == self.mask_ab_region and \
                        (region in p0.cdr_indices_dict):
                        mask_residue_selection = p0.cdr_indices_dict[region]
                        break
            if self.mask_ab_indices != None:
                mask_residue_selection = self.mask_ab_indices
        elif self.partner_selection == 'Ag':
            percent_index = 1
        else:
            import sys
            sys.exit('Partner selection not defined.')
        #print(p0.id, mask_residue_selection)
        sequence_masked_label_p0 = \
            p0.mask_sequence(contact_percent=self.percent_mask[percent_index],
                             subset_selection_indices=mask_residue_selection,
                             select_intersection=self.intersect_contact_and_region,
                             non_contact_percent=self.noncontact_mask_rate)
        sequence_masked_label_p1 = \
            p1.mask_sequence(contact_percent=self.percent_mask[percent_index-1],
                             non_contact_percent=self.noncontact_mask_rate)

        nfeats_p0, _ = \
            get_node_feats_for_prot(p0,
                                    masked=True)
        nfeats_p1, _ = \
            get_node_feats_for_prot(p1,
                                    masked=True)
        
        nfeats = torch.cat([nfeats_p0, nfeats_p1], dim=0)
        sequence_label = torch.cat(
            [sequence_masked_label_p0, sequence_masked_label_p1], dim=0)
        
        assert len(coords.shape) == 3
        self.num_atoms = coords.shape[0]
        split_coords = coords.permute(1, 0, 2)
        coords = mask_cb_coords(sequence_label, split_coords)
        # II. adding atom feats
        num_res = nfeats.shape[0]
        nfeats = nfeats.unsqueeze(1).expand(-1, self.num_atoms, -1).reshape(-1, nfeats.shape[-1])
        nfeats_atoms = torch.eye(self.num_atoms).unsqueeze(0).expand(num_res, -1, -1).reshape(-1, self.num_atoms)
        # masking cb atoms of label residues
        nfeats = mask_nfeats_cb(sequence_label, nfeats, nfeats_atoms)
            
        seq_length = p0.seq_len + p1.seq_len
        adj_mat = torch.ones((nfeats.shape[0], nfeats.shape[0])).long()
        
        assert coords is not None
        missing_residues_mask = torch.ones((nfeats.shape[0])).long()

        p0_noncontact_mask = p0.mask_non_contact_indices(torch.ones((p0.seq_len)))
        p1_noncontact_mask = p1.mask_non_contact_indices(torch.ones((p1.seq_len)))
        noncontact_residues_mask = torch.cat([p0_noncontact_mask, p1_noncontact_mask], dim=0).long()
        
        pos_indices = get_fragmented_partner_pos_indices(p0.fragment_indices,
                                                         p1.fragment_indices,
                                                         self.num_atoms)
        assert pos_indices.shape[0] == nfeats.shape[0]
        metadata = {}
        if self.dataset == 'abag':
            chain_breaks = [p0.heavy_len, p0.seq_len]
            metadata = {'cdrs':p0.cdr_indices_dict, 'Ab_len': p0.seq_len, 'Ag_len': p1.seq_len,
                        'Ab_seq':p0.prim, 'h_len': p0.heavy_len, 'chain_breaks':chain_breaks}
        else:
            chain_breaks = [p0.seq_len]
            metadata = {'p0_len': p0.seq_len, 'p1_len':p1.seq_len, 'chain_breaks':chain_breaks}
        metadata.update({'noncontact_mask': noncontact_residues_mask})
        assert len(coords.shape) == 2
        return p0.id, nfeats, coords, adj_mat, missing_residues_mask, \
                sequence_label, pos_indices, metadata


    def get_valid_indices(self):
        """Gets indices with proteins less than the max sequence length
        :return: A list of indices with sequence lengths less than the max
                 sequence length.
        :rtype: list
        """
        valid_indices = []
        if self.dataset == 'ppi':
            n_seqs = self.h5file['p0_chain_seq_len'].shape[0]
        else:
            n_seqs = self.h5file['heavy_chain_seq_len'].shape[0]

        for i in range(n_seqs):
            id_ = self.h5file['id'][i]
            if len(str(id_)) < 4:
                #print('Invalid id', i, id_)
                continue
            if self.dataset == 'ppi':
                p0_len = self.h5file['p0_chain_seq_len'][i]
                p1_len = self.h5file['p1_chain_seq_len'][i]
            else:
                p0_len = self.h5file['heavy_chain_seq_len'][i] + \
                        self.h5file['light_chain_seq_len'][i]
                p1_len = self.h5file['ag_chain_seq_len'][i]
            total_seq_len = p0_len + p1_len
            if total_seq_len < self.max_seq_len and \
                p0_len > 0 and \
                    p1_len > 0:
                valid_indices.append(i)
        return valid_indices

    def get_sorted_indices(self):
        lengths = []
        if self.dataset == 'ppi':
            n_seqs = self.h5file['p0_chain_seq_len'].shape[0]
        else:
            n_seqs = self.h5file['heavy_chain_seq_len'].shape[0]
        for i in range(n_seqs):

            id_ = self.h5file['id'][i]
            if len(str(id_)) < 4:
                #print('Invalid id', i, id_)
                continue
            if self.dataset == 'ppi':
                p0_len = self.h5file['p0_chain_seq_len'][i]
                p1_len = self.h5file['p1_chain_seq_len'][i]
            else:
                p0_len = self.h5file['heavy_chain_seq_len'][i] + \
                        self.h5file['light_chain_seq_len'][i]
                p1_len = self.h5file['ag_chain_seq_len'][i]
            if p0_len <= 0 | p1_len <= 0:
                continue
            lengths.append((i, p0_len + p1_len, p0_len, p1_len))

        lengths = [(t[0], t[2], t[3]) for t in lengths]
        sorted_lengths = sorted(lengths,
                                key=operator.itemgetter(1, 2),
                                reverse=True)

        return [t[0] for t in sorted_lengths]

    def __len__(self):
        return self.num_proteins