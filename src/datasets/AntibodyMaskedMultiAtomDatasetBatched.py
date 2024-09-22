import torch
import numpy as np
import h5py
import os, sys
import torch.utils.data as data

from src.datasets.SCNProteinMaskedMultiAtomDatasetBatched import SCNProteinMaskedMultiAtomBatchBatched
from src.datasets.masking_utils import mask_cb_coords, mask_nfeats_cb
from src.datasets.ppi_graph_dataset_utils \
    import get_inter_intra_edges, get_node_feats_for_prot
from src.datasets.InteractingProteinDataClass \
    import AntibodyTypeInteractingProtein
from src.data.constants import _aa_dict, letter_to_num


class AntibodyMaskedMultiAtomDatasetBatched(data.Dataset):
    def __init__(
            self,
            h5file,
            onehot_prim=True,
            max_seq_len=350,
            pe_nodes=16,
            pe_edges=16,
            contact_dist_threshold=8.0,
            max_mask=0.15,
            min_mask=0.0,
            gmodel='egnn-trans-ma',
            metadata_extra=False,
            span=True,
            min_span_length=3,
            max_span_length=5,
            mask_ab_region=None,
            mask_ab_indices=None
            ):
        """
        :param filename: The h5 file for the antibody data.
        :param onehot_prim:
            Whether or not to onehot-encode the primary structure data.
        """
        super(AntibodyMaskedMultiAtomDatasetBatched, self).__init__()
        self.h5file = h5py.File(h5file, 'r')
        self.num_proteins = self.h5file['heavy_chain_seq_len'].shape[0]
        self.onehot_prim = onehot_prim
        self.max_seq_len = max_seq_len
        self.valid_indices = None
        self.fixed_atoms = False

        self.n_pe_nodes = pe_nodes
        self.n_pe_edges = pe_edges
        self.contact_dist_threshold = contact_dist_threshold
        self.masking_rate_max = max_mask
        self.masking_rate_min = min_mask
        self.gmodel = gmodel
        self.metadata_extra = metadata_extra

        #spans, coords
        self.span_mask = span
        self.min_frag = min_span_length
        self.max_frag = max_span_length
        self.masked_residue_indices = None
        self.mask_ab_region = mask_ab_region
        self.mask_ab_indices = mask_ab_indices

    
    def __getitem__(self, index):

        id_ = self.h5file['id'][index]
        #print(id_)

        heavy_seq_len = self.h5file['heavy_chain_seq_len'][index]
        light_seq_len = self.h5file['light_chain_seq_len'][index]
        total_seq_len = heavy_seq_len + light_seq_len

        # Get the attributes from a protein and cut off zero padding
        heavy_prim = self.h5file['heavy_chain_primary'][index, :heavy_seq_len]
        light_prim = self.h5file['light_chain_primary'][index, :light_seq_len]
        h_seq = num_to_letter(heavy_prim, _aa_dict)[:heavy_seq_len]
        
        # Get CDR loops
        if 'h3_range' in self.h5file:
            h3 = self.h5file['h3_range'][index]
        
        p0 = AntibodyTypeInteractingProtein(index,
                                            'ab',
                                            id_,
                                            heavy_prim,
                                            light_prim
                                            )
        
        if self.onehot_prim:
            # Try preprocessing - how large?
            p0.to_one_hot()
            if self.add_delimiters:
                p0.add_delimiters_to_one_hot()
        else:
            p0.prim.unsqueeze_(1)

        # 1. Distance, angle matrix
        try:
            dist_angle_mat = self.h5file['dist_angle_mat'][
                index][:4, :total_seq_len, :total_seq_len]
            dist_angle_mat = torch.Tensor(dist_angle_mat).type(
                dtype=torch.float)
        except Exception:
            raise ValueError('Output matrix not defined')

        len_ab = len(heavy_prim) + len(light_prim)
        #print(id_, len_ab)

        p0.dist_angle_mat = dist_angle_mat[:, :len_ab, :len_ab]
        if 'bk_and_cb_coords' in self.h5file.keys():
            coords = torch.tensor(self.h5file['bk_and_cb_coords'][index][:4, :total_seq_len, :])
        else:
            coords = None
        
        mask_residue_selection = None
        if not self.mask_ab_region is None:
            mask_residue_selection = []
            cdr_indices_dict = p0.cdr_indices_dict
            if p0.cdr_indices_dict == {}:
                cdr_indices_dict = p0.get_fragments_with_cdrs()
            for region in ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']:
                if region == self.mask_ab_region and \
                    (region in cdr_indices_dict):
                    mask_residue_selection = cdr_indices_dict[region]
                    break
            if self.mask_ab_region=='cdrs':
                mask_residue_selection = [t for cdr in cdr_indices_dict 
                                            for t in cdr_indices_dict[cdr]]
        
        if self.mask_ab_indices != None:
            mask_residue_selection = self.mask_ab_indices
        print('Indices ', self.mask_ab_indices)
        sequence_label = \
            p0.mask_sequence(contact_percent=self.masking_rate_max, 
                             non_contact_percent=self.masking_rate_min,
                             subset_selection_indices=mask_residue_selection)
        self.masked_residue_indices = (sequence_label != 20).nonzero().flatten().tolist()

        nfeats, _ = \
            get_node_feats_for_prot(p0,
                                    self.n_pe_nodes,
                                    self.n_pe_edges,
                                    masked=True)

        assert len(coords.shape) == 3
        #print(coords.shape, sequence_label.shape)
        # coords -> Atoms, Res, xyz
        self.num_atoms = coords.shape[0]
        # split coords -> Res, Atoms, xyz
        split_coords = coords.detach().clone().permute(1, 0, 2)
        coords = coords.permute(1, 0, 2).reshape(split_coords.shape[0]*split_coords.shape[1], split_coords.shape[2])
        masked_coords = mask_cb_coords(sequence_label, split_coords)
        # II. adding atom feats
        num_res = nfeats.shape[0]
        nfeats = nfeats.unsqueeze(1).expand(-1, self.num_atoms, -1).reshape(-1, nfeats.shape[-1])
        nfeats_atoms = torch.eye(self.num_atoms).unsqueeze(0).expand(num_res, -1, -1).reshape(-1, self.num_atoms)
        # masking cb atoms of label residues
        
        nfeats = mask_nfeats_cb(sequence_label, nfeats, nfeats_atoms)
        adj_mat = torch.ones((nfeats.shape[0], nfeats.shape[0])).long()
        edges_all = get_inter_intra_edges(nfeats.shape[0], self.num_atoms)
        assert coords is not None
        missing_residues_mask = torch.ones((nfeats.shape[0])).long()
        seq_positions_res = torch.arange(nfeats.shape[0] // self.num_atoms)
        seq_positions = seq_positions_res.repeat_interleave(self.num_atoms)
        #seq_positions_res = p0.fragment_indices
        #seq_positions = seq_positions_res.repeat_interleave(self.num_atoms)
        metadata = {}
        if self.metadata_extra: # should not be true during training
            p0.get_cdr_region_dict_from_chothia_numbering()
            assert p0.cdr_indices_dict != {}
        chain_breaks = [p0.heavy_len]
        metadata = {'cdrs': p0.cdr_indices_dict, 'Ab_len': p0.seq_len,
                            'Ab_seq':p0.prim, 'h_len': p0.heavy_len, 'chain_breaks':chain_breaks}
        
        assert len(coords.shape) == 2
        return p0.id, nfeats, masked_coords, edges_all, missing_residues_mask, \
                sequence_label, seq_positions, metadata


    def debug_coord_masking(self, id, masked_coords, coords):
        import matplotlib.pyplot as plt
        #res selction
        if not self.masked_residue_indices is None:
            indices = self.masked_residue_indices
            c_split = masked_coords.reshape(masked_coords.shape[0] // self.num_atoms, self.num_atoms, 3).numpy()
            c_split_label = coords.reshape(coords.shape[0] // self.num_atoms, self.num_atoms, 3).numpy()
            plt.scatter(c_split[indices, 0, 0], c_split[indices,0,1], c='red')
            plt.scatter(c_split_label[indices, 0, 0], c_split_label[indices,0,1], c='green')
            plt.savefig('coords_{}_masked_and_labels.png'.format(id))
            plt.close()


    def get_valid_indices(self):
        """Gets indices with proteins less than the max sequence length
        :return: A list of indices with sequence lengths less than the max
                 sequence length.
        :rtype: list
        """
        n_seqs = self.h5file['heavy_chain_seq_len'].shape[0]
        valid_indices = []
        for i in range(n_seqs):
            for i in range(n_seqs):
                p0_len = self.h5file['heavy_chain_seq_len'][i] + \
                        self.h5file['light_chain_seq_len'][i]
                if p0_len < self.max_seq_len and \
                    p0_len > 0:
                    valid_indices.append(i)
        self.num_proteins = len(valid_indices)
        return valid_indices

    def get_sorted_indices(self):
        lengths = []
        n_seqs = self.h5file['heavy_chain_seq_len'].shape[0]
        for i in range(n_seqs):

            id_ = self.h5file['id'][i]
            if len(str(id_)) < 4:
                #print('Invalid id', i, id_)
                continue
            p0_len = self.h5file['heavy_chain_seq_len'][i] + \
                    self.h5file['light_chain_seq_len'][i]
            if p0_len <= 0:
                continue
            lengths.append((i, p0_len))
        import operator
        sorted_lengths = sorted(lengths,
                                key=operator.itemgetter(1),
                                reverse=True)
        return [t[0] for t in sorted_lengths]

    def __len__(self):
        return self.num_proteins

    @staticmethod
    def merge_samples_to_minibatch(samples):
        return SCNProteinMaskedMultiAtomBatchBatched(zip(*samples)).data()

    @staticmethod
    def merge_samples_to_minibatch_with_metadata(samples):
        return SCNProteinMaskedMultiAtomBatchBatched(zip(*samples)).data_with_metadata()



