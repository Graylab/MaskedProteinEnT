import torch
import numpy as np
import os
import torch.utils.data as data
from src.datasets.ppi_graph_dataset_utils \
    import get_edge_features, get_node_feats_for_prot, get_inter_intra_edges
from src.datasets.masking_utils import mask_cb_coords, mask_nfeats_cb
from src.datasets.InteractingProteinDataClass \
    import FragmentedMultiChainInteractingProtein
from src.data.constants import _aa_dict, letter_to_num
import os, sys
import h5py
# TODO: Fix

class ProteinMaskedMultiAtomDatasetBatched(data.Dataset):
    def __init__(
            self,
            filename,
            onehot_prim=True,
            num_bins=36,
            max_seq_len=350,
            clean_up=True,
            contact_dist_threshold=8.0,
            max_mask=0.15,
            gmodel='egnn-trans-ma',
            crop=True,
            atom_mode='backbone_and_cb',
            span=False,
            min_span_length=3,
            max_span_length=12
            ):
        """
        :param filename: The h5 file for the antibody data.
        :param onehot_prim:
            Whether or not to onehot-encode the primary structure data.
        :param num_bins:
            The number of bins to discretize the distance matrix into. If None,
            then the distance matrix remains continuous.
        """
        super(ProteinMaskedMultiAtomDatasetBatched, self).__init__()
        self.h5file = h5py.File(filename, 'r')
        self.onehot_prim = onehot_prim

        self.max_seq_len = max_seq_len
        self.valid_indices = None
        
        self.contact_dist_threshold = contact_dist_threshold
        self.percent_mask = max_mask
        self.gmodel = gmodel
        self.crop = crop
        self.atom_mode = atom_mode

        
    def calculate_distances(self):
        # Convert coords from N*atoms, 3 to N, atoms, 3
        #print('N*Ax3', self.coords.shape)
        split_coords = self.coords.reshape(self.num_atoms, -1, 3)
        distances = torch.zeros(
            (split_coords.shape[1], split_coords.shape[1], self.num_atoms), dtype=float)
        # extract distances for Ca, N, C, Cb
        for icoor in range(self.num_atoms):
            coords = split_coords[icoor, :, :]
            dvec_left = coords.unsqueeze(0)
            dvec_right = coords.unsqueeze(1)
            distances[:, :, icoor] = (dvec_left - dvec_right).norm(dim=-1)
        #print('distances: ', distances.shape)
        return distances.permute(2, 0, 1)

    def __getitem__(self, index):

        id_ = self.h5file['id'][index]
        
        sequence_len = self.h5file['p0_chain_seq_len'][index]
        # Get the attributes from a protein and cut off zero padding
        prim = self.h5file['p0_chain_primary'][index, :sequence_len]
        #seq = num_to_letter(prim, _aa_dict)
        #print(id_, seq)
        # 1. Distance, angle matrix
        try:
            dist_angle_mat = self.h5file['dist_angle_mat'][
                index][:4, :sequence_len, :sequence_len]
            dist_angle_mat = torch.Tensor(dist_angle_mat).type(
                dtype=torch.float)
        except Exception:
            raise ValueError('Output matrix not defined')
        
        if 'bk_and_cb_coords' in self.h5file.keys():
            coords = torch.tensor(self.h5file['bk_and_cb_coords'][index][:4, :sequence_len, :])
        else:
            coords = None

        p0_frag_positions = \
            torch.tensor((
                self.h5file['p0_frag_positions'][index,
                                :sequence_len]).astype(int)).long()
        
        #Crop larger proteins - if cropping enabled
        sel_seq_start, sel_seq_end = 0, sequence_len
        if sequence_len > self.max_seq_len:
            diff = sequence_len - self.max_seq_len
            sel_seq_start = np.random.randint(0, diff)
            sel_seq_end = sel_seq_start + self.max_seq_len
            sequence_len = sel_seq_end - sel_seq_start
            dist_angle_mat = dist_angle_mat[:, sel_seq_start:sel_seq_end, sel_seq_start:sel_seq_end]
            coords = coords[:, sel_seq_start:sel_seq_end, :]
            p0_frag_positions = p0_frag_positions[sel_seq_start:sel_seq_end]
            if not per_res_prop is None:
                per_res_prop = per_res_prop[sel_seq_start:sel_seq_end]
        
        # relative numbering is ok
        p0 = FragmentedMultiChainInteractingProtein(
            index,
            'protein',
            id_,
            contact_indices=torch.tensor([t for t in range(sequence_len)]).long(),
            fragment_indices=torch.tensor([t for t in range(sequence_len)]).long(),
            prim=prim,
            dist_angle_mat=dist_angle_mat)
        p0.to_one_hot()

        # Appropriating the object for a distance matrix with any dihedrals
        p0.dist_angle_mat_inputs = p0.dist_angle_mat

        sequence_masked_label_p0 = \
            p0.mask_sequence(contact_percent=self.percent_mask,
                             pdf_flip=self.pdf_flip, pdf_flip_index=self.pdf_flip_index)
        nfeats, _ = \
            get_node_feats_for_prot(p0,
                                    masked=True)
        sequence_label = sequence_masked_label_p0

        num_res = nfeats.shape[0]
        coords = coords.permute(1, 0, 2)
        coords = mask_cb_coords(sequence_label, coords)
        
        nfeats = nfeats.unsqueeze(1).expand(-1, self.num_atoms, -1).reshape(-1, nfeats.shape[-1])
        nfeats_atoms = torch.eye(self.num_atoms).unsqueeze(0).expand(num_res, -1, -1).reshape(-1, self.num_atoms)
        nfeats = mask_nfeats_cb(sequence_label, nfeats, nfeats_atoms)
        
        nfeats = nfeats.double()

        edge_indices_0 = \
        torch.tensor([t for t in range(sel_seq_start, sel_seq_end)]).repeat(sequence_len)

        edge_indices_1 = \
        torch.tensor([t for t in range(sel_seq_start, sel_seq_end)]).repeat_interleave(sequence_len)

        edge_indices = torch.stack([edge_indices_0, edge_indices_1],
                                    dim=0).long()
        
        seq_positions_res = torch.arange(nfeats.shape[0] // self.num_atoms)
        seq_positions = seq_positions_res.repeat_interleave(self.num_atoms)    
        assert coords is not None
        missing_residues_mask = torch.ones((nfeats.shape[0])).long()
        metadata = {}
        if self.gmodel in ['egnn-trans-ma','egnn-trans-ma-ppi']:
            edges_all = get_inter_intra_edges(nfeats.shape[0], self.num_atoms)
            if not per_res_prop is None:
                metadata = {per_res_prop_key:per_res_prop}
            return p0.id, nfeats, coords, edges_all, missing_residues_mask, \
                sequence_label, seq_positions, metadata
        else:
            print('Model not found.')
            sys.exit()


    def get_valid_indices(self):
        """Gets indices with proteins less than the max sequence length
        :return: A list of indices with sequence lengths less than the max
                 sequence length.
        :rtype: list
        """
        n_seqs = self.h5file['seq_len'].shape[0]
        valid_indices = []
        for i in range(n_seqs):
            for i in range(n_seqs):
                p0_len = self.h5file['seq_len'][i]
                if p0_len > 0:
                   valid_indices.append(i)
        self.num_proteins = len(valid_indices)
        return valid_indices

    def get_sorted_indices(self):
        lengths = []
        n_seqs = self.h5file['p0_chain_seq_len'].shape[0]
        for i in range(n_seqs):

            id_ = self.h5file['id'][i]
            if len(str(id_)) < 4:
                continue
            p0_len = self.h5file['p0_chain_seq_len'][i]
            if p0_len <= 0:
                continue
            lengths.append((i, p0_len))
        import operator
        sorted_lengths = sorted(lengths,
                                key=operator.itemgetter(1),
                                reverse=True)
        return [t[0] for t in sorted_lengths]
    

    def __len__(self):
        print(self.h5file.keys())
        
        self.num_proteins = self.h5file['p0_chain_seq_len'].shape[0]
        return self.num_proteins

    @staticmethod
    def merge_samples_to_minibatch(samples):
        return ProteinMaskedMultiAtomBatchBatched(zip(*samples)).data()

    @staticmethod
    def merge_samples_to_minibatch_with_metadata(samples):
        return ProteinMaskedMultiAtomBatchBatched(zip(*samples)).data_with_metadata()


class ProteinMaskedMultiAtomBatchBatched:
    def __init__(self, batch_data):
        (self.id_, self.feats, self.coords, self.edges, self.mask,
         self.seqlabel, self.seq_positions, self.metadata) = batch_data

    def data(self):
        return self.id_, self.features()

    def data_with_metadata(self):
        return self.id_, self.features(), self.metadata

    def features(self):
        batched_tens = {}
        batched_tens['feats'] = self.feats
        batched_tens['coords'] = self.coords
        batched_tens['edges'] = self.edges
        batched_tens['mask'] = self.mask
        batched_tens['seq_label'] = self.seqlabel
        batched_tens['seq_pos'] = self.seq_positions
        
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
            elif key == 'prop_mask':
                # 1D
                max_n_labels = max([t.shape[0] for t in batched_tens[key]])
                padded_tens = [
                    torch.cat([t, torch.zeros((max_n_labels - t.shape[0]))], dim=0)
                    for t in batched_tens[key]
                ]
            elif key == 'edges':
                                # 2D
                #print(key, [t.shape for t in batched_tens[key]], max_n)
                # 2D requires padding in 2 dims
                if len(batched_tens[key][0].shape) == 2:
                    padded_tens = [
                    torch.cat(
                        [t, torch.zeros((max_n - t.shape[0], t.shape[1]))],
                        dim=0) for t in batched_tens[key]
                    ]
                    padded_tens = [
                        torch.cat(
                            [t,
                            torch.zeros((t.shape[0], max_n - t.shape[1]))],
                            dim=1) for t in padded_tens
                    ]
                elif len(batched_tens[key][0].shape) == 3:
                    padded_tens = [
                    torch.cat(
                        [t, torch.zeros((max_n - t.shape[0], t.shape[1], t.shape[2]))],
                        dim=0) for t in batched_tens[key]
                    ]
                    padded_tens = [
                        torch.cat(
                            [t,
                            torch.zeros((t.shape[0], max_n - t.shape[1], t.shape[2]))],
                            dim=1) for t in padded_tens
                    ]
                else:
                    sys.exit('Shape for edges tensor {} not supported'.format(batched_tens[0].shape))
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
                #print('batched', key, batched_tens[key].shape, '\n')
        #print('batched ', batched_tens['feats'].shape, batched_tens['seq_label'].shape)
        return batched_tens['feats'].long(), batched_tens['coords'].double(),\
                batched_tens['edges'].long(), batched_tens['mask'].bool(),\
                batched_tens['seq_label'].long(), batched_tens['seq_pos'].long()
