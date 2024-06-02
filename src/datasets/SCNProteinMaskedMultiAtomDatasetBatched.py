import torch
import numpy as np
import os, sys
from sidechainnet.dataloaders.SCNDataset import SCNProtein
import torch.utils.data as data
from src.datasets.masking_utils import mask_cb_coords, mask_nfeats_cb, \
    mask_spans
from src.datasets.ppi_graph_dataset_utils \
    import get_inter_intra_edges
from src.datasets.InteractingProteinDataClass \
    import FragmentedMultiChainInteractingProtein
from src.data.constants import _aa_dict, letter_to_num

class SCNProteinMaskedMultiAtomDatasetBatched(data.Dataset):
    def __init__(
            self,
            scn_dataset,
            scn_dataset_keys,
            selected_ids_file='',
            onehot_prim=True,
            max_seq_len=350,
            contact_dist_threshold=8.0,
            max_mask=0.15,
            gmodel='egnn-trans-ma',
            crop=True,
            atom_mode='backbone_and_cb',
            span=False,
            min_span_length=3,
            max_span_length=12,
            ):
        """
        Gets data from sidechainnet; for subset of ids, provide file with selected_ids_file
        """
        super(SCNProteinMaskedMultiAtomDatasetBatched, self).__init__()

        self.onehot_prim = onehot_prim

        self.max_seq_len = max_seq_len
        self.valid_indices = None
        self.dataset_keys = scn_dataset_keys
        
        self.contact_dist_threshold = contact_dist_threshold
        # masking
        self.percent_mask = max_mask
        self.gmodel = gmodel
        self.crop = crop
        self.atom_mode = atom_mode
        self.span_mask = span
        self.min_frag = min_span_length
        self.max_frag = max_span_length
        
        if selected_ids_file != '':
            with open(selected_ids_file, 'r') as f:
               selected_ids_list = [t.split()[0] for t in f.readlines()]
               print(len(selected_ids_list))

        self.ids_to_SCNProtein = {}
        self.idx_to_SCNProtein = {}
        idx = 0
        for key in self.dataset_keys:
            d = scn_dataset[key]
            for c, a, s, u, m, e, n, r, z, i in zip(d['crd'], d['ang'],
                                                    d['seq'], d['ums'],
                                                    d['msk'], d['evo'],
                                                    d['sec'], d['res'],
                                                    d['mod'], d['ids']):
                if selected_ids_file != '':
                    if i not in selected_ids_list:
                        #print('Not using {} ... '.format(i))
                        continue
                p = SCNProtein(coordinates=c,
                               angles=a,
                               sequence=s,
                               unmodified_seq=u,
                               mask=m,
                               evolutionary=e,
                               secondary_structure=n,
                               resolution=r,
                               is_modified=z,
                               id=i,
                               split='')
                # For testing
                #if len(m) < 350:
                #    continue
                #if (not self.is_valid(p, crop=self.crop)):
                #    continue
                self.ids_to_SCNProtein[i] = p
                self.idx_to_SCNProtein[idx] = p
                idx += 1
        self.num_proteins = idx
        print("Dataset size: ", self.num_proteins)

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
        return distances.permute(2, 0, 1)

    def get_coords(self, scn_p0, sel_seq_start, sel_seq_end):
        split_coords = torch.tensor(scn_p0.coords.reshape(-1, 14, 3))
        #print('NxAx3: ', split_coords.shape)
        # 0, 1, 2: N, Ca, C
        # 3: backbone o
        # 4: cb
        coords = split_coords[sel_seq_start:sel_seq_end, [0, 1, 2, 4], :]
        num_coords = coords.shape[1]*coords.shape[0]
        self.num_atoms = coords.shape[1]
        self.coords = coords.reshape(num_coords, 3)

    def crop_sequences(self, scn_p0, missing_res_mask, len_non_missing_indices):
        sequence_len = len(scn_p0.mask)
        while((sequence_len > self.max_seq_len) or (len_non_missing_indices < 10)):
            diff = len(scn_p0.mask) - self.max_seq_len
            sel_seq_start = np.random.randint(0, diff)
            sel_seq_end = sel_seq_start + self.max_seq_len
            sequence_len = sel_seq_end - sel_seq_start
            missing_res_mask = scn_p0.mask[sel_seq_start:sel_seq_end]
            non_missing_indices = torch.tensor(
            [i for i, t in enumerate(missing_res_mask) if t == '+']).long()
            len_non_missing_indices = non_missing_indices.shape[0]
            print(len_non_missing_indices, sequence_len)
        return sel_seq_start, sel_seq_end

    def __getitem__(self, index):

        #print(index, self.idx_to_SCNProtein[index])
        scn_p0 = self.idx_to_SCNProtein[index]
        sequence_len = len(scn_p0.mask)
        sel_seq_start, sel_seq_end = 0, sequence_len
        if sequence_len > self.max_seq_len:
            diff = sequence_len - self.max_seq_len
            sel_seq_start = np.random.randint(0, diff)
            sel_seq_end = sel_seq_start + self.max_seq_len
        
        missing_res_mask = scn_p0.mask[sel_seq_start:sel_seq_end]
        non_missing_indices = torch.tensor(
            [i for i, t in enumerate(missing_res_mask) if t == '+']).long()
        
        sequence_len = sel_seq_end - sel_seq_start
        self.get_coords(scn_p0, sel_seq_start, sel_seq_end, mode=self.atom_mode)
        distances = self.calculate_distances()
        seq = scn_p0.seq[sel_seq_start:sel_seq_end]
        missing_res_mask = scn_p0.mask[sel_seq_start:sel_seq_end]

        # relative numbering is ok - should prevent overfitting
        non_missing_indices = torch.tensor(
            [i for i, t in enumerate(missing_res_mask) if t == '+']).long()

        # relative numbering is ok
        res_positions = torch.tensor([t for t in range(sequence_len)]).long()
        #print(index, len(seq))
        p0 = FragmentedMultiChainInteractingProtein(
            index,
            'protein',
            scn_p0.id,
            contact_indices=non_missing_indices,
            fragment_indices=res_positions,
            prim=letter_to_num(seq, _aa_dict)
            )
        p0.to_one_hot()

        if self.span_mask:
            span_indices = mask_spans(sequence_len,
                                      np.random.random_integers(self.min_frag,
                                                         self.max_frag, 1)[0])
            sequence_label = \
            p0.mask_sequence(contact_percent=1.0,
                            subset_selection_indices=span_indices,
                            select_intersection=True) #intersection ensures non-missing residues
        else:
            sequence_label = \
            p0.mask_sequence(contact_percent=self.percent_mask,
                             pdf_flip=self.pdf_flip, pdf_flip_index=self.pdf_flip_index)
        
        nfeats = p0.prim_masked.float()
        
        if self.gmodel in ['egnn-trans-ma']:
            split_coords = self.coords.reshape(-1, self.num_atoms, 3)
            self.coords = mask_cb_coords(sequence_label, split_coords)
            num_res = nfeats.shape[0]
            nfeats = nfeats.unsqueeze(1).expand(-1, self.num_atoms, -1).reshape(-1, nfeats.shape[-1])
            nfeats_atoms = torch.eye(self.num_atoms).unsqueeze(0).expand(num_res, -1, -1).reshape(-1, self.num_atoms)
            nfeats = mask_nfeats_cb(sequence_label, nfeats, nfeats_atoms)
            
        else:
            sys.exit('Model not implemented')
            
        nfeats = nfeats.double()

        edge_indices_0 = \
        torch.tensor([t for t in range(sel_seq_start, sel_seq_end)]).repeat(sequence_len)

        edge_indices_1 = \
        torch.tensor([t for t in range(sel_seq_start, sel_seq_end)]).repeat_interleave(sequence_len)
        
        seq_positions_res = torch.arange(nfeats.shape[0] // self.num_atoms)
        seq_positions = seq_positions_res.repeat_interleave(self.num_atoms)
        metadata = {}
        metadata['sec'] = scn_p0.secondary_structure
        missing_residues_mask = torch.tensor(
            [1 if t == '+' else 0 for t in missing_res_mask]).long()
        missing_residues_mask = missing_residues_mask.unsqueeze(1).expand(-1, self.num_atoms).reshape(num_res * self.num_atoms)
        if self.gmodel in ['egnn-trans-ma',
                           'egnn-trans-ma-ppi']:
            edges_all = get_inter_intra_edges(nfeats.shape[0], self.num_atoms)
            return p0.id, nfeats, self.coords, edges_all, missing_residues_mask, \
                sequence_label, seq_positions, metadata

    def is_valid(self, p, crop=True):
        p_len = len(p.mask)
        missing = p.mask.count('-') if isinstance(p.mask.count('-'),
                                                  int) else 0
        frac_missing = missing / float(p_len)
        bvalid = (frac_missing <= 0.50)
        if not crop:
            bvalid = (p_len <= self.max_seq_len)
        return bvalid

    def get_valid_indices(self):
        """Gets indices with proteins less than the max sequence length
        :return: A list of indices with sequence lengths less than the max
                 sequence length.
        :rtype: list
        """
        valid_indices = []
        n_seqs = self.num_proteins

        for i in range(n_seqs):
            scn_p0 = self.idx_to_SCNProtein[i]
            if self.is_valid(scn_p0):
                valid_indices.append(i)
        return valid_indices

    def get_sorted_indices(self):
        pass

    def __len__(self):
        return self.num_proteins

    @staticmethod
    def merge_samples_to_minibatch(samples):
        return SCNProteinMaskedMultiAtomBatchBatched(zip(*samples)).data()

    @staticmethod
    def merge_samples_to_minibatch_with_metadata(samples):
        return SCNProteinMaskedMultiAtomBatchBatched(zip(*samples)).data_with_metadata()

class SCNProteinMaskedMultiAtomBatchBatched:
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