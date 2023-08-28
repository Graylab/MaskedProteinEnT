from src.data.ppi_graph_dataset_utils import select_contact_indices
                                                    
from src.data.utils.pdb import cdr_indices_from_chothia_numbering, renumber_seq
import torch
import math
import numpy as np


def get_chothia_numbering(ab_seq, h_len):
    h_renum = renumber_seq(ab_seq[:h_len], scheme="-c")
    h_cnum = [t.split()[0].replace('H','H') for t in h_renum.split('\n') if t.rstrip() != '']
    l_renum = renumber_seq(ab_seq[h_len:], scheme="-c")
    l_cnum = [t.split()[0].replace('L','L') for t in l_renum.split('\n') if t.rstrip() != '']
    ab_chothia_pos = h_cnum + l_cnum
    return ab_chothia_pos

class MultiChainInteractingProtein():
    def __init__(self,
                 index,
                 type,
                 id,
                 prim=None,
                 contact_indices=None,
                 dist_angle_mat=None,
                 phi_psi_mat=None):
        super().__init__()

        self.index = index
        self.type = type
        self.id = id
        if prim is None:
            self.prim = None
            self.seq_len = 0
        else:
            self.prim = torch.Tensor(prim).type(dtype=torch.uint8)
            self.seq_len = self.prim.shape[0]
        self.contact_indices = contact_indices
        self.contact_indices_threshold = None
        self.dist_angle_mat = dist_angle_mat
        self.phi_psi_mat = phi_psi_mat
        self.edges_topk = None
        self.dist_angle_mat_inputs = None
        self.dist_angle_mat_labels = None

    def to_one_hot(self):
        self.prim = torch.nn.functional.one_hot(self.prim.long(),
                                                num_classes=20)
    
    def mask_non_contact_indices(self, mask):
        noncontact_indices = [
            t for t in range(self.seq_len) if t not in self.contact_indices
        ]
        mask.scatter_(0, torch.tensor(noncontact_indices).long(), torch.zeros((mask.shape)))
        return mask


    def mask_sequence(
            self,
            contact_percent=0.15,
            non_contact_percent=0.05,
            ignore_index=20,  # 19+1
            debug=False,
            pdf_flip=None,
            pdf_flip_index=None,
            subset_selection_indices=None,
            select_intersection=False,
            corrupt=False):

        prim = self.prim.detach().clone()

        mask = torch.ones((prim.shape[0]))
        num_cont_indices = len(self.contact_indices)
        
        if subset_selection_indices is None:
            num_masked_res = int(num_cont_indices * contact_percent)
            sel_cont_indices = \
                torch.tensor(np.random.choice(self.contact_indices, size=num_masked_res,
                                replace=False))
            num_noncont_indices = self.seq_len - len(self.contact_indices)
            num_masked_noncont = int(num_noncont_indices * non_contact_percent)
            noncontact_indices = [
            t for t in range(self.seq_len) if t not in self.contact_indices
            ]
        else:
            if select_intersection:
                selected_indices = [t for t in list(self.contact_indices) if t in subset_selection_indices]
                #print(selected_indices)
            else:
                selected_indices = subset_selection_indices
            
            num_masked_res = int(len(selected_indices) * contact_percent)
            sel_cont_indices = \
                torch.tensor(np.random.choice(selected_indices, size=num_masked_res,
                                replace=False))
            num_noncont_indices = self.seq_len - len(subset_selection_indices)
            num_masked_noncont = int(num_noncont_indices * non_contact_percent)
            noncontact_indices = [
            t for t in range(self.seq_len) if t not in subset_selection_indices
            ]

        #print(sel_cont_indices)
        if num_masked_res > 0:
            mask.scatter_(0, sel_cont_indices, torch.zeros((mask.shape)))
            #print('mask 1 ', mask)
            
        if num_masked_noncont > 0:
            sel_noncont_indices = \
            torch.tensor(np.random.choice(noncontact_indices,
                                          size=num_masked_noncont,
                                          replace=False)
                        )
            
            mask.scatter_(0, sel_noncont_indices, torch.zeros((mask.shape)))
        
        #labels are masked prim seq
        sequence_label = prim

        sequence_label_indices = sequence_label.max(dim=-1)[1]
        # flip labels that are being predicted
        #flip before masking -> easier
        if corrupt:
            sequence_label_uncorrupted = sequence_label.max(dim=-1)[1]
            
            # sequence_label_indices <- corrupted labels -> cannot use for label prediction
            sequence_label_indices_oh = torch.nn.functional.one_hot(sequence_label_indices, num_classes=20)
            sequence_label_indices_cor = sequence_label_indices.clone()
            sequence_label_indices_cor[mask == 1] = 0 # correct  
            sequence_label_indices_cor[mask == 0] = 1 # corrupted
            # Switch this back to correct labels for predictions
            sequence_label_indices[mask == 1] = ignore_index  #do not predict unmaksed labels
            #print('corruption ', sequence_label_indices[mask == 0])
            sequence_label_indices[mask == 0] =  sequence_label_uncorrupted[mask == 0] #correct corrupted labels
            #print('correct ', sequence_label_indices[mask == 0])

            mask = mask.unsqueeze_(1).expand(-1, self.prim.shape[-1])
            self.prim_masked = prim
            # corrupt primary sequence
            self.prim_masked[mask == 0] = sequence_label_indices_oh[mask==0]
            #print('Corrupted')
            #print(sequence_label_indices_cor.sum(), sequence_label_indices.sum())
            assert sequence_label_indices.sum() != sequence_label_indices_cor.sum()

            return sequence_label_indices, sequence_label_indices_cor


        

        sequence_label_indices[
            mask == 1] = ignore_index  #do not predict unmaksed labels

        mask = mask.unsqueeze_(1).expand(-1, self.prim.shape[-1])
        self.prim_masked = prim
        self.prim_masked[mask ==
                         0] = 0  #masked residues are not seen by the model

        if debug:
            self.plot_masked_sequences(sequence_label,
                                       sequence_label_indices,
                                       mask,
                                       mpercent='c{}_nc{}'.format(
                                           contact_percent,
                                           non_contact_percent))
        #if contact_percent > 0:
        #    print('sequence_label_indices: ', sequence_label_indices)
        return sequence_label_indices

    def plot_masked_sequences(self,
                              sequence_label,
                              sequence_label_indices,
                              mask,
                              mpercent=0):

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, squeeze=False)
        fig.suptitle('mask percentage = {}'.format(mpercent))
        outname = '{}_masked_sequence_{}.png'.format(
            str(self.id)[2:6], self.type)
        print(outname)
        ax = axes[0, 0]
        im = ax.imshow(self.prim, aspect='equal')
        ax.set_title('prim')
        fig.colorbar(im, ax=ax)

        ax = axes[0, 1]
        im = ax.imshow(mask, aspect='equal')
        ax.set_title('mask')
        fig.colorbar(im, ax=ax)

        ax = axes[0, 2]
        im = ax.imshow(self.prim_masked, aspect='equal')
        ax.set_title('prim_masked')
        fig.colorbar(im, ax=ax)

        ax = axes[1, 0]
        im = ax.imshow(sequence_label, vmin=0, aspect='equal')
        ax.set_title('seq_label')
        fig.colorbar(im, ax=ax)

        sequence_label_indices_plt = sequence_label_indices.unsqueeze(
            1).expand(-1, 20)
        ax = axes[1, 1]
        im = ax.imshow(sequence_label_indices_plt, vmin=0, aspect='equal')
        ax.set_title('seq_label_ind')
        fig.colorbar(im, ax=ax)

        ax = axes[1, 2]
        im = ax.imshow(sequence_label_indices_plt, aspect='equal', vmax=-1)
        ax.set_title('seq_label_ind_ignore')
        fig.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(outname, transparent=True, dpi=600)
        #plt.show()
        plt.close()


class AntibodyTypeInteractingProtein(MultiChainInteractingProtein):
    def __init__(self,
                 index,
                 type,
                 id,
                 heavy_prim,
                 light_prim,
                 cdrs=[],
                 cdr_names=['h1', 'h2', 'h3', 'l1', 'l2', 'l3'],
                 contact_indices=None,
                 dist_angle_mat=None,
                 phi_psi_mat=None,
                 fragment_indices=None):

        super().__init__(index,
                         type,
                         id,
                         contact_indices=contact_indices,
                         dist_angle_mat=dist_angle_mat,
                         phi_psi_mat=phi_psi_mat)

        self.heavy_prim = torch.Tensor(heavy_prim).type(dtype=torch.uint8)
        self.light_prim = torch.Tensor(light_prim).type(dtype=torch.uint8)
        self.prim = torch.cat([self.heavy_prim, self.light_prim], dim=0)
        self.seq_len = self.prim.shape[0]
        self.heavy_len = self.heavy_prim.shape[0]
        self.cdrs = cdrs
        self.cdr_names = cdr_names
        
        self.cdr_indices_dict = {}
        self.loop_indices = None
        self.contact_indices = None
        if self.cdrs != []:
            self.set_loop_indices()
            self.contact_indices = self.loop_indices
        if self.contact_indices is None:
            self.contact_indices = torch.arange(self.seq_len).long()
        #fragment indices should only be used for positional embeddings
        #offset light chain indices by 100
        if fragment_indices == None:
            self.fragment_indices = torch.tensor([t 
                                                  if t < self.heavy_len
                                                  else t+self.heavy_len+100
                                                  for t in range(self.seq_len)]).long()
            #self.fragment_indics = torch.tensor([t for t in range(self.seq_len)]).long()
        else:
            self.fragment_indices = fragment_indices

    def set_loop_indices(self):
        loop_indices = []
        for cdr_n, cdr in zip(self.cdr_names, self.cdrs):
            if (cdr[1] - cdr[0]) > 0:
                self.cdr_indices_dict[cdr_n] = [t for t in range(cdr[0], cdr[1] + 1)]
                loop_indices += self.cdr_indices_dict[cdr_n]
        self.loop_indices = torch.Tensor(loop_indices).long()


    def get_fragments_with_cdrs(self):
        cdrs = {'h1': [t for t in range(23, 35)],
                'h2': [t for t in range(49, 60)],
                'h3':[t for t in range(93, self.heavy_len - 7)]}
        #print(self.heavy_prim.shape[0], self.light_prim.shape[0])
        if self.light_prim.shape[0] != 0:
            cdrs.update({'l1': [t+self.heavy_len for t in range(23, 35)], 
                    'l2': [t+self.heavy_len for t in range(49, 60)],
                    'l3':[t+self.heavy_len for t in range(88, min(103, self.light_prim.shape[0] - 3))]})
        
        return cdrs

    def to_one_hot(self):

        self.heavy_prim = torch.nn.functional.one_hot(self.heavy_prim.long(),
                                                      num_classes=20)
        self.light_prim = torch.nn.functional.one_hot(self.light_prim.long(),
                                                      num_classes=20)
        self.prim = torch.nn.functional.one_hot(self.prim.long(),
                                                num_classes=20)


    def set_contact_indices(self, subset_contact):
        # For antibody need to calculate subset of contact indices
        # from loop indices
        self.contact_indices = subset_contact

    def get_cdr_region_dict_from_chothia_numbering(self):
        ab_seq = torch.argmax(self.prim, dim=-1)
        ab_chothia_pos = get_chothia_numbering(ab_seq, len(self.heavy_prim))
        l_len = self.seq_len - self.heavy_len
        region_dict = {}
        region_dict_ch = {}
        for cdr in ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']:
            chain_id = cdr[0].upper()
            if l_len < 1 and chain_id=='L':
                continue
            ab_chothia_pos_ind = [int(t[1:]) if t[1:].isdigit() else int(t[1:-1]) 
                                    for t in ab_chothia_pos]
            cdr_start, cdr_end = cdr_indices_from_chothia_numbering(ab_chothia_pos_ind,
                                                                    cdr, self.heavy_len,
                                                                    chain_id)
            region_dict[cdr] = [t for t in range(cdr_start, cdr_end+1)]
            region_dict_ch[cdr] = [ab_chothia_pos[t] for t in range(cdr_start, cdr_end+1)]
        self.cdr_indices_dict = region_dict
        return region_dict, region_dict_ch


class FragmentedMultiChainInteractingProtein(MultiChainInteractingProtein):
    def __init__(self,
                 index,
                 type,
                 id,
                 prim,
                 contact_indices=None,
                 fragment_indices=None,
                 n_terms=None,
                 c_terms=None,
                 dist_angle_mat=None,
                 phi_psi_mat=None):

        super().__init__(index,
                         type,
                         id,
                         prim=prim,
                         contact_indices=contact_indices,
                         dist_angle_mat=dist_angle_mat,
                         phi_psi_mat=dist_angle_mat)

        self.fragment_indices = fragment_indices
        self.n_terms = n_terms
        self.c_terms = c_terms
        self.average_nn_coords = None


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


class InteractingPartners():
    def __init__(self, p0, p1, dist_angle_mat):
        super().__init__()

        self.p0 = p0
        self.p1 = p1
        self.dist_angle_mat_full = dist_angle_mat

        len_p0 = len(p0.prim)
        self.dist_angle_mat_int = dist_angle_mat[:, :len_p0, len_p0:]
        self.dist_angle_mat_int_asym = dist_angle_mat[:, len_p0:, :len_p0]
        self.dist_angle_mat_int_p0_p1_select = None
        self.dist_angle_mat_int_p0_p1_asym_select = None
        self.pairs_mat = None
        self.paratope = None
        self.epitope = None
        self.dist_angle_mat_full_topk_select = None

    def filter_dist_angle_mat_by_contact_indices(self):
        # PREPROCESS
        self.dist_angle_mat_int_p0_p1_select,\
            self.dist_angle_mat_int_p0_p1_asym_select =\
            select_contact_indices(self.dist_angle_mat_int,
                                   self.dist_angle_mat_int_asym,
                                   self.p0.contact_indices,
                                   self.p1.contact_indices)

    def setup_paratope_epitope(self, contact_dist_threshold):

        contact_dist = self.dist_angle_mat_int[0]

        contact_nodes = \
            torch.where(contact_dist <=
                        contact_dist_threshold,
                        torch.ones(contact_dist.size()).long(),
                        torch.zeros(contact_dist.size()).long())
        
        #still keeping the analogy to ab-ag binding
        self.paratope = torch.max(contact_nodes, 1)[0]
        self.epitope = torch.max(contact_nodes, 0)[0]
        
    