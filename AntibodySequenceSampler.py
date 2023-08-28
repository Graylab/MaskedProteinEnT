import os
import json
import glob
import torch
import numpy as np
import pandas as pd
from utils.metrics \
    import get_recovery_metrics_for_batch
from src.model.ProteinMaskedLabelModel_EnT_MA import ProteinMaskedLabelModel_EnT_MA
from src.data.constants import num_to_letter, _aa_dict, letter_to_num
from utils.prepare_model_inputs_from_pdb \
                import get_ppi_info_from_pdb_file, get_abag_info_from_pdb_file
from utils.command_line_utils import _get_args
import sys

from tqdm import tqdm

from utils.utils_plotting \
    import plot_seq_logo, plot_histogram_for_array,\
    sequences_to_probabilities

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)


def get_cleanid_from_numpy_string(id):
    if type(id) is str:
        return id
    cleanid= str(id)[2:-1]
    return cleanid

def get_multimer_seq_from_array(seq, p0_len, chain_breaks = [], insert_char=':',
                                include_Ag=False):
    letter_seq_full = num_to_letter(seq, _aa_dict)
    letter_seq = letter_seq_full
    if not include_Ag:
        letter_seq = letter_seq_full[:p0_len]
    
    if chain_breaks == []:
        return letter_seq[:p0_len] + insert_char + letter_seq[p0_len:] + '\n'
    else:
        print(chain_breaks)
        seq_list = [t for t in letter_seq]
        chain_breaks.reverse()
        for position in chain_breaks:
            if position < len(seq_list):
                seq_list.insert(position, ':')
        return ''.join(seq_list)+'\n'
        

def get_seq_from_array(seq, p0_len, insert_char='', partner_selection='Ab'):
    letter_seq = num_to_letter(seq, _aa_dict)
    if partner_selection == 'Ab':
        return letter_seq[:p0_len] + '\n'
    elif partner_selection == 'Ag':
        return letter_seq[p0_len:] + '\n'
    else:
        return letter_seq[:p0_len] + insert_char + letter_seq[p0_len:] + '\n'


def fast_ld_calculation(seqs, ref_seqs):
    int_dseqs = np.array([letter_to_num(t, _aa_dict) for t in seqs])
    int_seqs = np.array([letter_to_num(t, _aa_dict) for t in ref_seqs])
    print(int_seqs.shape, int_dseqs.shape)
    overlap_seqs = [np.logical_not(np.equal(int_seqs, int_dseqs[i, :])).astype(int) \
            for i in range(int_dseqs.shape[0])]
    ld =[np.sum(t, axis=1) for t in overlap_seqs]
    ld_min_ref = [np.amin(t) for t in ld]
    return ld, ld_min_ref


def fast_ld_calculation_self(seqs):
    int_dseqs = np.array([letter_to_num(t, _aa_dict) for t in seqs])
    print(int_dseqs.shape)
    ld = []
    min_size = 1000
    index_splits = [t for t in range(min_size, len(seqs), min_size)]
    print(index_splits)
    batched_array = np.split(int_dseqs, index_splits)
    print(len(batched_array))
    for batch in batched_array:
        print(batch.shape)
        overlap_seqs = [np.logical_not(np.equal(np.delete(batch, i, axis=0),
                                                batch[i, :])).astype(int) \
                for i in range(batch.shape[0])]
        ld += [np.sum(t, axis=1).tolist() for t in overlap_seqs]
    from itertools import chain
    ld = list(chain.from_iterable(ld))

    ld_min_ref = [np.amin(t) for t in ld]
    return ld, ld_min_ref


def plot_sequence_metrics(sequences, wt_seq=None, indices=[], outfile_pattern='./{}.png'):
    seq_probs = sequences_to_probabilities(sequences)
    plot_seq_logo(seq_probs, indices=indices, wt_seq=wt_seq, outfile=outfile_pattern.format('logo'))
    print('Written logo')
    ld, ld_min = fast_ld_calculation_self(sequences)
    plot_histogram_for_array(ld, outfile_pattern.format('ldself'))
    print('Written LD')


class AntibodyAntigenSequenceSampler():
    def __init__(self,
                mr=1.0
                ):
        super().__init__()

        self.args = _get_args()
        self.gmodel = self.args.protein_gmodel
        self.model = ProteinMaskedLabelModel_EnT_MA.load_from_checkpoint(self.args.model).to(device)
        self.model.freeze()
        
        self.write_sequences = False
        self.args.train_split = 0
        self.args.shuffle_dataset=False
        self.args.masking_rate_max = mr
        self.outdir = self.args.output_dir
        os.makedirs(self.outdir, exist_ok=True)


    def get_dataloader(self, partner_selection='Ab',
                       region_selection=None,
                       intersect_with_contacts=False,
                       mask_indices=None,
                       subset_ids=[]):
        self.partner_selection = partner_selection
        self.region_selection = region_selection
        self.intersect_with_contacts = intersect_with_contacts
        
        mr_min = 0.0
        if partner_selection == 'both':
            mr_min = self.args.masking_rate_max

        if self.args.from_pdb == '':
            self.d_loader = get_dataloader_for_testing(mr=self.args.masking_rate_max,
                                    val_split=(1-self.args.train_split),
                                    partner_selection=partner_selection,
                                    with_metadata=True,
                                    region_selection=region_selection,
                                    mask_indices=mask_indices,
                                    intersect_with_contacts=intersect_with_contacts,
                                    mr_min=mr_min)
        else:
            assert os.path.exists(self.args.from_pdb)
            assert os.path.exists(self.args.partners_json)

            self.args.masking_rate_min = 0.0
            if partner_selection == 'both':
                mr_p1 = self.args.masking_rate_max
                mr_p0 = self.args.masking_rate_max
            elif partner_selection == 'Ab':
                mr_p0 = self.args.masking_rate_max
                mr_p1 = self.args.masking_rate_min
            elif partner_selection == 'Ag':
                mr_p0 = self.args.masking_rate_min
                mr_p1 = self.args.masking_rate_max
            else:
                print(f'{partner_selection} not supported')
                sys.exit()
                
            ppi_partners = json.load(open(self.args.partners_json, 'r'))
            if os.path.isdir(self.args.from_pdb):
                pdb_files = glob.glob(self.args.from_pdb + '.pdb')
                dirname = self.args.from_pdb
            else:
                pdb_files = [self.args.from_pdb]
                dirname = os.path.dirname(self.args.from_pdb)
            
            self.d_loader = []
            print(pdb_files)
            for pdbid in ppi_partners:
                partners = ppi_partners[pdbid].split('_')
                pdb_file = glob.glob(f'{dirname}/{pdbid.lower()}_*.pdb')
                print(pdb_file)
                if len(pdb_file)>0:
                    pdb_file = pdb_file[0]
                else:
                    continue
                args.mask_ab_region = None if args.mask_ab_region == '' else args.mask_ab_region
                args.mask_ab_indices = None if args.mask_ab_indices == '' else args.mask_ab_indices
                batch = get_abag_info_from_pdb_file(pdb_file,
                                                    partners=partners,
                                                    mr_p0=mr_p0,
                                                    mr_p1=mr_p1,
                                                    partner_selection=partner_selection,
                                                    mask_ab_region=args.mask_ab_region,
                                                    mask_ab_indices=args.mask_ab_indices,
                                                    assert_contact=args.contact_residues_only,
                                                    with_metadata=True
                                                    )
                if batch is None:
                    continue
                self.d_loader.append(batch)

        self.outdir = self.args.output_dir
        
        self.lengths_dict = {}
        self.chain_breaks = {}
        contact_res_indices_p0 = {}
        contact_res_indices_p1 = {}
        with torch.no_grad():
            ids_seen = []
            for batch in self.d_loader:
                id, _, metadata = batch
                cleanid = get_cleanid_from_numpy_string(id[0])
                #print(cleanid, metadata[0])
                if (subset_ids != []) and (not cleanid in subset_ids):
                    print('continuing', cleanid)
                    continue
                self.lengths_dict[cleanid] = metadata[0]['Ab_len']
                self.chain_breaks[cleanid] = []
                if 'chain_breaks' in metadata[0]:
                    self.chain_breaks[cleanid] = metadata[0]['chain_breaks']
                if 'noncontact_mask' in metadata[0]:
                    contact_res_mask = metadata[0]['noncontact_mask']
                    contact_res_indices_p0[cleanid] = ','.join([str(t) 
                    for t in contact_res_mask.nonzero().flatten().tolist()
                    if t < self.lengths_dict[cleanid]])
                    contact_res_indices_p1[cleanid] = ','.join([str(t) 
                    for t in contact_res_mask.nonzero().flatten().tolist()
                    if t >= self.lengths_dict[cleanid]])

        if not contact_res_indices_p0 == {}: 
            with open(os.path.join(self.outdir, 'contact_res_indices_p0.txt'), 'w') as f:
                f.write('\n'.join([f'{cleanid}\t{contact_res_indices_p0[cleanid]}'
                                    for cleanid in contact_res_indices_p0]))
            with open(os.path.join(self.outdir, 'contact_res_indices_p1.txt'), 'w') as f:
                f.write('\n'.join([f'{cleanid}\t{contact_res_indices_p1[cleanid]}'
                                    for cleanid in contact_res_indices_p1]))


    def sample(self, temp=1.0, N=100, 
               partner_selection='Ab', region_selection=None,
               intersect_with_contacts=False, write_fasta_for_colab_argmax=False,
               write_fasta_for_colab_sampled=False,
               subset_ids=[], mask_indices=None,
               plot_metrics=True,
               write_igfold_output=True
               ):
        import json
        import numpy as np

        print('Subset ids:', subset_ids)

        self.get_dataloader(partner_selection=partner_selection,
                            region_selection=region_selection,
                            intersect_with_contacts=intersect_with_contacts,
                            subset_ids=subset_ids,
                            mask_indices=mask_indices)

        seqrec_sampled_dict = {}
        seqrec_argmax_dict= {}
        perplexity_dict = {}
        total_nodes = {}
        
        with torch.no_grad():
            ids_seen = []
            for batch in self.d_loader:
                id, _, _ = batch
                cleanid= get_cleanid_from_numpy_string(id[0])
                if cleanid in ids_seen:
                    continue
                recovery_dict = get_recovery_metrics_for_batch(batch, self.model, temp, N,
                                                   gmodel_data=self.args.protein_gmodel)
                print(cleanid, recovery_dict['seqrecargmax'])
                seqrec_argmax_dict[cleanid] = recovery_dict['seqrecargmax']
                seqrec_sampled_dict[cleanid] = recovery_dict['seqrecsampled_all']

                total_nodes[cleanid] = recovery_dict['total_nodes']
                loss = recovery_dict['loss']
                perplexity_dict[cleanid] = float(np.exp(loss.cpu().numpy()))
                sequence_argmax = recovery_dict['sequence_argmax']
                sequences_sampled = recovery_dict['sequences_sampled']
                sequence_wt = recovery_dict['wt']

                p0_len = self.lengths_dict[cleanid]
                heavy_length = self.chain_breaks[cleanid][0]
                comma_separated_outstr = f'pdbid={cleanid},total_nodes={total_nodes[cleanid]},heavy_len={heavy_length},perplexity={perplexity_dict[cleanid]},score={loss},'
                comma_separated_outstr += f'recovery={{}}'

                insert_char=''
                if partner_selection=='both':
                    insert_char=':'
                outstr = '>seqwildtype\n'
                outstr += get_seq_from_array(sequence_wt, p0_len, insert_char=insert_char, 
                                             partner_selection=partner_selection)
                outstr += f'>seqargmax,T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
                outstr += get_seq_from_array(sequence_argmax, p0_len, insert_char=insert_char, 
                                             partner_selection=partner_selection)
                
                for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled_dict[cleanid])):
                    outstr += f'>seq{j},T={temp},'+comma_separated_outstr.format(rec)+'\n'
                    outstr += get_seq_from_array(seq, p0_len, insert_char=insert_char, 
                                                 partner_selection=partner_selection)
                
                outdir = f'{self.outdir}/{cleanid}'
                os.makedirs(outdir, exist_ok=True)
                outfile_sampled = f'{outdir}/{cleanid}_sequences_sampled_temp{temp}_N{N}_{self.partner_selection}.fasta'
                with open(outfile_sampled, 'w') as f:
                    f.write(outstr)

                if write_igfold_output:
                    print(recovery_dict['design_indices'])
                    outfile_indices = f'{outdir}/{cleanid}_sequences_sampled_temp{temp}_N{N}_{self.partner_selection}.txt'
                    indices_seqs = []
                    design_indices = recovery_dict['design_indices']
                    for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled_dict[cleanid])):
                        full_seq = get_seq_from_array(seq, p0_len, insert_char='', 
                                                    partner_selection=partner_selection)
                        indices_seqs.append(''.join([full_seq[ind] for ind in design_indices]))
                    uniq_seqs = list(set(indices_seqs))
                    with open(outfile_indices, 'w') as f:
                        f.write('\n'.join(uniq_seqs))
                    if plot_metrics:
                        wt_seq_letters = get_seq_from_array(sequence_wt, p0_len, insert_char=insert_char, 
                                             partner_selection=partner_selection)
                        wt_seq_indices = ''.join([wt_seq_letters[ind] for ind in design_indices])
                        outfile_pattern_indices = f'{outdir}/{{}}_{cleanid}_sequences_sampled_temp{temp}_N{N}_{self.partner_selection}.png'
                        plot_sequence_metrics(uniq_seqs, indices=design_indices, wt_seq=wt_seq_indices, 
                                                outfile_pattern=outfile_pattern_indices)
    
                
                if write_fasta_for_colab_sampled:
                    # Sampled
                    colab_file = f'{self.outdir}/sampled_multimer_colab_{partner_selection}_temp{temp}/{cleanid}_seq{{}}_sampled_temp{temp}_{partner_selection}.fasta'
                    os.makedirs(f'{self.outdir}/sampled_multimer_colab_{partner_selection}_temp{temp}', exist_ok=True)
                    for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled_dict[cleanid])):
                        outstr = f'>seq{j},T={temp},'+comma_separated_outstr.format(rec)+'\n'
                        outstr += get_multimer_seq_from_array(seq, p0_len,  
                                                              chain_breaks=self.chain_breaks[cleanid],
                                                              include_Ag=False)
                        with open(colab_file.format(j), 'w') as f:
                            f.write(outstr)
                
                if write_fasta_for_colab_argmax:
                    colab_file = f'{self.outdir}/argmax_multimer_colab_{partner_selection}/{cleanid}_argmax_temp0_{partner_selection}.fasta'
                    if not os.path.exists(colab_file):
                        os.makedirs(f'{self.outdir}/argmax_multimer_colab_{partner_selection}/', exist_ok=True)
                        outstr = f'>seqargmax_{cleanid},T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
                        outstr += get_multimer_seq_from_array(sequence_argmax, p0_len, 
                                                              chain_breaks=self.chain_breaks[cleanid],
                                                              include_Ag=False)
                        with open(colab_file, 'w') as f:
                            f.write(outstr)

                if write_fasta_for_colab_sampled or write_fasta_for_colab_argmax:
                    colab_file = f'{self.outdir}/wildtype_multimer_colab_{partner_selection}/{cleanid}_wildtype.fasta'
                    if not os.path.exists(colab_file):
                        os.makedirs(f'{self.outdir}/wildtype_multimer_colab_{partner_selection}', exist_ok=True)
                        outstr = f'>seqwildtype_{cleanid}\n'
                        outstr += get_multimer_seq_from_array(sequence_wt, p0_len, 
                                                              chain_breaks=self.chain_breaks[cleanid],
                                                              include_Ag=False)
                        with open(colab_file, 'w') as f:
                            f.write(outstr)
                    
                    outfile_json = f'{self.outdir}/sequence_recovery_argmax_{partner_selection}.json'
                    print(seqrec_argmax_dict)
                    json.dump(seqrec_argmax_dict, open(outfile_json, 'w'))


                outfile_argmax = f'{outdir}/{cleanid}_sequences_argmax_{self.partner_selection}.fasta'
                outfile_argmax = f'{outdir}/{cleanid}_sequences_argmax.fasta'
                outstr = '>seqwildtype\n'
                outstr += num_to_letter(sequence_wt, _aa_dict) + '\n'
                outstr += f'>seqargmax,T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
                outstr += num_to_letter(sequence_argmax, _aa_dict) + '\n'
                with open(outfile_argmax, 'w') as f:
                    f.write(outstr)

                os.makedirs(f'{outdir}/wildtype/', exist_ok=True)
                outfile_wt = f'{outdir}/wildtype/{cleanid}_widtype.fasta'
                outstr = '>seqwildtype\n'
                outstr += num_to_letter(sequence_wt, _aa_dict) + '\n'
                with open(outfile_wt, 'w') as f:
                    f.write(outstr)

                outfile_loss_res = f'{outdir}/{cleanid}_per_res_loss_{self.partner_selection}.npy'
                np.save(outfile_loss_res, recovery_dict['loss_full'], allow_pickle=True)
        outfile_json = f'{self.outdir}/sequence_recovery_argmax_{self.partner_selection}.json'
        print(seqrec_argmax_dict)
        json.dump(seqrec_argmax_dict, open(outfile_json, 'w'))


if __name__ == '__main__':

    args = _get_args()
    psampler = AntibodyAntigenSequenceSampler()
    
    temperatures = [float(t) for t in args.sample_temperatures.split(',')]
    n_samples = [int(t) for t in args.num_samples.split(',')]
    mask_indices = [int(t) for t in args.mask_ab_indices.split(',')] \
                        if args.mask_ab_indices!='' \
                        else None
    region_selection = args.mask_ab_region \
                        if args.mask_ab_region!='' \
                        else None
    
    for temp in temperatures:
        for N in n_samples:
            psampler.sample(temp=temp, N=N, partner_selection=args.partner_name,
                            mask_indices=mask_indices,
                            region_selection=region_selection,
                            write_fasta_for_colab_sampled=True)
    