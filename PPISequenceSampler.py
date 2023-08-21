import os
import json
import glob
import torch
import numpy as np
import pandas as pd
from utils.metrics \
    import get_recovery_metrics_for_batch
from src.data.dataloaders import get_dataloader_for_testing
from utils.util import num_to_letter, _aa_dict
from utils.inference.prepare_model_inputs_from_pdb \
                import get_ppi_info_from_pdb_file
from utils.inference import load_model
from utils.command_line_utils \
     import _get_args
import sys

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

def get_cleanid_from_numpy_string(id):
    if type(id) is str:
        return id
    cleanid= str(id)[2:-1]
    return cleanid

def get_multimer_seq_from_array(seq, p0_len, chain_breaks = [], insert_char=':'):
    #print(len(seq), p0_len)
    letter_seq = num_to_letter(seq, _aa_dict)
    if chain_breaks == []:
        return letter_seq[:p0_len] + insert_char + letter_seq[p0_len:] + '\n'
    else:
        seq_list = [t for t in letter_seq]
        for position in reversed(chain_breaks):
            if position < len(seq_list):
                seq_list.insert(position, ':')
        seq_string = ''.join(seq_list)
        #print([len(t) for t in seq_string.split(':')])
        return seq_string +'\n'


def get_seq_from_array(seq, p0_len, insert_char='', partner_selection='Ab'):
    letter_seq = num_to_letter(seq, _aa_dict)
    if partner_selection == 'Ab':
        return letter_seq[:p0_len] + '\n'
    elif partner_selection == 'Ag':
        return letter_seq[p0_len:] + '\n'
    else:
        return letter_seq[:p0_len] + insert_char + letter_seq[p0_len:] + '\n'


class PPISequenceSampler():
    def __init__(self, args, mr=1.0):
        super().__init__()

        self.args = args
        self.gmodel = self.args.protein_gmodel
    
        self.model = ProteinMaskedLabelModel_EnT_MA.load_from_checkpoint(self.args.model).to(device)
        self.model.freeze()
        
        self.write_sequences = True

        self.args.train_split = 0
        self.args.shuffle_dataset=False
        self.args.masking_rate_max = mr


    def get_dataloader(self, partner_selection='Ab', output_indices=False, 
                        subset_ids=[], max_samples=None):
        
        if self.args.from_pdb == '':
            mr_min = 0.0
            if partner_selection == 'both':
                mr_min = self.args.masking_rate_max
            self.d_loader = get_dataloader_for_testing(mr=self.args.masking_rate_max,
                                    mr_min=mr_min,
                                    partner_selection=partner_selection,
                                    with_metadata=False)
            d_loader_with_meta = get_dataloader_for_testing(mr=self.args.masking_rate_max,
                                    mr_min=mr_min,
                                    partner_selection=partner_selection,
                                    with_metadata=True)
        else:
            assert os.path.exists(self.args.from_pdb)
            assert os.path.exists(self.args.ppi_partners_json)

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
                
            ppi_partners = json.load(open(self.args.ppi_partners_json, 'r'))
            if os.path.isdir(self.args.from_pdb):
                pdb_files = glob.glob(self.args.from_pdb + '.pdb')
                dirname = self.args.from_pdb
            else:
                pdb_files = [self.args.from_pdb]
                dirname = os.path.dirname(self.args.from_pdb)
            self.d_loader = []
            for pdbid in ppi_partners:
                partners = ppi_partners[pdbid].split('_')
                pdb_file = glob.glob(f'{dirname}/{pdbid.lower()}_*.pdb')[0]
                batch = get_ppi_info_from_pdb_file(pdb_file, partners=partners,
                                                   mr_p0=mr_p0,
                                                   mr_p1=mr_p1)
                if batch is None:
                    continue
                self.d_loader.append(batch)
            d_loader_with_meta = []
            for pdbid in ppi_partners:
                partners = ppi_partners[pdbid].split('_')
                pdb_file = glob.glob(f'{dirname}/{pdbid.lower()}_*.pdb')[0]
                batch = get_ppi_info_from_pdb_file(pdb_file, partners=partners,
                                                   mr_p0=mr_p0,
                                                   mr_p1=mr_p1,
                                                   with_metadata=True)
                if batch is None:
                    continue
                d_loader_with_meta.append(batch)
        
        
        self.outdir = self.args.output_dir
        
        self.lengths_dict = {}
        self.chain_breaks = {}
        contact_res_indices_p0 = {}
        contact_res_indices_p1 = {}
        contact_res_indices_p1_rel = {}
        with torch.no_grad():
            for batch in d_loader_with_meta:
                id, _, metadata = batch
                cleanid = get_cleanid_from_numpy_string(id[0])
                if (subset_ids != []) and (not cleanid in subset_ids):
                    print('continuing', cleanid)
                    continue
                self.lengths_dict[cleanid] = metadata[0]['p0_len']
                self.chain_breaks[cleanid] = []
                if 'chain_breaks' in metadata[0]:
                    self.chain_breaks[cleanid] = metadata[0]['chain_breaks']
                contact_res_mask = metadata[0]['noncontact_mask']
                contact_res_indices_p0[cleanid] = ','.join([str(t) 
                for t in contact_res_mask.nonzero().flatten().tolist()
                if t < self.lengths_dict[cleanid]])
                contact_res_indices_p1[cleanid] = ','.join([str(t) 
                for t in contact_res_mask.nonzero().flatten().tolist()
                if t >= self.lengths_dict[cleanid]])
                contact_res_indices_p1_rel[cleanid] = ','.join([str(t - self.lengths_dict[cleanid]) 
                for t in contact_res_mask.nonzero().flatten().tolist()
                if t >= self.lengths_dict[cleanid]])
        if output_indices:    
            with open(os.path.join(self.outdir, 'contact_res_indices_p0.txt'), 'w') as f:
                f.write('\n'.join([f'{cleanid}\t{contact_res_indices_p0[cleanid]}'
                                    for cleanid in contact_res_indices_p0]))
            with open(os.path.join(self.outdir, 'contact_res_indices_p1.txt'), 'w') as f:
                f.write('\n'.join([f'{cleanid}\t{contact_res_indices_p1[cleanid]}'
                                    for cleanid in contact_res_indices_p1]))
            with open(os.path.join(self.outdir, 'contact_res_indices_p1_rel.txt'), 'w') as f:
                f.write('\n'.join([f'{cleanid}\t{contact_res_indices_p1_rel[cleanid]}'
                                    for cleanid in contact_res_indices_p1_rel]))


    def sample(self, temp=1.0, N=100, write_fasta_for_colab_argmax=False,
               write_fasta_for_colab_sampled=False,
                subset_ids=[], partner_name='p0'):
        print('Subset ids:', subset_ids)

        if partner_name in ['Ab', 'p0']:
            partner_selection = 'Ab'
        elif partner_name in ['Ag', 'p1']:
            partner_selection = 'Ag'
        elif partner_name in ['p0p1', 'AbAg']:
            partner_selection = 'both'
        else:
            print(f'{partner_name} not supported')
            sys.exit()
        print(partner_name, partner_selection)

        self.get_dataloader(partner_selection=partner_selection, subset_ids=subset_ids)
        
        
        seqrec_sampled_dict = {}
        seqrec_argmax_dict= {}
        perplexity_dict = {}
        total_nodes = {}
        per_res_recovery = {}
        per_res_recovery['id'] = []
        per_res_recovery['dG'] = []
        per_res_recovery['correct'] = []
        with torch.no_grad():
            ids_seen = []
            for batch in self.d_loader:
                id, _ = batch
                cleanid= get_cleanid_from_numpy_string(id[0])
                print(cleanid)
                if subset_ids != []:
                    if not cleanid in subset_ids:
                        continue
                if cleanid in ids_seen:
                    continue
                print(cleanid)
                
                recovery_dict = \
                    get_recovery_metrics_for_batch(batch, self.model, temp, N,
                                                   gmodel_data=self.args.protein_gmodel
                                                   )
                seqrec_argmax_dict[cleanid] = recovery_dict['seqrecargmax']
                seqrec_sampled_dict[cleanid] = recovery_dict['seqrecsampled_all']
                sequence_wt = recovery_dict['wt']
                
                total_nodes[cleanid] = recovery_dict['total_nodes']
                loss = recovery_dict['loss']
                perplexity_dict[cleanid] = float(np.exp(loss.cpu().numpy()))
                sequence_argmax = recovery_dict['sequence_argmax']
                sequences_sampled = recovery_dict['sequences_sampled']
                comma_separated_outstr = f'pdbid={cleanid},total_nodes={total_nodes[cleanid]},perplexity={perplexity_dict[cleanid]},score={loss},'
                comma_separated_outstr += f'recovery={{}}'
                
                p0_len = self.lengths_dict[cleanid]
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
                
                if not ( write_fasta_for_colab_sampled or write_fasta_for_colab_argmax):
                    outdir = f'{self.outdir}/{cleanid}'
                    os.makedirs(outdir, exist_ok=True)
                    outfile_sampled = f'{outdir}/{cleanid}_sequences_sampled_temp{temp}_N{N}_{partner_name}.fasta'
                    if not os.path.exists(outfile_sampled):
                        print('Writing ', outfile_sampled)
                        with open(outfile_sampled, 'w') as f:
                            f.write(outstr)
                    else:
                        print(f'{outfile_sampled} exists. Will not overwrite !!!')
                    outfile_argmax = f'{outdir}/{cleanid}_sequences_argmax_{partner_name}.fasta'
                    outstr = '>seqwildtype\n'
                    outstr += get_seq_from_array(sequence_wt, p0_len, insert_char=insert_char, 
                                                partner_selection=partner_selection)
                    outstr += f'>seqargmax,T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
                    outstr += get_seq_from_array(sequence_argmax, p0_len, insert_char=insert_char, 
                                    partner_selection=partner_selection)
                    with open(outfile_argmax, 'w') as f:
                        f.write(outstr)
                    os.makedirs(f'{outdir}/wildtype/', exist_ok=True)
                    outfile_wt = f'{outdir}/wildtype/{cleanid}_wildtype.fasta'
                    outstr = f'>seqwildtype_{partner_name}\n'
                    outstr += get_seq_from_array(sequence_wt, p0_len, insert_char=insert_char, 
                                                partner_selection=partner_selection)
                    with open(outfile_wt, 'w') as f:
                        f.write(outstr)
                
                chain_breaks = self.chain_breaks[cleanid]
                if write_fasta_for_colab_sampled:
                    # Sampled
                    colab_file = f'{self.outdir}/sampled_multimer_colab_{partner_name}_temp{temp}/{cleanid}_seq{{}}_sampled_temp{temp}_{partner_name}.fasta'
                    os.makedirs(f'{self.outdir}/sampled_multimer_colab_{partner_name}_temp{temp}', exist_ok=True)
                    for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled_dict[cleanid])):
                        outstr = f'>seq{j},T={temp},'+comma_separated_outstr.format(rec)+'\n'
                        outstr += get_multimer_seq_from_array(seq, p0_len,  
                                                              chain_breaks=chain_breaks)
                        with open(colab_file.format(j), 'w') as f:
                            f.write(outstr)
                
                if write_fasta_for_colab_argmax:
                    colab_file = f'{self.outdir}/argmax_multimer_colab_{partner_name}/{cleanid}_argmax_temp0_{partner_name}.fasta'
                    if not os.path.exists(colab_file):
                        os.makedirs(f'{self.outdir}/argmax_multimer_colab_{partner_name}/', exist_ok=True)
                        outstr = f'>seqargmax_{cleanid},T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
                        outstr += get_multimer_seq_from_array(sequence_argmax, p0_len, 
                                                              chain_breaks=chain_breaks)
                        with open(colab_file, 'w') as f:
                            f.write(outstr)

                if write_fasta_for_colab_sampled or write_fasta_for_colab_argmax:
                    colab_file = f'{self.outdir}/wildtype_multimer_colab_{partner_name}/{cleanid}_wildtype.fasta'
                    if not os.path.exists(colab_file):
                        os.makedirs(f'{self.outdir}/wildtype_multimer_colab_{partner_name}', exist_ok=True)
                        outstr = f'>seqwildtype_{cleanid}\n'
                        outstr += get_multimer_seq_from_array(sequence_wt, p0_len, 
                                                              chain_breaks=chain_breaks)
                        with open(colab_file, 'w') as f:
                            f.write(outstr)
                    
                    outfile_json = f'{self.outdir}/sequence_recovery_argmax_{partner_name}.json'
                    print(seqrec_argmax_dict)
                    json.dump(seqrec_argmax_dict, open(outfile_json, 'w'))

                
        outfile_json = f'{self.outdir}/sequence_recovery_argmax_{partner_name}.json'
        print(seqrec_argmax_dict)
        json.dump(seqrec_argmax_dict, open(outfile_json, 'w'))


if __name__ == '__main__':
    args = _get_args()
    psampler = PPISequenceSampler(args)
    temperatures = [float(t) for t in args.sample_temperatures.split(',')]
    n_samples = [int(t) for t in args.num_samples.split(',')]
    print(temperatures, n_samples)
    ids = [t for t in args.ids.split(',') if t!='']
    for temp in temperatures:
        for N in n_samples:
            print('Here')
            psampler.sample(temp=temp, N=N, partner_name=args.partner_name,
                            subset_ids=ids, write_fasta_for_colab_sampled=True)