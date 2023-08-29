import os
import sys
import json
from src.data.constants import num_to_letter, _aa_dict

class PPISequenceWriter():
    def __init__(outdir, chain_breaks_dict, partner_selection):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.chain_breaks = chain_breaks_dict
        self.partner_selection = partner_selection

    @staticmethod
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

    @staticmethod
    def get_seq_from_array(seq, p0_len, insert_char='', partner_selection='Ab'):
        letter_seq = num_to_letter(seq, _aa_dict)
        if partner_selection == 'Ab':
            return letter_seq[:p0_len] + '\n'
        elif partner_selection == 'Ag':
            return letter_seq[p0_len:] + '\n'
        else:
            return letter_seq[:p0_len] + insert_char + letter_seq[p0_len:] + '\n'


    def write_sequences(self, recovery_dict, partner_name, write_fasta_for_colab_sampled=False, 
                        write_fasta_for_colab_argmax=False):
        sequence_wt = recovery_dict['wt']
                
        total_nodes[cleanid] = recovery_dict['total_nodes']
        loss = recovery_dict['loss']
        #perplexity_dict[cleanid] = float(np.exp(loss.cpu().numpy()))
        sequence_argmax = recovery_dict['sequence_argmax']
        sequences_sampled = recovery_dict['sequences_sampled']
        comma_separated_outstr = f'pdbid={cleanid},total_nodes={total_nodes[cleanid]},score={{}},'
        comma_separated_outstr += f'recovery={{}}'
        
        p0_len = self.lengths_dict[cleanid]
        insert_char=''
        partner_selection = self.partner_selection
        if partner_selection=='both':
            insert_char=':'
        outstr = '>seqwildtype\n'
        outstr += self.get_seq_from_array(sequence_wt, p0_len, insert_char=insert_char, 
                                        partner_selection=partner_selection)
        outstr += f'>seqargmax,T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
        outstr += self.get_seq_from_array(sequence_argmax, p0_len, insert_char=insert_char, 
                                        partner_selection=partner_selection)
        for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled_dict[cleanid])):
            outstr += f'>seq{j},T={temp},'+comma_separated_outstr.format(rec)+'\n'
            outstr += self.get_seq_from_array(seq, p0_len, insert_char=insert_char, 
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
            outstr += self.get_seq_from_array(sequence_wt, p0_len, insert_char=insert_char, 
                                        partner_selection=partner_selection)
            outstr += f'>seqargmax,T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
            outstr += self.get_seq_from_array(sequence_argmax, p0_len, insert_char=insert_char, 
                            partner_selection=partner_selection)
            with open(outfile_argmax, 'w') as f:
                f.write(outstr)
            os.makedirs(f'{outdir}/wildtype/', exist_ok=True)
            outfile_wt = f'{outdir}/wildtype/{cleanid}_wildtype.fasta'
            outstr = f'>seqwildtype_{partner_name}\n'
            outstr += self.get_seq_from_array(sequence_wt, p0_len, insert_char=insert_char, 
                                        partner_selection=partner_selection)
            with open(outfile_wt, 'w') as f:
                f.write(outstr)
        
        
        if write_fasta_for_colab_sampled:
            # Sampled
            colab_file = f'{self.outdir}/sampled_multimer_colab_{partner_name}_temp{temp}/{cleanid}_seq{{}}_sampled_temp{temp}_{partner_name}.fasta'
            os.makedirs(f'{self.outdir}/sampled_multimer_colab_{partner_name}_temp{temp}', exist_ok=True)
            for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled_dict[cleanid])):
                outstr = f'>seq{j},T={temp},'+comma_separated_outstr.format(rec)+'\n'
                outstr += self.get_multimer_seq_from_array(seq, p0_len,  
                                                        chain_breaks=self.chain_breaks)
                with open(colab_file.format(j), 'w') as f:
                    f.write(outstr)
        
        if write_fasta_for_colab_argmax:
            colab_file = f'{self.outdir}/argmax_multimer_colab_{partner_name}/{cleanid}_argmax_temp0_{partner_name}.fasta'
            if not os.path.exists(colab_file):
                os.makedirs(f'{self.outdir}/argmax_multimer_colab_{partner_name}/', exist_ok=True)
                outstr = f'>seqargmax_{cleanid},T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
                outstr += self.get_multimer_seq_from_array(sequence_argmax, p0_len, 
                                                        chain_breaks=self.chain_breaks)
                with open(colab_file, 'w') as f:
                    f.write(outstr)

        if write_fasta_for_colab_sampled or write_fasta_for_colab_argmax:
            colab_file = f'{self.outdir}/wildtype_multimer_colab_{partner_name}/{cleanid}_wildtype.fasta'
            if not os.path.exists(colab_file):
                os.makedirs(f'{self.outdir}/wildtype_multimer_colab_{partner_name}', exist_ok=True)
                outstr = f'>seqwildtype_{cleanid}\n'
                outstr += self.get_multimer_seq_from_array(sequence_wt, p0_len, 
                                                        chain_breaks=chain_breaks)
                with open(colab_file, 'w') as f:
                    f.write(outstr)
            
            outfile_json = f'{self.outdir}/sequence_recovery_argmax_{partner_name}.json'
            print(seqrec_argmax_dict)
            json.dump(seqrec_argmax_dict, open(outfile_json, 'w'))

