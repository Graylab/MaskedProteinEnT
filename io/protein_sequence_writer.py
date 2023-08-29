import os
import sys
import json
from src.data.constants import num_to_letter, _aa_dict

class ProteinSequenceWriter():
    def __init__(outdir):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def write_sequences(self, recovery_dict, write_fasta_for_colab_sampled=False, 
                        write_fasta_for_colab_argmax=False):
        cleanid = recovery_dict['id']
        seqrec_argmax = recovery_dict['seqrecargmax']
        seqrec_sampled = recovery_dict['seqrecsampled_all']

        #perplexity_dict[cleanid] = float(np.exp(loss.cpu().numpy()))
        sequence_argmax = recovery_dict['sequence_argmax']
        sequences_sampled = recovery_dict['sequences_sampled']
        sequences_sampled_loss = recovery_dict['sequences_sampled_loss']
        comma_separated_outstr = f'pdbid={cleanid},score={{}},'
        comma_separated_outstr += f'recovery={{}}'

        if write_fasta_for_colab_sampled:
            os.makedirs(f'{self.outdir}/{cleanid}/{cleanid}_for_colab_N{N}', exist_ok=True)
            colab_pattern = f'{self.outdir}/{cleanid}/{cleanid}_for_colab_N{N}/seq{{}}_sampled_temp{temp}.fasta'
            for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled)):
                outstr = f'>seq{j},T={temp},'+comma_separated_outstr.format(sequences_sampled_loss[j], rec)+'\n'
                outstr += num_to_letter(seq, _aa_dict) + '\n'
                with open(colab_pattern.format(j), 'w') as f:
                    f.write(outstr)

        if write_fasta_for_colab_argmax:
            argmax_loss = recovery_dict['argmax_loss']
            colab_file = f'{self.outdir}/argmax_colab/{cleanid}_argmax_temp0.fasta'
            if not os.path.exists(colab_file):
                os.makedirs(f'{self.outdir}/argmax_colab/', exist_ok=True)
                outstr = f'>seqargmax_{cleanid},T=0,'+comma_separated_outstr.format(argmax_loss, seqrec_argmax)+'\n'
                outstr += num_to_letter(sequence_argmax, _aa_dict) + '\n'
                with open(colab_file, 'w') as f:
                    f.write(outstr)

        if not (write_fasta_for_colab_argmax or write_fasta_for_colab_sampled):
            sequence_wt = recovery_dict['wt']
            outfile_argmax = f'{outdir}/{cleanid}_sequences_argmax.fasta'
            if not os.path.exists(outfile_argmax):
                outstr = '>seqwildtype\n'
                outstr += num_to_letter(sequence_wt, _aa_dict) + '\n'
                outstr += f'>seqargmax,T=0,'+comma_separated_outstr.format(argmax_loss, seqrec_argmax)+'\n'
                outstr += num_to_letter(sequence_argmax, _aa_dict) + '\n'
                with open(outfile_argmax, 'w') as f:
                    f.write(outstr)

            outfile_wt = f'{outdir}/wildtype/{cleanid}_widtype.fasta'
            if not os.path.exists(outfile_wt):
                os.makedirs(f'{outdir}/wildtype/', exist_ok=True)
                outstr = '>seqwildtype\n'
                outstr += num_to_letter(sequence_wt, _aa_dict) + '\n'
                with open(outfile_wt, 'w') as f:
                    f.write(outstr)

            outstr = '>seqwildtype\n'
            outstr += num_to_letter(sequence_wt, _aa_dict) + '\n'
            outstr += f'>seqargmax,T=0,'+comma_separated_outstr.format(argmax_loss,seqrec_argmax)+'\n'
            outstr += num_to_letter(sequence_argmax, _aa_dict) + '\n'
            for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled)):
                outstr += f'>seq{j},T={temp},'+comma_separated_outstr.format(sequences_sampled_loss[j],rec)+'\n'
                outstr += num_to_letter(seq, _aa_dict) + '\n'

            outdir = f'{self.outdir}/{cleanid}'
            os.makedirs(outdir, exist_ok=True)
            outfile_sampled = f'{outdir}/{cleanid}_sequences_sampled_temp{temp}_N{N}.fasta'
            with open(outfile_sampled, 'w') as f:
                f.write(outstr)

