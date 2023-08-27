import os
import torch
import glob
from src.data.constants import num_to_letter, _aa_dict
from utils.command_line_utils import _get_args
from utils.prepare_model_inputs_from_pdb import get_protein_info_from_pdb_file
from src.model.ProteinMaskedLabelModel_EnT_MA import ProteinMaskedLabelModel_EnT_MA
from utils.metrics import get_recovery_metrics_for_batch

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

def get_cleanid_from_numpy_string(id):
    if type(id) is str:
        return id
    cleanid= str(id)[2:-1]
    return cleanid


class ProteinSequenceSampler():
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
        
        self.d_loader = None
        if self.args.from_pdb == '':
            from src.data.datamodules.MaskedSequenceStructureMADataModule import \
            MaskedSequenceStructureMADataModule
            datamodule = MaskedSequenceStructureMADataModule(self.args)
            datamodule.setup()
            self.d_loader = datamodule.test_dataloader()
        else:
            assert os.path.exists(self.args.from_pdb)
            if os.path.isdir(self.args.from_pdb):
                pdb_files = glob.glob(self.args.from_pdb + '/*.pdb')
            else:
                pdb_files = [self.args.from_pdb]
            self.d_loader = []
            print(f'Found {len(pdb_files)} files.')
            for pdb_file in pdb_files:
                batch = get_protein_info_from_pdb_file(pdb_file)
                self.d_loader.append(batch)
        self.outdir = self.args.output_dir


    def sample(self, temp=1.0, N=100, write_fasta_for_colab_argmax=False,
               write_fasta_for_colab_sampled=False,
                subset_ids=[]):
        import json
        import numpy as np

        seqrec_sampled_dict = {}
        seqrec_argmax_dict= {}
        perplexity_dict = {}
        total_nodes = {}
        
        with torch.no_grad():
            ids_seen = []
            for batch in self.d_loader:
                id, _ = batch
                cleanid= get_cleanid_from_numpy_string(id[0])
                
                if subset_ids != []:
                    if not cleanid in subset_ids:
                        continue
                if cleanid in ids_seen:
                    continue
                recovery_dict = get_recovery_metrics_for_batch(batch, self.model, temp, N)
                print(cleanid, recovery_dict['seqrecargmax'])
                seqrec_argmax_dict[cleanid] = recovery_dict['seqrecargmax']
                seqrec_sampled_dict[cleanid] = recovery_dict['seqrecsampled_all']

                total_nodes[cleanid] = recovery_dict['total_nodes']
                loss = recovery_dict['loss']
                perplexity_dict[cleanid] = float(np.exp(loss.cpu().numpy()))
                sequence_argmax = recovery_dict['sequence_argmax']
                sequences_sampled = recovery_dict['sequences_sampled']
                sequence_wt = recovery_dict['wt']
                comma_separated_outstr = f'pdbid={cleanid},perplexity={perplexity_dict[cleanid]},score={loss},'
                comma_separated_outstr += f'recovery={{}}'
                
                if write_fasta_for_colab_sampled:
                    os.makedirs(f'{self.outdir}/{cleanid}/{cleanid}_for_colab_N{N}', exist_ok=True)
                    colab_pattern = f'{self.outdir}/{cleanid}/{cleanid}_for_colab_N{N}/seq{{}}_sampled_temp{temp}.fasta'
                    for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled_dict[cleanid])):
                        outstr = f'>seq{j},T={temp},'+comma_separated_outstr.format(rec)+'\n'
                        outstr += num_to_letter(seq, _aa_dict) + '\n'
                        with open(colab_pattern.format(j), 'w') as f:
                            f.write(outstr)

                if write_fasta_for_colab_argmax:
                    colab_file = f'{self.outdir}/argmax_colab/{cleanid}_argmax_temp0.fasta'
                    if not os.path.exists(colab_file):
                        os.makedirs(f'{self.outdir}/argmax_colab/', exist_ok=True)
                        outstr = f'>seqargmax_{cleanid},T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
                        outstr += num_to_letter(sequence_argmax, _aa_dict) + '\n'
                        with open(colab_file, 'w') as f:
                            f.write(outstr)
                
                if not (write_fasta_for_colab_argmax or write_fasta_for_colab_sampled):
                    outfile_argmax = f'{outdir}/{cleanid}_sequences_argmax.fasta'
                    if not os.path.exists(outfile_argmax):
                        outstr = '>seqwildtype\n'
                        outstr += num_to_letter(sequence_wt, _aa_dict) + '\n'
                        outstr += f'>seqargmax,T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
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
                    outstr += f'>seqargmax,T=0,'+comma_separated_outstr.format(seqrec_argmax_dict[cleanid])+'\n'
                    outstr += num_to_letter(sequence_argmax, _aa_dict) + '\n'
                    for j, (seq, rec) in enumerate(zip(sequences_sampled, seqrec_sampled_dict[cleanid])):
                        outstr += f'>seq{j},T={temp},'+comma_separated_outstr.format(rec)+'\n'
                        outstr += num_to_letter(seq, _aa_dict) + '\n'
                    
                    outdir = f'{self.outdir}/{cleanid}'
                    os.makedirs(outdir, exist_ok=True)
                    outfile_sampled = f'{outdir}/{cleanid}_sequences_sampled_temp{temp}_N{N}.fasta'
                    with open(outfile_sampled, 'w') as f:
                        f.write(outstr)

        outfile_json = f'{self.outdir}/sequence_recovery_argmax.json'
        print(seqrec_argmax_dict)
        json.dump(seqrec_argmax_dict, open(outfile_json, 'w'))


if __name__ == '__main__':
    args = _get_args()
    psampler = ProteinSequenceSampler(args)
    temperatures = [float(t) for t in args.sample_temperatures.split(',')]
    n_samples = [int(t) for t in args.num_samples.split(',')]
    ids = [t for t in args.ids.split(',') if t!='']
    print(temperatures, n_samples)
    for temp in temperatures:
        for N in n_samples:
            psampler.sample(temp=temp, N=N, subset_ids=ids,
                            write_fasta_for_colab_sampled=True,
                            write_fasta_for_colab_argmax=True)
