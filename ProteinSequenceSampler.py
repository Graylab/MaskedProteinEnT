import os
import torch
import glob
from src.data.constants import num_to_letter, _aa_dict
from utils.command_line_utils import _get_args
from utils.prepare_model_inputs_from_pdb import get_protein_info_from_pdb_file,\
get_antibody_info_from_pdb_file
from src.model.ProteinMaskedLabelModel_EnT_MA import ProteinMaskedLabelModel_EnT_MA
from utils.metrics import get_recovery_metrics_for_batch, score_sequences, get_cleanid_from_numpy_string
from utils.protein_sequence_writer import ProteinSequenceWriter
import warnings
warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)


class ProteinSequenceSampler():
    def __init__(self, args, mr=1.0):
        super().__init__()

        self.args = args
        self.gmodel = self.args.protein_gmodel
    
        self.model = ProteinMaskedLabelModel_EnT_MA.load_from_checkpoint(self.args.model).to(device)
        self.model.freeze()
        
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
                if args.antibody:
                    batch = get_antibody_info_from_pdb_file(pdb_file)
                else:
                    batch = get_protein_info_from_pdb_file(pdb_file)
                self.d_loader.append(batch)
        self.outdir = self.args.output_dir
        self.sequence_writer = ProteinSequenceWriter(self.outdir)


    def sample(self, temp=1.0, N=100, write_fasta_for_colab_argmax=False,
               write_fasta_for_colab_sampled=False,
                subset_ids=[]):
        import json
        import numpy as np

        seqrec_sampled_dict = {}
        seqrec_argmax_dict= {}
        
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
                self.sequence_writer.write_sequences(recovery_dict, 
                                     write_fasta_for_colab_argmax=write_fasta_for_colab_argmax,
                                     write_fasta_for_colab_sampled=write_fasta_for_colab_sampled)

        outfile_json = f'{self.outdir}/sequence_recovery_argmax.json'
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
