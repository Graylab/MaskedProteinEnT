import os
import json
import glob
import torch
import numpy as np
import pandas as pd
from utils.metrics \
    import get_recovery_metrics_for_batch, get_cleanid_from_numpy_string
from src.model.ProteinMaskedLabelModel_EnT_MA import ProteinMaskedLabelModel_EnT_MA
from src.data.constants import num_to_letter, _aa_dict
from utils.prepare_model_inputs_from_pdb \
                import get_ppi_info_from_pdb_file, get_abag_info_from_pdb_file
from utils.command_line_utils import _get_args
from utils.ppi_sequence_writer import PPISequenceWriter
import sys
import warnings
warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)


class PPISequenceSampler():
    def __init__(self, args, mr=1.0):
        super().__init__()

        self.args = args
        self.gmodel = self.args.protein_gmodel
    
        self.model = ProteinMaskedLabelModel_EnT_MA.load_from_checkpoint(self.args.model).to(device)
        self.model.freeze()
        
        self.args.train_split = 0
        self.args.shuffle_dataset=False
        self.args.masking_rate_max = mr
        if self.args.antibody:
            self.region_selection = self.args.mask_ab_region
        
        self.outdir = self.args.output_dir

    def get_dataloader(self, partner_selection='Ab', output_indices=False, 
                        subset_ids=[], max_samples=None):
        
        if self.args.from_pdb == '':
            from src.datamodules.dataloaders import get_dataloader_for_testing
            mr_min = 0.0
            if partner_selection == 'both':
                mr_min = self.args.masking_rate_max
            self.d_loader = get_dataloader_for_testing(mr=self.args.masking_rate_max,
                                    mr_min=mr_min,
                                    partner_selection=partner_selection,
                                    with_metadata=False,
                                    region_selection=self.region_selection,
                                    intersect_with_contacts=self.args.contact_residues_only)
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
                if args.antibody:
                    args.mask_ab_region = None if args.mask_ab_region == '' else args.mask_ab_region
                    args.mask_ab_indices = None if args.mask_ab_indices == '' else args.mask_ab_indices
                    print('Masking regions: ', args.mask_ab_region)
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
                else:
                    batch = get_ppi_info_from_pdb_file(pdb_file, partners=partners,
                                                        mr_p0=mr_p0,
                                                        mr_p1=mr_p1,
                                                        with_metadata=True)
                if batch is None:
                    continue
                self.d_loader.append(batch)
        
        self.lengths_dict = {}
        self.chain_breaks = {}
        contact_res_indices_p0 = {}
        contact_res_indices_p1 = {}
        with torch.no_grad():
            for batch in self.d_loader:
                id, _, metadata = batch
                cleanid = get_cleanid_from_numpy_string(id[0])
                if (subset_ids != []) and (not cleanid in subset_ids):
                    print('continuing', cleanid)
                    continue
                if args.antibody:
                    self.lengths_dict[cleanid] = metadata[0]['Ab_len']
                else:
                    self.lengths_dict[cleanid] = metadata[0]['p0_len']
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
        
        self.sequence_writer = PPISequenceWriter(self.outdir, self.chain_breaks, partner_selection, self.lengths_dict)



    def sample(self, temp=1.0, N=100, 
               write_fasta_for_colab_argmax=False,
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
        total_nodes = {}
        with torch.no_grad():
            ids_seen = []
            for batch in self.d_loader:
                id, _, _ = batch
                cleanid= get_cleanid_from_numpy_string(id[0])
                if subset_ids != []:
                    if not cleanid in subset_ids:
                        continue
                if cleanid in ids_seen:
                    continue
                
                recovery_dict = \
                    get_recovery_metrics_for_batch(batch, self.model, temp, N)
                print(cleanid, recovery_dict['seqrecargmax'])
                seqrec_argmax_dict[cleanid] = recovery_dict['seqrecargmax']
                seqrec_sampled_dict[cleanid] = recovery_dict['seqrecsampled_all']
                self.sequence_writer.write_sequences(recovery_dict, partner_name,
                                                    write_fasta_for_colab_argmax=write_fasta_for_colab_argmax,
                                                    write_fasta_for_colab_sampled=write_fasta_for_colab_sampled)
                
        outfile_json = f'{self.outdir}/sequence_recovery_argmax_{partner_name}.json'
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
            psampler.sample(temp=temp, N=N, partner_name=args.partner_name,
                            subset_ids=ids, write_fasta_for_colab_sampled=True)