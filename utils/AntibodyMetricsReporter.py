import torch
import numpy as np
import random
import os
import sys
import json
from deeph3.datasets.AntibodyMaskedMultiAtomDatasetBatched\
     import AntibodyMaskedMultiAtomDatasetBatched 
from deeph3.models.ProteinMaskedLabelModel.utils.command_line_utils \
     import _get_args, split_dataset
from deeph3.models.ProteinMaskedLabelModel.dataloaders.MaskedSequenceStructureMADataModule\
     import get_ppi_dataset_setup, _helper_loader
from deeph3.util.util import _aa_dict, num_to_letter
from tqdm import tqdm
from deeph3.models.ProteinMaskedLabelModel.train_masked_model \
         import load_model, get_dataloaders_ppi


torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

def get_clean_id(id):
    return str(id)[2:-1]

class AntibodyMetricsReporter():
    def __init__(self,
                partner_selection='random',
                region_selection=None,
                intersect_with_contacts=False
                ):
        super().__init__()

        self.args = _get_args()
        self.use_multiple_models = self.args.use_multiple_models
        self.gmodel = self.args.protein_gmodel
        self.partner_selection = partner_selection
        self.region_selection = region_selection
        self.intersect_with_contacts = intersect_with_contacts
        self.region_keys = ['h1', 'h2', 'h3', 'l1', 'l2', 'l3', 'non-cdr', 'Ab', 'Ag']
        self.acc_region_dataset = (torch.zeros((len(self.region_keys))).to(device),
                  torch.zeros((len(self.region_keys))).to(device)
                 )
        if not self.use_multiple_models:
            self.model = load_model({}, self.args.model, self.args.protein_gmodel).to(device)
            self.model.freeze()
        else:
            self.model_1 = load_model({}, self.args.model_1, self.args.protein_gmodel).to(device)
            self.model_2 = load_model({}, self.args.model_2, self.args.protein_gmodel).to(device)
            self.model_1.freeze()
            self.model_2.freeze()
            self.model_dict = {type(self.model_1): self.model_1, type(self.model_2): self.model_2}
        
        self.write_sequences = False


    def setup_pertarget_dictionaries(self, trials_sampling, trials_masking):
        self.ids = [get_clean_id(a[0][0]) for a in tqdm(self.val_dataloader)]
        #print(self.ids)

        self.acc_region_sample = {}
        for id in self.ids:
            self.acc_region_sample[id] = torch.full((len(self.region_keys),
                                                    trials_masking), float('nan')).to(device)
                    
        self.acc_region_dataset = (torch.zeros((len(self.region_keys))).to(device),
                  torch.zeros((len(self.region_keys))).to(device)
                 )
        
        self.acc_region_sample_sampled = {}
        for id in self.ids:
            self.acc_region_sample_sampled[id] = torch.full((len(self.region_keys),
                                                    trials_sampling, trials_masking),
                                                    float('nan')).to(device)
        
        self.acc_region_sampled_dataset = (torch.zeros((len(self.region_keys))).to(device),
                  torch.zeros((len(self.region_keys))).to(device)
                 )
        
        self.perplexity_region_sample = {}
        for id in self.ids:
            self.perplexity_region_sample[id] = (torch.zeros((len(self.region_keys))).to(device),
                    torch.zeros((len(self.region_keys))).to(device)
                    )

        self.sequences_argmax = {}
        self.sequences_samples = {}
        self.index_strings = {}
        for id in self.ids:
            self.sequences_argmax[id] = []
            self.sequences_samples[id] = []
            self.index_strings[id] = []


    def load_datasets(self, mr, mr_min=0):
        
        # load ppi dataset
        if self.args.h5_file != '' or self.args.h5_file_ppi != '':
            self.args.masking_rate_min = mr_min
            self.args.masking_rate_max = mr
            _, self.val_dataloader = get_dataloaders_ppi(self.args, with_metadata=True)
            self.val_dataloader.dataset.dataset.noncontact_mask_rate = 0.0
            self.val_dataloader.dataset.dataset.partner_selection = self.partner_selection
            self.val_dataloader.dataset.dataset.mask_ab_region = self.region_selection
            self.val_dataloader.dataset.dataset.intersect_contact_and_region=self.intersect_with_contacts
            assert self.val_dataloader.dataset.dataset.percent_mask[0] == mr
            assert self.val_dataloader.dataset.dataset.percent_mask[1] == 0
            assert self.val_dataloader.dataset.dataset.noncontact_mask_rate == 0.
            assert self.val_dataloader.dataset.dataset.intersect_contact_and_region \
                ==self.intersect_with_contacts
            #print(self.val_dataloader.dataset.dataset.num_proteins)

        if self.args.h5_file_ab != '':
            self.args.masking_rate_min = mr_min
            self.args.masking_rate_max = mr
            #print('Ab dataset')
            shared_arguments = get_ppi_dataset_setup(self.args)
            # Load this abag dataset like a ppi dataset
            dataset = AntibodyMaskedMultiAtomDatasetBatched(
                self.args.h5_file_ab, **shared_arguments)
            #print(dataset)
            _, self.validation_ab_dataset = split_dataset(
                dataset, self.args)
            self.val_dataloader_abonly = _helper_loader(self.validation_ab_dataset,
                                                        self.args, with_metadata=True)
            self.val_dataloader_abonly.dataset.dataset.noncontact_mask_rate = 0.0
            self.val_dataloader_abonly.dataset.dataset.partner_selection = self.partner_selection
            self.val_dataloader_abonly.dataset.dataset.mask_ab_region = self.region_selection
            self.val_dataloader_abonly.dataset.dataset.intersect_contact_and_region=self.intersect_with_contacts
            assert self.val_dataloader_abonly.dataset.dataset.percent_mask[0] == mr
            assert self.val_dataloader_abonly.dataset.dataset.percent_mask[1] == 0
            assert self.val_dataloader_abonly.dataset.dataset.noncontact_mask_rate == 0.
            assert self.val_dataloader_abonly.dataset.dataset.intersect_contact_and_region \
                ==self.intersect_with_contacts

    def update_per_region_correct_argmax(self, id, region_dict, i_trial,
                                         target_unmasked, correct_unmasked):
        for ikey, key in enumerate(self.region_keys):
            if key in region_dict:
                # Use these for multiregion masking - collect over full dataset
                num_of_masked_residues = torch.sum(target_unmasked[region_dict[key]])
                self.acc_region_dataset[1][ikey] += \
                    num_of_masked_residues
                self.acc_region_dataset[0] [ikey] += \
                    torch.sum(correct_unmasked[region_dict[key]])
                
                # per target accuracy - only should be used for 1 region masking
                if num_of_masked_residues.item() > 0:
                    self.acc_region_sample[id][ikey, i_trial] = \
                        (torch.sum(correct_unmasked[region_dict[key]]) /
                                        num_of_masked_residues)
                else:
                    self.acc_region_sample[id][ikey, i_trial] = float('nan')
                #print('1',key, id, self.acc_region_sample[id][ikey, i_trial],
                #        num_of_masked_residues.item())
    

    def update_per_region_correct_sampled(self, id, region_dict, i_sample, i_mask,
                                         target_unmasked, correct_unmasked):
        for ikey, key in enumerate(self.region_keys):
            if key in region_dict:
                # Use these for multiregion masking
                self.acc_region_sampled_dataset[1][ikey] += \
                    torch.sum(target_unmasked[region_dict[key]])
                self.acc_region_sampled_dataset[0] [ikey] += \
                    torch.sum(correct_unmasked[region_dict[key]])
                # per target accuracy - only should be used for 1 region masking
                num_of_masked_residues = torch.sum(target_unmasked[region_dict[key]])
                if num_of_masked_residues > 0:
                    self.acc_region_sample_sampled[id][ikey, i_sample, i_mask] =\
                        torch.sum(correct_unmasked[region_dict[key]]) /\
                                        num_of_masked_residues
                else:
                    self.acc_region_sample_sampled[id][ikey, i_sample, i_mask] = float('nan')


    def aggregate_per_sample_metrics(self, acc_region_sample):
        # aggregate accuracy over trials performed per benchmark point
        avg_acc_region_sample = {}
        id_acc_region_sample = {}
        for ikey, key in enumerate(self.region_keys):
            acc_region_sample_id_dict = {}
            for id in self.ids:
                #remove nans -> cases where num of masked residues was 0
                acc_region_sample_id_ikey = torch.tensor(acc_region_sample[id][ikey, :])
                acc_nonzero = acc_region_sample_id_ikey == acc_region_sample_id_ikey

                # if this is true for all elements; set nan
                # nan -> regions that usually do not contact Ag 
                # empty -> abs without certain loops etc. light loops for nanobodies
                if ((acc_nonzero).nonzero().nelement() == 0) or \
                    (acc_region_sample_id_ikey.nelement()==0):
                    acc_region_sample_id_dict[id] = float('nan')
                # calculate mean for non-nan cases
                else:    
                    acc_region_sample_id_dict[id] \
                    = np.mean(acc_region_sample_id_ikey[acc_nonzero].cpu().numpy()).item()
                #print(key, id, acc_region_sample_id_dict[id])
            avg_acc_temp = np.array(list(acc_region_sample_id_dict.values()))
            avg_acc_region_sample[ikey] = avg_acc_temp[~np.isnan(avg_acc_temp)]
            id_acc_region_sample[key] = acc_region_sample_id_dict
            #print(key, avg_acc_temp[~np.isnan(avg_acc_temp)].mean())
        return avg_acc_region_sample, id_acc_region_sample


    def run_batch(self, batch, i_mask, trials_sampling, temp=1.0):
        id, input_data, metadata = batch
        cleanid = get_clean_id(id[0])
        input_data_len = len(input_data)
        if input_data_len == 6:
            nfeats, coords, _, mask, labels, pos_indices = input_data
        elif input_data_len == 5:
            nfeats, coords, _, mask, labels = input_data
        else:
            sys.exit('Model with inputs not supported: Input length: {}'.format(input_data_len))
        labels_list = list(labels.detach().numpy())
        region_dict = metadata[0]['cdrs']
        if not self.region_selection is None:
            if not self.region_selection in region_dict:
                return
        cdr_indices = []
        for key in region_dict:
            cdr_indices += region_dict[key]
        #print(cdr_indices)
        region_dict['non-cdr'] = [t for t in range(metadata[0]['Ab_len'])
                                    if not t in cdr_indices]
        region_dict['Ab'] = [t for t in range(metadata[0]['Ab_len'])]
        region_dict['Ag'] = [t for t in range(len(labels_list)) 
                            if t >= metadata[0]['Ab_len']]
        
        nfeats_residues = nfeats.clone().reshape(nfeats.shape[0], labels.shape[0], -1, nfeats.shape[-1])[:, :, 0, :20].squeeze(0)
        
        nfeats = nfeats.to(device)
        coords = coords.to(device)
        mask = mask.to(device)

        if self.gmodel in ['egnn-trans', 'egnn-trans-ppi']:
            y_hat, _, _ = self.model(nfeats, coords.double(), mask=mask)
        elif self.gmodel in ['egnn-trans-ma', 'egnn-trans-ma-ppi']:
            pos_indices = pos_indices.to(device)
            y_hat, _, *_ = self.model(nfeats, coords.double(), mask=mask, pos_indices=pos_indices)
        else:
            sys.exit('MODEL {} not found'.format(self.gmodel))
        
        labels = labels.to(device)
        loss = torch.nn.functional.cross_entropy(y_hat,
                                            labels,
                                            ignore_index=20,
                                            reduction='mean')

        dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                    float('-inf')).type_as(y_hat)
        y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)
        preds_max = y_hat_ignore_index.argmax(dim=-1)
        assert preds_max.shape == labels.shape
        preds_max = preds_max.cpu()
        labels = labels.cpu()
        pred_indices = list(torch.nonzero(labels!=20).numpy().flatten())
        fixed_indices = list(torch.nonzero(labels==20).numpy().flatten())
        fixed_labels = nfeats_residues[fixed_indices, :].argmax(dim=-1)
        self.index_strings[cleanid] = pred_indices
        #print(num_to_letter(preds_max[pred_indices].numpy(), _aa_dict), 
        #    num_to_letter(labels[pred_indices].numpy(), _aa_dict))
        correct_unmasked = torch.zeros(labels.shape).type_as(labels)
        correct_unmasked[preds_max == labels] = 1
        correct_unmasked[labels == 20] = 0
        target_unmasked = torch.ones(labels.shape).type_as(labels)
        target_unmasked[labels == 20] = 0
        self.update_per_region_correct_argmax(cleanid, region_dict, i_mask,
                                            target_unmasked, correct_unmasked)
        if self.write_sequences:
            preds_max[fixed_indices] = fixed_labels
            self.sequences_argmax[cleanid].append(preds_max.numpy()[:metadata[0]['Ab_len']])

        #sampled not argmax
        for i_sample in range(trials_sampling):
            y_predicted_all = torch.nn.functional.softmax(y_hat.cpu()/temp, dim=1)
            y_sampled = torch.multinomial(y_predicted_all, 1, replacement=True).squeeze(1)
            correct_unmasked = torch.zeros(labels.shape).type_as(labels)
            correct_unmasked[y_sampled == labels] = 1
            correct_unmasked[labels == 20] = 0
            target_unmasked = torch.ones(labels.shape).type_as(labels)
            target_unmasked[labels == 20] = 0
            self.update_per_region_correct_sampled(cleanid, region_dict, i_sample, i_mask,
                                                    target_unmasked, correct_unmasked)
            if self.write_sequences:
                y_sampled[fixed_indices] = fixed_labels
                self.sequences_samples[cleanid].append(y_sampled.numpy()[:metadata[0]['Ab_len']])

        return loss
    

    def run_trials_for_seqrec(self, trials_masking=10,
                              trials_sampling=10, mr=0.15, 
                              temp=0.1, 
                              suffix='', subdir=None,
                              write_sequences=False):
        if (self.region_selection in self.region_keys) and mr==1.0 and \
            (self.partner_selection == 'Ab') and write_sequences:
            self.write_sequences=True
        else:
            self.write_sequences=False
        
        #ORDER is important - write_sequences must be set before targetdict
        self.load_datasets(mr)
        self.setup_pertarget_dictionaries(trials_sampling, trials_masking)
        
        for i_mask in range(trials_masking):
            random.seed(random.randrange(100000))
            with torch.no_grad():
                if not self.use_multiple_models:
                    for batch in tqdm(self.val_dataloader):
                        self.run_batch(batch, i_mask, trials_sampling, temp=temp)
                else:
                    sys.exit('Multiple models not implemented')
        # aggregate argmax: temp=0
        print('argmax')
        avg_acc_region_sample,id_acc_region_sample = self.aggregate_per_sample_metrics(self.acc_region_sample)
        
        # aggregate sampled: temp>0; Sampling from multinomial
        # first flatten sampled metrics for masking and sampling
        for idkey in self.acc_region_sample_sampled:
            self.acc_region_sample_sampled[idkey] = \
                self.acc_region_sample_sampled[idkey].reshape(len(self.region_keys),
                                                              trials_sampling*trials_masking)
        print('Sampled')
        avg_acc_region_sample_sampled,id_acc_region_sample_sampled = \
             self.aggregate_per_sample_metrics(self.acc_region_sample_sampled)
        
        output_dir = self.args.output_dir
        if not subdir is None:
            output_dir = os.path.join(output_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)

        filename = '{}/{{}}_mr{}_sel{}_trials{}{}.npy'.format(output_dir,
                                                            mr,
                                                            self.partner_selection,
                                                            trials_masking,
                                                            suffix)
        
        self.acc_region_dataset[1][:][self.acc_region_dataset[1][:]==0] = 1.0    
        acc_region_avg = self.acc_region_dataset[0] / self.acc_region_dataset[1]

        self.acc_region_sampled_dataset[1][:][self.acc_region_sampled_dataset[1][:]==0] = 1.0
        acc_region_sampled_avg = self.acc_region_sampled_dataset[0] / self.acc_region_sampled_dataset[1]
        #print(self.acc_region_sampled_dataset[1], self.acc_region_sampled_dataset[0])
        #print(acc_region_sampled_avg)

        outfile = filename.format('accuracy_by_region_dataset')
        np.save(outfile, acc_region_avg.cpu().numpy(),
                allow_pickle=False
                )
        
        acc_dict = {}
        acc_dict_sampled = {}
        acc_dict_sampled_max = {}
        for ikey, key in enumerate(self.region_keys):
            if (key in ['Ab', 'Ag']) and mr < 1.0:
                #print('1.', ikey, key, acc_region_avg[ikey])
                acc_dict[key] = acc_region_avg[ikey].item()
                acc_dict_sampled[key] = acc_region_sampled_avg[ikey].item()
                acc_dict_sampled_max[key] = float('nan')
            elif (key in ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']) \
                 and self.intersect_with_contacts:
                # here too each sample may have varying number of contact residues
                # best to average over all samples
                acc_dict[key] = acc_region_avg[ikey].item()
                acc_dict_sampled[key] = acc_region_sampled_avg[ikey].item()
                # get max anyway
                #max per id
                acc_region_sample_sampled_max = {}
                for idkey in self.acc_region_sample_sampled:
                    tmp = self.acc_region_sample_sampled[idkey][ikey, :].cpu().numpy().flatten()
                    if tmp[~np.isnan(tmp)].size == 0:
                        continue
                    acc_region_sample_sampled_max[idkey] = (tmp[~np.isnan(tmp)]).max().item()
                acc_dict_sampled_max[key] = np.mean(list(acc_region_sample_sampled_max.values()))
            else:
                # either 100% masking or per-loop masking
                acc_dict[key] = np.mean(avg_acc_region_sample[ikey]).item()
                acc_dict_sampled[key] = np.mean(avg_acc_region_sample_sampled[ikey]).item()
                #max per id
                acc_region_sample_sampled_max = {}
                for idkey in self.acc_region_sample_sampled:
                    tmp = self.acc_region_sample_sampled[idkey][ikey, :].cpu().numpy().flatten()
                    if tmp[~np.isnan(tmp)].size == 0:
                        continue
                    acc_region_sample_sampled_max[idkey] = (tmp[~np.isnan(tmp)]).max().item()
                acc_dict_sampled_max[key] = np.mean(list(acc_region_sample_sampled_max.values()))
        
        filename_ac_dict = '{}/accuracy_by_region_mr{}_sel{}_trials{}{}.json'.format(output_dir,
                                                                                     mr,
                                                                                     self.partner_selection,
                                                                                     trials_masking,
                                                                                     suffix
                                                                                     )
        acc_dict['model'] = self.args.model
        open(filename_ac_dict, 'w').write(json.dumps(acc_dict))
        
        filename_ac_dict = '{}/accuracy_by_region_sampled_mr{}_temp{}_sel{}_trials{}{}.json'.format(output_dir,
                                                                                     mr,
                                                                                     temp,
                                                                                     self.partner_selection,
                                                                                     trials_masking,
                                                                                     suffix
                                                                                     )
        acc_dict_sampled['model'] = self.args.model
        open(filename_ac_dict, 'w').write(json.dumps(acc_dict_sampled))

        filename_ac_dict = '{}/accuracy_by_region_sampledMax_mr{}_temp{}_sel{}_trials{}{}.json'.format(output_dir,
                                                                                     mr,
                                                                                     temp,
                                                                                     self.partner_selection,
                                                                                     trials_masking,
                                                                                     suffix
                                                                                     )
        acc_dict_sampled_max['model'] = self.args.model
        open(filename_ac_dict, 'w').write(json.dumps(acc_dict_sampled_max))

        if (mr==1.0):
            filename_ac_pertarget_dict = '{}/accuracy_pertarget_by_region_sampled_mr{}_temp{}_sel{}_trials{}{}.json'.format(output_dir,
                                                                                        mr,
                                                                                        temp,
                                                                                        self.partner_selection,
                                                                                        trials_masking,
                                                                                        suffix
                                                                                        )
            output = id_acc_region_sample_sampled
            if self.region_selection:
                output = id_acc_region_sample_sampled[self.region_selection]
            output['model'] = self.args.model
            open(filename_ac_pertarget_dict, 'w').write(json.dumps(output))

            filename_ac_pertarget_dict = '{}/accuracy_pertarget_by_region_mr{}_sel{}_trials{}{}.json'.format(output_dir,
                                                                                     mr,
                                                                                     self.partner_selection,
                                                                                     trials_masking,
                                                                                     suffix
                                                                                     )
            output = id_acc_region_sample
            if self.region_selection: #not none
                output = id_acc_region_sample[self.region_selection]
            output['model'] = self.args.model
            open(filename_ac_pertarget_dict, 'w').write(json.dumps(output))

        if self.write_sequences:
            for id in self.sequences_samples:
                if len(self.sequences_samples[id]) == 0:
                    continue
                filename = '{}/{}_mr{}_sel{}_cdr{}_trials{}{}.txt'.format(output_dir,
                                                                id,
                                                                mr,
                                                                self.partner_selection,
                                                                self.region_selection,
                                                                trials_sampling,
                                                                suffix)
                seq_recoveries = self.acc_region_sample_sampled[id][self.region_keys.index(self.region_selection),
                                                                    :].flatten().tolist()
                indices = ';'.join([str(t) for t in self.index_strings[id]])
                assert len(seq_recoveries) == len(self.sequences_samples[id])
                with open(filename, 'w') as f:
                    seqs = ['>{},sequence_recovery={},indices={},mr={},temp={}\n{}\n'.format(id,
                                                                    seq_rec,
                                                                    indices,
                                                                    mr,
                                                                    temp,
                                                                    num_to_letter(pred_array, _aa_dict)
                                                                    )
                            for pred_array, seq_rec in zip(self.sequences_samples[id],seq_recoveries) ]
                    f.write(''.join(seqs))

            for id in self.sequences_argmax:
                if len(self.sequences_argmax[id]) == 0:
                    continue
                filename = '{}/{}_mr{}_sel{}_cdr{}_argmax{}.txt'.format(output_dir,
                                                                id,
                                                                mr,
                                                                self.partner_selection,
                                                                self.region_selection,
                                                                suffix)
                seq_recoveries = self.acc_region_sample[id][self.region_keys.index(self.region_selection),
                                                                    :].flatten().tolist()
                indices = ';'.join([str(t) for t in self.index_strings[id]])
                assert len(seq_recoveries) == len(self.sequences_argmax[id])
                with open(filename, 'w') as f:
                    seqs = ['>{},sequence_recovery={},indices={},mr={},temp={}\n{}\n'.format(id,
                                                                    seq_rec,
                                                                    indices,
                                                                    mr,
                                                                    temp,
                                                                    num_to_letter(pred_array, _aa_dict)
                                                                    )
                            for pred_array, seq_rec in zip(self.sequences_argmax[id],seq_recoveries)
                            ]
                    f.write(''.join(seqs))
        
        return (acc_dict, id_acc_region_sample), (acc_dict_sampled, id_acc_region_sample_sampled, acc_dict_sampled_max)

