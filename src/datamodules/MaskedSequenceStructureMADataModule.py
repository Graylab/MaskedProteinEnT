import pytorch_lightning as pl
import pickle, os, sys
import torch
from typing import Optional
from pytorch_lightning.trainer.supporters import CombinedLoader
from src.data.datasets.datasets \
import split_dataset, get_protein_dataset_setup, get_ppi_dataset_setup
from src.data.dataloaders import _helper_loader
from src.data.datasets.AntibodyMaskedMultiAtomDatasetBatched import AntibodyMaskedMultiAtomDatasetBatched
from src.data.datasets.H5AbAgPPIMaskedMultiAtomDatasetBatched import H5AbAgPPIMaskedMultiAtomDatasetBatched
from src.data.datasets.ProteinMaskedMultiAtomDatasetBatched import ProteinMaskedMultiAtomDatasetBatched
from src.data.datasets.SCNProteinMaskedMultiAtomDatasetBatched import SCNProteinMaskedMultiAtomDatasetBatched


class MaskedSequenceStructureMADataModule(pl.LightningDataModule):
    def __init__(self, args):
        self.cli_args = args
        
    def prepare_data(self):
        print('Prepared')

    def setup(self, stage: Optional[str] = None):
        args = self.cli_args
        shared_arguments_protein = get_protein_dataset_setup(args)
        casp_version = args.scn_casp_version
        thinning = args.scn_sequence_similarity
        scn_path = '/scratch16/jgray21/smahaja4_active/datasets/sidechainnet'
        input_file = '{}_c{}_ss{}/sidechainnet_casp{}_{}.pkl'.format(
            scn_path, casp_version, thinning, casp_version, thinning)
        self.train_protein_dataset = None
        self.validation_protein_dataset = None
        self.test_protein_dataset = None
        ab_train_dataset_list = []
        ab_val_dataset_list = []
        if is_protein_dataset(args):
            if args.h5_file_protein == '':
                tmp_input_file = '/tmp/{}'.format(os.path.basename(input_file))
                if os.path.exists(tmp_input_file):
                    input_file = tmp_input_file
                
                if os.path.exists(input_file):
                    with open(input_file, "rb") as f:
                        d = pickle.load(f)

                    self.train_protein_dataset = SCNProteinMaskedMultiAtomDatasetBatched(
                                                    d, ['train'],
                                                    selected_ids_file=args.file_with_selected_scn_ids_for_training,
                                                    **shared_arguments_protein)
                    
                    validation_dataset_keys = [
                        key for key in d.keys() if key.find('valid') != -1
                    ]

                    self.validation_protein_dataset = SCNProteinMaskedMultiAtomDatasetBatched(d, validation_dataset_keys,
                                                                **shared_arguments_protein)

                    print(self.validation_protein_dataset)

                    if args.use_scn_valid_and_test:
                        self.test_protein_dataset = {}
                        for key in ['test'] + validation_dataset_keys:
                            self.test_protein_dataset[key] = SCNProteinMaskedMultiAtomDatasetBatched(d, [key],
                                                                **shared_arguments_protein)
                    elif args.file_with_selected_scn_ids_for_testing != '':
                        self.test_protein_dataset = SCNProteinMaskedMultiAtomDatasetBatched(d, ['test', 'train']
                                                                                        +validation_dataset_keys,
                                                                selected_ids_file=args.file_with_selected_scn_ids_for_testing,
                                                                **shared_arguments_protein)
                    else:
                        self.test_protein_dataset = SCNProteinMaskedMultiAtomDatasetBatched(d, ['test'],
                                                                **shared_arguments_protein)
            else:
                assert os.path.exists(args.h5_file_protein)
                dataset = ProteinMaskedMultiAtomDatasetBatched(
                    args.h5_file_protein, **shared_arguments_protein)
                self.train_protein_dataset, self.validation_protein_dataset = split_dataset(
                    dataset, args)
                # For testing, the h5 file provided will contain all the samples
                self.test_protein_dataset = ProteinMaskedMultiAtomDatasetBatched(
                    args.h5_file_protein, **shared_arguments_protein)
        
        if not args.h5_file_ppi == '':
            print('PPI')
            shared_arguments = get_ppi_dataset_setup(args, dataset_name='ppi')
            dataset = H5AbAgPPIMaskedMultiAtomDatasetBatched(
                args.h5_file_ppi, **shared_arguments)
            #print(dataset)
            self.train_ppi_dataset, self.validation_ppi_dataset = split_dataset(
                dataset, args)

        if not args.h5_file == '':
            print('ABAg')
            shared_arguments = get_ppi_dataset_setup(args, dataset_name='abag')
            dataset = H5AbAgPPIMaskedMultiAtomDatasetBatched(
                args.h5_file, **shared_arguments)
            #print(dataset)
            self.train_abag_dataset, self.validation_abag_dataset = split_dataset(
                dataset, args)

        if not args.h5_file_abag_ppi == '':
            # Load this abag dataset like a ppi dataset
            shared_arguments = get_ppi_dataset_setup(args, dataset_name='ppi')
            dataset = H5AbAgPPIMaskedMultiAtomDatasetBatched(
                args.h5_file_abag_ppi, **shared_arguments)
            #print(dataset)
            self.train_abag_ppi_dataset, self.validation_abag_ppi_dataset = split_dataset(
                dataset, args)

        if not args.h5_file_ab == '':
            print('Ab dataset')
            shared_arguments = get_ppi_dataset_setup(args)
            # Load this abag dataset like a ppi dataset
            if args.mask_ab_region != '':
                shared_arguments.update(dict(mask_ab_region=args.mask_ab_region))
            dataset = AntibodyMaskedMultiAtomDatasetBatched(
                args.h5_file_ab, **shared_arguments)
            #print(dataset)
            self.train_ab_dataset, self.validation_ab_dataset = split_dataset(
                dataset, args)
            ab_train_dataset_list.append(self.train_ab_dataset)
            ab_val_dataset_list.append(self.validation_ab_dataset)
        if not args.h5_file_afab == '':
            print('AfAb dataset')
            # Load this abag dataset like a ppi dataset
            shared_arguments = get_ppi_dataset_setup(args)
            if args.mask_ab_region != '':
                shared_arguments.update(dict(mask_ab_region=args.mask_ab_region))
            dataset = AntibodyMaskedMultiAtomDatasetBatched(
                args.h5_file_afab, **shared_arguments)
            #print(dataset)
            self.train_afab_dataset, self.validation_afab_dataset = split_dataset(
                dataset, args)
            ab_train_dataset_list.append(self.train_afab_dataset)
            ab_val_dataset_list.append(self.validation_afab_dataset)
        if not args.h5_file_afsc == '':
            print('AfSC dataset')
            shared_arguments = get_ppi_dataset_setup(args)
            if args.mask_ab_region != '':
                shared_arguments.update(dict(mask_ab_region=args.mask_ab_region))
            # Load this abag dataset like a ppi dataset
            dataset = AntibodyMaskedMultiAtomDatasetBatched(
                args.h5_file_afsc, **shared_arguments)
            #print(dataset)
            self.train_afsc_dataset, self.validation_afsc_dataset = split_dataset(
                dataset, args)
            ab_train_dataset_list.append(self.train_afsc_dataset)
            ab_val_dataset_list.append(self.validation_afsc_dataset)

        if args.concat_ab_datasets:
            
            self.train_ab_all = torch.utils.data.ConcatDataset(ab_train_dataset_list)
            #self.val_ab_all = torch.utils.data.ConcatDataset(ab_val_dataset_list)
        
    def train_dataloader(self, with_metadata=False):
        trainloaders = {}
        args = self.cli_args

        if not args.h5_file_ppi == '':
            train_loader = _helper_loader(self.train_ppi_dataset, args, with_metadata=with_metadata)
            trainloaders.update(dict(ppi=train_loader))

        if not args.h5_file == '':
            train_loader = _helper_loader(self.train_abag_dataset, args, with_metadata=with_metadata)
            trainloaders.update(dict(abag=train_loader))

        if not args.h5_file_abag_ppi == '':
            train_loader = _helper_loader(self.train_abag_ppi_dataset, args, with_metadata=with_metadata)
            trainloaders.update(dict(abag_ppi=train_loader))

        if is_protein_dataset(args):
            if self.train_protein_dataset is None:
                print('Protein train set not found!')
                sys.exit()
            train_loader = _helper_loader(self.train_protein_dataset, args, with_metadata=with_metadata)
            trainloaders.update(dict(protein=train_loader))
        # works for multiple dataloaders?

        if args.concat_ab_datasets:
            train_loader = _helper_loader(self.train_ab_all, args, with_metadata=with_metadata)
            trainloaders.update(dict(ab=train_loader))
        else:
            if not args.h5_file_ab == '':
                train_loader = _helper_loader(self.train_ab_dataset, args, with_metadata=with_metadata)
                trainloaders.update(dict(ab=train_loader))

            if not args.h5_file_afab == '':
                train_loader = _helper_loader(self.train_afab_dataset, args, with_metadata=with_metadata)
                trainloaders.update(dict(afab=train_loader))

            if not args.h5_file_afsc == '':
                train_loader = _helper_loader(self.train_afsc_dataset, args, with_metadata=with_metadata)
                trainloaders.update(dict(afsc=train_loader))


        return trainloaders

    def val_dataloader(self, with_metadata=False):
        valloaders = {}
        args = self.cli_args
        if not args.h5_file_ppi == '':
            data_loader = _helper_loader(self.validation_ppi_dataset, args, with_metadata=with_metadata)
            valloaders.update(dict(ppi=data_loader))

        if not args.h5_file == '':
            data_loader = _helper_loader(self.validation_abag_dataset, args, with_metadata=with_metadata)
            valloaders.update(dict(abag=data_loader))

        if not args.h5_file_abag_ppi == '':
            data_loader = _helper_loader(self.validation_abag_ppi_dataset, args, with_metadata=with_metadata)
            valloaders.update(dict(abag_ppi=data_loader))

        if is_protein_dataset(args):
            if self.validation_protein_dataset is None:
                print('SCN file not found!')
                sys.exit()
            data_loader = _helper_loader(self.validation_protein_dataset, args, with_metadata=with_metadata)
            valloaders.update(dict(protein=data_loader))

        if not args.h5_file_ab == '':
            data_loader = _helper_loader(self.validation_ab_dataset, args, with_metadata=with_metadata)
            valloaders.update(dict(ab=data_loader))

        if not args.h5_file_afab == '':
            data_loader = _helper_loader(self.validation_afab_dataset, args, with_metadata=with_metadata)
            valloaders.update(dict(afab=data_loader))

        if not args.h5_file_afsc == '':
            data_loader = _helper_loader(self.validation_afsc_dataset, args, with_metadata=with_metadata)
            valloaders.update(dict(afsc=data_loader))

        print('Val loadrs:', valloaders.keys())
        combined_loaders = CombinedLoader(valloaders, "min_size")
        return combined_loaders
