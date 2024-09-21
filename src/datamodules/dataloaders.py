from src.data.datamodules.utils \
import split_dataset, get_protein_dataset_setup, get_ppi_dataset_setup, _helper_loader
from src.data.datasets.AntibodyMaskedMultiAtomDatasetBatched import AntibodyMaskedMultiAtomDatasetBatched
from src.data.datasets.H5AbAgPPIMaskedMultiAtomDatasetBatched import H5AbAgPPIMaskedMultiAtomDatasetBatched
from src.data.datasets.ProteinMaskedMultiAtomDatasetBatched import ProteinMaskedMultiAtomDatasetBatched
from src.data.datasets.SCNProteinMaskedMultiAtomDatasetBatched import SCNProteinMaskedMultiAtomDatasetBatched


def get_dataloaders_ppi(args, dataset_model=None, 
                        with_metadata=False,
                        partner_selection='random'):
    
    shared_arguments = dict(max_mask=args.masking_rate_max,
                            min_mask=args.masking_rate_min,
                            partner_selection=partner_selection)
    if dataset_model is None:
        dataset_model =  args.protein_gmodel
    
    def setup_dataloader(h5_file, dataset_label):
        shared_arguments.update(dict(dataset=dataset_label))

        if dataset_model in ['egnn-trans-ma-ppi']:
            dataset = H5AbAgPPIMaskedMultiAtomDatasetBatched(
                h5_file, **shared_arguments)
            #print(dataset)
            train_dataset, validation_dataset = split_dataset(
                dataset, args)
            train_loader = None
            validation_loader = None
            if train_dataset.__len__() > 0:
                train_loader = _helper_loader(train_dataset, args, with_metadata=with_metadata)
            if validation_dataset.__len__() > 0:
                validation_loader = _helper_loader(validation_dataset, args, with_metadata=with_metadata)

        return train_loader, validation_loader

    if not args.h5_file_ppi == '':
        train_loader, validation_loader = setup_dataloader(args.h5_file_ppi, 'ppi')
    elif not args.h5_file == '':
        train_loader, validation_loader = setup_dataloader(args.h5_file, 'abag')
    else:
        sys.exit('Failed to load dataset')
    
    return train_loader, validation_loader


def get_dataloader_for_testing(mr=0.0, mr_min=0.0, mr_noncontact=0.0,
                               partner_selection='random', 
                               mask_indices=None,
                               with_metadata=True,
                               val_split=1,
                               region_selection=None,
                               intersect_with_contacts=False):
    
    # load ppi dataset
    args.train_split = 1-val_split
    args.masking_rate_min = mr_min
    args.masking_rate_max = mr
    
    if args.h5_file != '' or args.h5_file_ppi != '':
        args.masking_rate_min = mr_min
        args.masking_rate_max = mr
        _, val_dataloader = get_dataloaders_ppi(args, with_metadata=with_metadata)
        val_dataloader.dataset.dataset.noncontact_mask_rate = mr_noncontact
        val_dataloader.dataset.dataset.intersect_contact_and_region = intersect_with_contacts
        assert val_dataloader.dataset.dataset.percent_mask[0] == mr
        assert val_dataloader.dataset.dataset.percent_mask[1] == mr_min
        assert val_dataloader.dataset.dataset.noncontact_mask_rate == mr_noncontact
        val_dataloader.dataset.dataset.partner_selection = partner_selection
        assert val_dataloader.dataset.dataset.partner_selection == partner_selection
        val_dataloader.dataset.dataset.mask_ab_region = region_selection
        val_dataloader.dataset.dataset.mask_ab_indices = mask_indices
        return val_dataloader
    
    if not args.h5_file_ab == '':
        shared_arguments = get_ppi_dataset_setup(args)
        # Load this abag dataset like a ppi dataset
        shared_arguments.update(dict(metadata_extra=True))
        dataset = AntibodyMaskedMultiAtomDatasetBatched(
            args.h5_file_ab, **shared_arguments)
        _, val_dataset = split_dataset(dataset, args)
        val_dataloader = _helper_loader(val_dataset, args, with_metadata=with_metadata)
        val_dataloader.dataset.dataset.masking_rate_max = mr
        val_dataloader.dataset.dataset.masking_rate_min = mr_min
        val_dataloader.dataset.dataset.partner_selection = partner_selection
        val_dataloader.dataset.dataset.mask_ab_region = region_selection
        assert val_dataloader.dataset.dataset.masking_rate_max == mr
        assert val_dataloader.dataset.dataset.masking_rate_min == mr_min
        val_dataloader.dataset.dataset.mask_ab_indices = mask_indices
        return val_dataloader
    
    if not args.h5_file_protein == '':
        shared_arguments = get_protein_dataset_setup(args)
        # Load this abag dataset like a ppi dataset
        dataset = ProteinMaskedMultiAtomDatasetBatched(
            args.h5_file_protein, **shared_arguments)
        _, val_dataset = split_dataset(dataset, args)
        val_dataloader = _helper_loader(val_dataset, args, with_metadata=with_metadata)
        val_dataloader.dataset.dataset.masking_rate_max = mr
        assert val_dataloader.dataset.dataset.masking_rate_max == mr
        return val_dataloader
    
    casp_version = args.scn_casp_version
    thinning = args.scn_sequence_similarity
    scn_path = '/scratch16/jgray21/smahaja4_active/datasets/sidechainnet'
    input_file = '{}_c{}_ss{}/sidechainnet_casp{}_{}.pkl'.format(
        scn_path, casp_version, thinning, casp_version, thinning)
    shared_arguments = get_protein_dataset_setup(args)
    tmp_input_file = '/tmp/{}'.format(os.path.basename(input_file))
    if os.path.exists(tmp_input_file):
        input_file = tmp_input_file
    
    if os.path.exists(input_file):
        with open(input_file, "rb") as f:
            d = pickle.load(f)

        validation_dataset_keys = [
                        key for key in d.keys() if key.find('valid') != -1
                    ]
        validation_protein_dataset = SCNProteinMaskedMultiAtomDatasetBatched(d, validation_dataset_keys,
                                                                **shared_arguments)
        data_loader = _helper_loader(validation_protein_dataset, args, with_metadata=with_metadata)
        return data_loader

