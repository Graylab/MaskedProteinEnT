import torch
from src.data.datasets.SCNProteinMaskedMultiAtomDatasetBatched import SCNProteinMaskedMultiAtomDatasetBatched


def collate_function_getter(with_metadata=False):
    if not with_metadata:
        return SCNProteinMaskedMultiAtomDatasetBatched.merge_samples_to_minibatch
    else:
        return SCNProteinMaskedMultiAtomDatasetBatched.merge_samples_to_minibatch_with_metadata


def _helper_loader(dataset, args, with_metadata=False):
    return torch.utils.data.DataLoader(
    dataset,
    shuffle=args.shuffle_dataset,
    collate_fn=collate_function_getter(with_metadata=with_metadata),
    batch_size=args.batch_size,
    pin_memory=True,
    num_workers=args.num_procs)


def split_dataset(dataset,args):

    train_split_length = int(len(dataset) * args.train_split)
    assert train_split_length <= len(dataset)
    import random
    random.seed(args.seed)
    indices_data = list(range(0,len(dataset)))
    random.shuffle(indices_data)
    indices_val = indices_data[:len(dataset) - train_split_length]
    assert(len(indices_val)==len(set(indices_val)))
    indices_train = [t for t in range(len(dataset)) if t not in indices_val]

    if args.max_train is not None:
        indices_train = indices_train[:args.max_train]
    if args.max_val is not None:
        indices_val = indices_val[:args.max_val]

    sorted_indices = dataset.get_sorted_indices()
    sorted_val_indices = [i for i in sorted_indices if i in indices_val]
    sorted_train_indices = [i for i in sorted_indices if i in indices_train]
    assert(set(sorted_train_indices).intersection(set(sorted_val_indices)) == set())

    print("Effective dataset sizes: ", len(sorted_train_indices), len(sorted_val_indices),\
         len(sorted_train_indices) + len(sorted_val_indices))

    train_dataset = torchdata.Subset(dataset, indices=sorted_train_indices)
    validation_dataset = torchdata.Subset(dataset, indices=sorted_val_indices)
    return train_dataset, validation_dataset

def is_protein_dataset(args):
    if args.use_scn or args.h5_file_protein != '':
        return True
    elif (args.h5_file == '' and args.h5_file_ppi == ''
         and args.h5_file_ab == '' and args.h5_file_abag_ppi == '' and
         args.h5_file_afsc == '' and args.h5_file_afab == ''):
        return True
    else:
        # using scn is the default dataset
        return False

def get_protein_dataset_setup(args):
    shared_arguments = dict(max_seq_len=args.max_seq_len,
                            topk_ag=args.max_ag_neighbors,
                            max_mask=args.masking_rate_max,
                            crop=args.crop_sequences,
                            gmodel=args.protein_gmodel,
                            atom_mode=args.atom_types
                            )
    return shared_arguments
        


def get_ppi_dataset_setup(args, dataset_name=None):
    shared_arguments = dict(topk_ab=args.max_ab_neighbors,
                            topk_ag=args.max_ag_neighbors,
                            max_mask=args.masking_rate_max,
                            min_mask=args.masking_rate_min,
                            gmodel=args.protein_gmodel,
                            max_seq_len=args.max_seq_len
                            )
    if dataset_name is not None:
        shared_arguments.update(dict(dataset=dataset_name))
    return shared_arguments

