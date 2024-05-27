import torch
import os
import pytorch_lightning as pl
import sys
import json
from datetime import datetime

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

#model
from src.models.ProteinMaskedLabelModel.ProteinMaskedLabelModel_EnT_MA \
     import ProteinMaskedLabelModel_EnT_MA

#datamodule
from src.dataloaders.MaskedSequenceStructureMADataModule \
    import MaskedSequenceStructureMADataModule

#datasets
from sidechainnet.utils.organize import get_proteinnetIDs_by_split
from src.datasets.SCNProteinMaskedDatasetBatched\
    import SCNProteinMaskedDatasetBatched
from src.datasets.H5AbAgPPIMaskedMultiAtomDatasetBatched \
    import H5AbAgPPIMaskedMultiAtomDatasetBatched

from src.models.ProteinMaskedLabelModel.utils.command_line_utils\
     import split_dataset, _get_args


stamp = datetime.now().strftime("%Y%m%d%H%M")

sys.stderr = open('err_{}.txt'.format(stamp), 'a')
sys.stderr.write("\n________\nSTART\n ")
sys.stderr.write("\n" + str(datetime.now()) + "\n")


def get_date_stamped_id():
    s = datetime.now()
    stamp = s.strftime("%Y%m%d%H%M")
    return stamp


def get_dataloaders_ppi(args, dataset_model=None, 
                        with_metadata=False,
                        partner_selection='random'):
    
    shared_arguments = dict(num_bins=args.num_bins,
                            onehot_prim=(not args.no_onehot_prim),
                            add_loop_features=args.add_loop_features,
                            add_esm_embeddings=args.add_esm_embeddings,
                            esm_embeddings_ab=args.esm_embeddings_ab,
                            esm_embeddings_ag=args.esm_embeddings_ag,
                            add_additional_features=args.add_additional_features,
                            num_dist_bins_int=args.num_dist_bins_int,
                            dist_bin_interval=args.dist_interval,
                            topk_ab=args.max_ab_neighbors,
                            topk_ag=args.max_ag_neighbors,
                            rbf_d_count=args.rbf_count,
                            real_dist=args.real_distance,
                            inverted_real_dist= \
                                (not args.no_real_valued_inverted_distance),
                            max_dist_value=args.real_distance_max,
                            min_dist_value=args.real_distance_min,
                            pe_edges=args.model_dim,
                            pe_nodes=args.model_dim,
                            include_orientations=(not args.no_orientations),
                            include_bb=args.include_bb,
                            threshold=args.max_dist_connected_graph,
                            max_mask=args.masking_rate_max,
                            min_mask=args.masking_rate_min,
                            fully_connected=args.full_connectivity,
                            flip_seq_labels=args.flip_seq_labels,
                            gmodel=args.protein_gmodel,
                            partner_selection=partner_selection)
    if dataset_model is None:
        dataset_model =  args.protein_gmodel
    
    def setup_dataloader(h5_file, dataset_label):
        shared_arguments.update(dict(dataset=dataset_label))
        if dataset_model in ['egnn-trans-ma-ppi']:
            from deeph3.models.ProteinMaskedLabelModel.dataloaders.MaskedSequenceStructureMADataModule\
                 import _helper_loader
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


def get_dataloaders(args):

    shared_arguments = dict(max_seq_len=args.max_seq_len,
                            num_bins=args.num_bins,
                            onehot_prim=(not args.no_onehot_prim),
                            add_loop_features=args.add_loop_features,
                            add_esm_embeddings=args.add_esm_embeddings,
                            esm_embeddings_ab=args.esm_embeddings_ab,
                            esm_embeddings_ag=args.esm_embeddings_ag,
                            add_additional_features=args.add_additional_features,
                            num_dist_bins_int=args.num_dist_bins_int,
                            dist_bin_interval=args.dist_interval,
                            topk_ab=args.max_ab_neighbors,
                            topk_ag=args.max_ag_neighbors,
                            rbf_d_count=args.rbf_count,
                            real_dist=args.real_distance,
                            inverted_real_dist= \
                                (not args.no_real_valued_inverted_distance),
                            max_dist_value=args.real_distance_max,
                            min_dist_value=args.real_distance_min,
                            pe_edges=args.model_dim,
                            pe_nodes=args.model_dim,
                            include_orientations=(not args.no_orientations),
                            include_bb=args.include_bb,
                            threshold=args.max_dist_connected_graph,
                            max_mask=args.masking_rate_max,
                            min_mask=args.masking_rate_min,
                            fully_connected=args.full_connectivity,
                            flip_seq_labels=args.flip_seq_labels,
                            crop=args.crop_sequences,
                            max_train=args.max_train,
                            gmodel=args.protein_gmodel)

    casp_version = args.scn_casp_version
    thinning = args.scn_sequence_similarity
    scn_path = f'{args.scn_dataset_path}/sidechainnet'
    input_file = '{}_c{}_ss{}/sidechainnet_casp{}_{}.pkl'.format(
        scn_path, casp_version, thinning, casp_version, thinning)
    if not os.path.exists(input_file):
        print('File not found: ', input_file)
        sys.exit()

    import pickle
    with open(input_file, "rb") as f:
        d = pickle.load(f)

    ids = get_proteinnetIDs_by_split(casp_version, thinning)

    train_dataset = SCNProteinMaskedDatasetBatched(d, ['train'], selected_ids_file=args.file_with_selected_scn_ids_for_training,
                                                   **shared_arguments)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=args.shuffle_dataset,
        collate_fn=SCNProteinMaskedDatasetBatched.merge_samples_to_minibatch,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=6)
    print(d.keys())

    validation_dataset_keys = [
        key for key in d.keys() if key.find('valid') != -1
    ]

    val_dataset = SCNProteinMaskedDatasetBatched(d, validation_dataset_keys,
                                                 **shared_arguments)
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=SCNProteinMaskedDatasetBatched.merge_samples_to_minibatch,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=6)

    test_dataset = SCNProteinMaskedDatasetBatched(d, ['test'],
                                                  **shared_arguments)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=SCNProteinMaskedDatasetBatched.merge_samples_to_minibatch,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=6)

    return train_loader, validation_loader, test_loader


def get_project_name():
    return 'MaskedProteinMAENTransformer'


def get_datamodule(args):
    return MaskedSequenceStructureMADataModule(args)


def get_model(hyperparams_model):
    return ProteinMaskedLabelModel_EnT_MA(**hyperparams_model)


def load_model(update_hyperparams, in_model):
    return ProteinMaskedLabelModel_EnT_MA.load_from_checkpoint(in_model,
                                                                **update_hyperparams)


def get_hyperparamters(args):
    n_labels = 20 if not args.add_esm_embeddings else 1280
    n_labels_score = 16
    mode = 'backbone_and_cb' #args.atom_types
    natoms = 4
    hyperparams_model = dict(base_model_ckpt=args.masked_protein_base_model,
                             n_hidden=args.model_dim,
                             dropout=args.dropout,
                             n_heads=args.heads,
                             n_labels=n_labels,
                             n_labels_score=n_labels_score,
                             layers=args.layers,
                             top_k_metrics=args.topk_metrics,
                             weight_decay=args.weight_decay,
                             max_seq_len=args.max_seq_len,
                             max_neighbors=args.max_ag_neighbors,
                             predict_from_esm=(args.predict_from_esm
                                               and args.add_esm_embeddings),
                             thinning=args.scn_sequence_similarity,
                             natoms=natoms,
                             min_epoch=args.start_epoch_prop,
                             checkpoint=args.internal_checkpoints,
                             lr_scheduler=args.scheduler)

    return hyperparams_model


def setup_and_train_ppi_entransformer_all(args, gpu_setup, gmodel):
    
    project_name = get_project_name()
    config_run = dict(architechture=project_name, output_dir=args.output_dir)
    hyperparams_model = get_hyperparamters(args)
    hyperparams_model.update(config_run)
    hyperparams_model.update(dict(init_lr=args.lr))

    lr_scheduler_config = {
        'patience': args.lr_patience,
        'cooldown': args.lr_cooldown
    }
    hyperparams_model.update(dict(lr=lr_scheduler_config))
    hyperparams_model.update(dict(scheduler=args.scheduler))

    #set up model and wandb
    if args.model == '' or args.fine_tune:
        if not args.fine_tune:
            pl.seed_everything(args.seed, workers=True)
        else:
            assert os.path.exists(args.model)
        datestamped_id = get_date_stamped_id()
        wlogger_init = dict(project=project_name,
                            entity='saipooja',
                            config=config_run,
                            id=datestamped_id)
        out_dir = os.path.join(args.output_dir, datestamped_id)
        if not os.path.isdir(out_dir):
                print('Making {} ...'.format(out_dir))
                os.makedirs(out_dir, exist_ok=True)
        open("{}/wandbrunid".format(out_dir),
             'w').write(datestamped_id)
    else:
        assert os.path.exists(args.model)
        assert os.path.exists("{}/wandbrunid".format(args.output_dir))
        datestamped_id = open("{}/wandbrunid".format(args.output_dir),
                            'r').read()
        wlogger_init = dict(project=project_name,
                            #reinit=True,
                            resume="must",
                            entity='saipooja',
                            config=config_run,
                            id=datestamped_id.rstrip())
        out_dir = args.output_dir

    outfile_json_args = os.path.join(out_dir, 'log_run_args.json')
    if not os.path.exists(outfile_json_args):
        
        args_dict = vars(args)
        open(outfile_json_args, 'w').write(json.dumps(args_dict))
    
    try:
        wblogger = pl.loggers.WandbLogger(**wlogger_init)
    except:
        wblogger = None
    os.environ['WANDB_DIR'] = out_dir
    
    # Checkpointing
    callbacks_list = []
    filename = "model.p.e{epoch}"
    callback_checkpoint_every = ModelCheckpoint(dirpath=out_dir,
                                          filename=filename,
                                          every_n_epochs=args.save_every,
                                          save_on_train_epoch_end=True
                                          )
    filename_best = "vallossmin_model.p.e{epoch}.s{step}"
    callback_checkpoint_best = ModelCheckpoint(dirpath=out_dir,
                                          filename=filename_best,
                                          save_top_k=3,
                                          monitor="val_loss",
                                          mode='min')
    filename_best_train = "trainlossmin_model.p.e{epoch}.s{step}"
    callback_checkpoint_best_train = ModelCheckpoint(dirpath=out_dir,
                                          filename=filename_best_train,
                                          save_top_k=3,
                                          monitor="train_loss",
                                          mode='min')
    callbacks_list = [callback_checkpoint_best, callback_checkpoint_every,
                     callback_checkpoint_best_train]
    if args.lightning_save_last_model and (args.save_every != 1):
        callback_checkpoint_last = ModelCheckpoint(dirpath=out_dir,
                                                   save_last=args.lightning_save_last_model,
                                                   save_on_train_epoch_end=True
                                                    )
        callbacks_list.append(callback_checkpoint_last)
    callback_lr = LearningRateMonitor(logging_interval='epoch')
    callbacks_list.append(callback_lr)

    datamodule = get_datamodule(args, gmodel)
    trainer_config = dict(max_epochs=args.epochs, accumulate_grad_batches=8)
    if args.model != '':
        trainer_config.update(dict(resume_from_checkpoint=args.model))

    gradient_clip_value = 0.0
    if args.clip_gradients:
        gradient_clip_value = 0.5
    trainer = pl.Trainer(logger=wblogger,
                         callbacks=callbacks_list,
                         **trainer_config,
                         **gpu_setup,
                         precision=64,
                         #deterministic=True,
                         multiple_trainloader_mode='min_size',
                         gradient_clip_val=gradient_clip_value,
                         detect_anomaly=args.lightning_detect_anomaly)
    if not wblogger is None:
        wblogger.log_hyperparams(hyperparams_model)

    if args.model == '':
        model = get_model(hyperparams_model, gmodel,
                          protein_base_model_ckpt=args.masked_protein_base_model)
    else:
        # read learning rate from command line args
        update_hyperparams = dict(init_lr=args.lr)
        lr_scheduler_config = {
                                'patience': args.lr_patience,
                                'cooldown': args.lr_cooldown
                                }
        update_hyperparams.update(dict(lr=lr_scheduler_config))
        model = load_model(update_hyperparams, args.model, gmodel,
                          protein_base_model_ckpt=args.masked_protein_base_model)
        assert model.init_lr == args.lr

    if args.model == '':
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.model)


def _cli():
    """Command line interface for training/fine-tuning masked models"""

    args = _get_args()
    # gpu/mulit-gpu setup
    gpus = args.num_gpus if torch.cuda.is_available() else 0
    gpu_args = dict(auto_select_gpus=False, gpus=gpus)
    if gpus > 0:
        gpu_args["auto_select_gpus"] = True
    if gpus > 1:
        gpu_args["accelerator"] = "ddp"

    setup_and_train_ppi_entransformer_all(args, gpu_args, args.protein_gmodel)


if __name__ == '__main__':
    _cli()
