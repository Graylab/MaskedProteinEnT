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

from src.command_line_utils import _get_args


stamp = datetime.now().strftime("%Y%m%d%H%M")

sys.stderr = open('err_{}.txt'.format(stamp), 'a')
sys.stderr.write("\n________\nSTART\n ")
sys.stderr.write("\n" + str(datetime.now()) + "\n")


def get_date_stamped_id():
    s = datetime.now()
    stamp = s.strftime("%Y%m%d%H%M")
    return stamp


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
