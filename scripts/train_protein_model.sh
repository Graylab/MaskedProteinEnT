#!/bin/bash -l

source <YOUR ENVIRONMENT>
### MUST be set up #####
WANDB_NAME=<YOUR WANDB NAME>

#Default settings
LAYERS=4
HEADS=8
DIM=256
BS=1
SS=50 
save_every=5
gmodel=egnn-trans-ma
atom_types=backbone_and_cb
NN=48
gd2_dataset_ids=ids_train_casp12nr50_nr70Ig_nr40Others.fasta

#### procs, gpus ###############
n_proc=6
num_gpus=1

#### training #######
SEED=1
MODELS_DIR=models
EPOCHS=100

date
which python3 #MAKE SURE THIS MATCHES THE INSTALLED ENV
python3 train_masked_model.py \
 --save_every ${save_every}  --lr 0.00001  --batch_size $BS  --heads $HEADS  --model_dim $DIM  --epochs $EPOCHS  --dropout 0.2 --masking_rate_max 0.15 --topk_metrics 1 --layers $LAYERS --num_gpus $num_gpus --crop_sequences --scn_sequence_similarity ${SS} --protein_gmodel $gmodel --lr_patience 350 --lr_cooldown 20 --max_ag_neighbors ${NN}  --atom_types ${atom_types}  --file_with_selected_scn_ids_for_training ${gd2_dataset_ids} --lightning_save_last_model  --use_scn --num_proc $n_proc --output_dir ${MODELS_DIR} --seed $SEED --wandb_entity ${WANDB_NAME}

