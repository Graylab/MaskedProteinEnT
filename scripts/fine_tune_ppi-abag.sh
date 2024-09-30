#!/bin/bash -l

source <YOUR ENVIRONMENT>

### MUST be set up ####
### WANDB ENTITY ###
WANDB_ENTITY="YOUR_WANDB_ENTITY"
if [ "$WANDB_ENTITY" = "YOUR_WANDB_ENTITY" ]; then
  echo "Error: Please set your WANDB_ENTITY variable."
  exit 1
fi

### DOWNLOADING TRAINING DATASETS IF NECESSARY ###
mkdir -p training_datasets
if [ ! -f training_datasets/ids_train_casp12nr50_nr70Ig_nr40Others.fasta ]; then
 wget -P training_datasets https://zenodo.org/records/13831403/files/ids_train_casp12nr50_nr70Ig_nr40Others.fasta
fi
gd2_dataset_ids=training_datasets/ids_train_casp12nr50_nr70Ig_nr40Others.fasta

# DOWNLOAD SEPARATELY IF NECESSARY 
if [ ! -f training_datasets/sidechainnet_casp12_50.pkl ]; then
  wget -P training_datasets https://zenodo.org/records/13831403/files/sidechainnet_casp12_50.pkl
fi

if [ ! -f training_datasets/AbSCSAbDAb_trainnr90_bkandcbcoords_aug2022.h5 ]; then
  wget -P training_datasets https://zenodo.org/records/13831403/files/AbSCSAbDAb_trainnr90_bkandcbcoords_aug2022.h5
fi

if [ ! -f training_datasets/ppi_trainset_5032_noabag_aug2022.h5 ]; then
  wget -P training_datasets https://zenodo.org/records/13831403/files/ppi_trainset_5032_noabag_aug2022.h5
fi

### CHECK/DOWNLOAD->DECOMPRESS PRETRAINED MODEL ###
mkdir -p models
if [ ! -f models/model.tar.gz ] && [ ! -f models/ProtEnT_backup.ckpt ]; then
  wget -P models https://zenodo.org/records/8313466/files/model.tar.gz
fi
tar --skip-old-files --strip-components=1 -C models -xvzf models/model.tar.gz models/ProtEnT_backup.ckpt

### procs, gpus ###
n_proc=12
num_gpus=1

### Default settings ###
LAYERS=4
HEADS=8
DIM=256
BS=1
SS=50 
save_every=5
gmodel=egnn-trans-ma
atom_types=backbone_and_cb
NN=48

if [ ! -f /tmp/AbSCSAbDAb_trainnr90_bkandcbcoords_aug2022.h5 ]; then
  cp training_datasets/AbSCSAbDAb_trainnr90_bkandcbcoords_aug2022.h5 /tmp/.
fi

if [ ! -f /tmp/ppi_trainset_5032_noabag_aug2022.h5 ]; then
  cp training_datasets/ppi_trainset_5032_noabag_aug2022.h5 /tmp/.
fi

if [ ! -f /tmp/trainset_highres_nr90_vhh-rabd-dms_abnr90agnr70_aug2022.h5 ]
then
        cp /scratch16/jgray21/smahaja4_active/datasets/trainset_highres_nr90_vhh-rabd-dms_abnr90agnr70_aug2022.h5 /tmp/.
fi

scn_dataset_path='training_datasets'

H5_FILE_PPI=/tmp/ppi_trainset_5032_noabag_aug2022.h5
# H5_FILE_AbAg=/tmp/AbSCSAbDAb_trainnr90_bkandcbcoords_aug2022.h5
H5_FILE_AbAg=/tmp/trainset_highres_nr90_vhh-rabd-dms_abnr90agnr70_aug2022.h5

### training ###
SEED=1
MODELS_DIR=models
EPOCHS=600
MODEL=models/ProtEnT_backup.ckpt
date
which python3 #MAKE SURE THIS MATCHES THE INSTALLED ENV
python3 train_masked_model.py \
 --save_every ${save_every}  --lr 0.00001  --batch_size $BS  --heads $HEADS  --model_dim $DIM  --epochs $EPOCHS  --dropout 0.2 --masking_rate_max 0.15 --topk_metrics 1 --layers $LAYERS --num_gpus $num_gpus --crop_sequences --scn_sequence_similarity ${SS} --protein_gmodel $gmodel --lr_patience 350 --lr_cooldown 20 --max_ag_neighbors ${NN}  --atom_types ${atom_types}  --file_with_selected_scn_ids_for_training ${gd2_dataset_ids} --lightning_save_last_model  --use_scn --num_proc $n_proc --output_dir ${MODELS_DIR} --seed $SEED --wandb_entity ${WANDB_ENTITY} --model $MODEL --h5_file_ppi $H5_FILE_PPI --h5_file $H5_FILE_AbAg --fine_tune --scn_path ${scn_dataset_path}
