#!/bin/bash

module load cuda/11.1.0

### MODEL AND TRAINING PARAMETERS
LAYERS=4
HEADS=8
DIM=256
OLD=0
BS=1
SS=50 # protein model was trained on 90ss
save_every=5
gmodel=egnn-trans-ma
atom_types=backbone_and_cb
NN=48

### WANDB ENTITY
WANDB_ENTITY="fadh-johns-hopkins-university"
if [ "$WANDB_ENTITY" = "YOUR_WANDB_ENTITY" ]; then
  echo "Error: Please set your WANDB_ENTITY variable."
  exit 1
fi

### CHECK AND CREATE DATASETS DIRECTORY
if [ ! -d datasets ]; then
  mkdir training_datasets
fi

### DOWNLOAD DATASET IF NOT EXISTS
if [ ! -f training_datasets/ids_train_casp12nr50_nr70Ig_nr40Others.fasta ]; then
  wget -P training_datasets https://zenodo.org/records/13831403/files/ids_train_casp12nr50_nr70Ig_nr40Others.fasta
fi

gd2_dataset_ids=$(pwd)/training_datasets/ids_train_casp12nr50_nr70Ig_nr40Others.fasta

### SCRATCH DIRECTORY, DATASET AND GPU SETUP
# n_proc=$SLURM_NTASKS_PER_NODE ### Ran into issues while doing troubleshooting on interactive node
n_proc=$(( $SLURM_NTASKS / $SLURM_JOB_NUM_NODES ))

num_gpus=0
# env | grep -a SLURM | tee slurm_env
qu="a100"
if [ "$SLURM_JOB_PARTITION" = "$qu"  ]; then
  IFS=','
  read -a strarr <<< "$SLURM_STEP_GPUS"
  num_gpus=${#strarr[*]}
fi

### COPY DATASET TO LOCAL TEMPORARY DIRECTORY
if [ ! -f /tmp/sidechainnet_casp12_50.pkl ]
then
    cp /scratch16/jgray21/smahaja4_active/datasets/sidechainnet_c12_ss50/sidechainnet_casp12_50.pkl /tmp/.
fi

### Training seed and output directory
SEED=1
### Maybe need to change output model directory
MODELS_DIR=out_dir_seed_$SEED
EPOCHS=10

# date

### RUNNING TRAINING SCRIPT
# echo "this is n_proc var $n_proc" ### Debug interactive node

python3 train_masked_model.py \
  --save_every ${save_every}  --lr 0.00001  --batch_size $BS  \
  --heads $HEADS  --model_dim $DIM  --epochs $EPOCHS  --dropout 0.2 \
  --masking_rate_max 0.25 --topk_metrics 1 --layers $LAYERS \
  --num_gpus $num_gpus --crop_sequences \
  --scn_sequence_similarity ${SS} --protein_gmodel $gmodel \
  --lr_patience 350 --lr_cooldown 20 --max_ag_neighbors ${NN}  \
  --atom_types ${atom_types}  \
  --file_with_selected_scn_ids_for_training ${gd2_dataset_ids} \
  --lightning_save_last_model  --use_scn --num_procs $n_proc \
  --output_dir ${MODELS_DIR} --seed $SEED --wandb_entity $WANDB_ENTITY
