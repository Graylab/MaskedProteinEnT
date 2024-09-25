#!/bin/bash
# removed -l to avoid loading bash profile stuff

# module unload python
module load cuda/11.1.0
# module load python/3.9.0
# module load git

# module list

# source /home/smahaja4/repositories/clone_masked_model/venv_py39_torch19/bin/activate
# export PYTHONPATH="$PYTHONPATH:/home/smahaja4/repositories/clone_masked_model/venv_py39_torch19:/home/smahaja4/repositories/clone_masked_model/venv_py39_torch19/bin:/scratch16/jgray21/smahaja4_active/repositories/240519/MaskedProteinEnT:/scratch16/jgray21/smahaja4_active/repositories/240519/MaskedProteinEnT/src"

### MODEL AND TRAINING PARAMETERS
LAYERS=4
HEADS=8
DIM=256
OLD=0
BS=1
SS=50 #protein model was trained on 90ss
save_every=5
gmodel=egnn-trans-ma
atom_types=backbone_and_cb
NN=48 

### SCRATCH DIRECTORY, DATASET AND GPU SETUP
# BASENAME_SCRATCH=/scratch16/jgray21/smahaja4_active/pooja-ppi-masked-sequence/trained_models ### Is this needed?
gd2_dataset_ids=/scratch16/jgray21/smahaja4_active/datasets/nredundant_train_test_lists_New/ids_train_casp12nr50_nr70Ig_nr40Others.fasta

### SLURM GPU DETECTION
#### procs, gpus ###############
# n_proc=$SLURM_NTASKS_PER_NODE ### Ran into issues while doing troubleshooting on interactive node
n_proc=$(( $SLURM_NTASKS / $SLURM_JOB_NUM_NODES ))


num_gpus=0
env | grep -a SLURM | tee slurm_env
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
date

### RUNNING TRAINING SCRIPT
#SEEDS RUN
# cd /scratch16/jgray21/smahaja4_active/repositories/240519/MaskedProteinEnT
# which python3
echo "this is n_proc var $n_proc"

# python3 /scratch16/jgray21/smahaja4_active/repositories/240519/MaskedProteinEnT/train_masked_model.py \
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
  --output_dir ${MODELS_DIR} --seed $SEED --wandb_entity fadh-johns-hopkins-university
