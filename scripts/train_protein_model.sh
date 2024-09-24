#!/bin/bash -l

module unload python
module load cuda/11.1.0
module load python/3.9.0
module load git

module list

source /home/smahaja4/repositories/clone_masked_model/venv_py39_torch19/bin/activate
export PYTHONPATH="$PYTHONPATH:/home/smahaja4/repositories/clone_masked_model/venv_py39_torch19:/home/smahaja4/repositories/clone_masked_model/venv_py39_torch19/bin:/scratch16/jgray21/smahaja4_active/repositories/240519/MaskedProteinEnT:/scratch16/jgray21/smahaja4_active/repositories/240519/MaskedProteinEnT/src"


LAYERS=4
HEADS=8
DIM=256
OLD=0

BS=1 #$1
SS=50 #protein model was trained on 90ss
save_every=5
gmodel=egnn-trans-ma
atom_types=backbone_and_cb

#TRAIN
BASENAME_SCRATCH=/scratch16/jgray21/smahaja4_active/pooja-ppi-masked-sequence/trained_models
NN=48
gd2_dataset_ids=/scratch16/jgray21/smahaja4_active/datasets/nredundant_train_test_lists_New/ids_train_casp12nr50_nr70Ig_nr40Others.fasta
#### procs, gpus ###############
n_proc=$SLURM_NTASKS_PER_NODE
num_gpus=0
env | grep -a SLURM | tee slurm_env
qu="a100"
if [ "$SLURM_JOB_PARTITION" = "$qu"  ]; then
  IFS=','
  read -a strarr <<< "$SLURM_STEP_GPUS"
  num_gpus=${#strarr[*]}
fi

if [ ! -f /tmp/sidechainnet_casp12_50.pkl ]
then
        cp /scratch16/jgray21/smahaja4_active/datasets/sidechainnet_c12_ss50/sidechainnet_casp12_50.pkl /tmp/.
fi

SEED=1
MODELS_DIR=/scratch16/jgray21/smahaja4_active/pooja-ppi-masked-sequence/trained_models_seeds/protein_egnn-trans-ma_masked/h8_l4_d50_nn48_gd3_masif_new_backbone_and_cb_wpi_protein/20240527/seeds_$SEED
EPOCHS=10
date
#SEEDS RUN
cd /scratch16/jgray21/smahaja4_active/repositories/240519/MaskedProteinEnT
which python3
python3 /scratch16/jgray21/smahaja4_active/repositories/240519/MaskedProteinEnT/train_masked_model.py \
 --save_every ${save_every}  --lr 0.00001  --batch_size $BS  --heads $HEADS  --model_dim $DIM  --epochs $EPOCHS  --dropout 0.2 --masking_rate_max 0.25 --topk_metrics 1 --layers $LAYERS --num_gpus $num_gpus --crop_sequences --scn_sequence_similarity ${SS} --protein_gmodel $gmodel --lr_patience 350 --lr_cooldown 20 --max_ag_neighbors ${NN}  --atom_types ${atom_types}  --file_with_selected_scn_ids_for_training ${gd2_dataset_ids} --lightning_save_last_model  --use_scn --num_proc $n_proc --output_dir ${MODELS_DIR} --seed $SEED --wandb_entity saipooja

