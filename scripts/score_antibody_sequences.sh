#!/bin/bash -l

module unload python
module load cuda/11.1.0
module load python/3.9.0
module load git

source /home/smahaja4/repositories/clone_masked_model/venv_py39_torch19/bin/activate

export OMP_NUM_THREADS=12
gmodel=egnn-trans-ma-ppi

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
##############################

MODEL=../trained_models/AbPlusEnT_backup.ckpt
TEST_RESULTS_BASE=/scratch16/jgray21/smahaja4_active/tmp_ab/
PDB_FILE=/scratch16/jgray21/smahaja4_active/repositories/MaskedProteinEnT/data/antibodies/1n8z_trunc.pdb
SFILE=
python3 /scratch16/jgray21/smahaja4_active/repositories/MaskedProteinEnT/score_sequences.py  \
	--model $MODEL \
	--sequences_file $SFILE \
        --pdb_file $PDB_FILE \
	--output_dir $TEST_RESULTS_BASE

