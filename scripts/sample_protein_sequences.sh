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

MODEL=../trained_models/ProtEnT_backup.ckpt
TEST_RESULTS_BASE=/scratch16/jgray21/smahaja4_active/tmp/
PDB_DIR=/scratch16/jgray21/smahaja4_active/repositories/MaskedProteinEnT/data/proteins
python3 /scratch16/jgray21/smahaja4_active/repositories/MaskedProteinEnT/ProteinSequenceSampler.py  \
	--output_dir ${TEST_RESULTS_BASE} \
	--num_gpus $num_gpus \
	--num_procs $n_proc \
	--model $MODEL \
	--from_pdb $PDB_DIR \
	--sample_temperatures 0.2,0.5 \
	--num_samples 100

