#!/bin/bash -l

module unload python
module load cuda/11.1.0
module load python/3.9.0
module load git

module list

source /home/smahaja4/repositories/clone_masked_model/venv_py39_torch19/bin/activate


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


MODEL=../trained_models/ProtAbAgEnT_backup.ckpt
TEST_RESULTS_BASE=/scratch16/jgray21/smahaja4_active/tmp_abag/
PDB_DIR=/scratch16/jgray21/smahaja4_active/repositories/MaskedProteinEnT/data/abag/
PPI_PARTNERS_DICT=/scratch16/jgray21/smahaja4_active/repositories/MaskedProteinEnT/data/abag/1n8z_partners.json
python3 /scratch16/jgray21/smahaja4_active/repositories/MaskedProteinEnT/PPIAbAgSequenceSampler.py  \
        --output_dir ${TEST_RESULTS_BASE} \
        --num_gpus $num_gpus \
        --num_procs $n_proc \
        --model $MODEL \
        --from_pdb $PDB_DIR \
	--sample_temperatures 0.2,0.5 \
       	--num_samples 100 \
	--partners_json ${PPI_PARTNERS_DICT} \
	--partner_name Ab \
        --antibody
 

