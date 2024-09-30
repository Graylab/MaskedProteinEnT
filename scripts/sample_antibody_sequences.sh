#!/bin/bash -l

source <YOUR ENVIRONMENT> 

### CHECK/DOWNLOAD->DECOMPRESS PRETRAINED MODEL ###
mkdir -p models
if [ ! -f models/model.tar.gz ] && [ ! -f models/AbPlusEnT_backup.ckpt ]; then
  wget -P models https://zenodo.org/records/8313466/files/model.tar.gz
fi
tar --skip-old-files --strip-components=1 -C models -xvzf models/model.tar.gz models/AbPlusEnT_backup.ckpt

### procs, gpus ###
n_proc=6
num_gpus=1

echo "Running inference"
date

MODEL=models/AbPlusEnT_backup.ckpt
OUTDIR=sampled_sequences
PDB_DIR=data/antibodies
SAMPLER_SCRIPT=ProteinSequenceSampler.py
# SAMPLER_SCRIPT=AntibodySequenceSampler.py
python3 $SAMPLER_SCRIPT \
	--output_dir ${OUTDIR} \
	--model $MODEL \
	--from_pdb $PDB_DIR \
	--sample_temperatures 0.2,0.5 \
	--num_samples 100 \
	--antibody \
	--mask_ab_region cdrs \
	--num_gpus $num_gpus \
	--num_procs $n_proc \
	# --mask_ab_indices 10,11,12 \ # 0-indexed
