#!/bin/bash -l

source <YOUR ENVIRONMENT> 

### CHECK/DOWNLOAD->DECOMPRESS PRETRAINED MODEL ###
mkdir -p models
if [ ! -f models/model.tar.gz ] && [ ! -f models/ProtEnT_backup.ckpt ]; then
  wget -P models https://zenodo.org/records/8313466/files/model.tar.gz
fi
tar --skip-old-files --strip-components=1 -C models -xvzf models/model.tar.gz models/ProtEnT_backup.ckpt

### procs, gpus ###
n_proc=6
num_gpus=1

echo "Running inference"
date

MODEL=models/ProtEnT_backup.ckpt
OUTDIR=sampled_sequences
PDB_DIR=data/proteins
python3 ProteinSequenceSampler.py  \
	--output_dir ${OUTDIR} \
	--model $MODEL \
	--from_pdb $PDB_DIR \
	--sample_temperatures 0.2,0.5 \
	--num_samples 100 \
	--num_gpus $num_gpus \
	--num_procs $n_proc \
