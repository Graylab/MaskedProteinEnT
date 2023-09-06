#!/bin/bash -l

#ACTIVATE ENV

MODEL=models/ProtPPIEnT_backup.ckpt
OUTDIR=outdir
PDB_DIR=data/ppis
PPI_PARTNERS_DICT=data/ppis/heteromers_partners_example.json
python3 PPIAbAgSequenceSampler.py  \
        --output_dir ${OUTDIR} \
        --model $MODEL \
        --from_pdb $PDB_DIR \
	--sample_temperatures 0.2,0.5 \
       	--num_samples 100 \
	--partners_json ${PPI_PARTNERS_DICT} \
	--partner_name p1
 

