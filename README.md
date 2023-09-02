# MaskedProteinEnT
Code to sample sequences with a contextual Masked EnTransformer as described in  ["Contextual protein and antibody encodings from equivariant graph transformers"](https://pubmed.ncbi.nlm.nih.gov/37503113/).
<picture>
MainFigure_Model.tiff
</picture>

## Installation
In your virtual environment, pip install as follows:
```bash
# Install torch (for cuda11):
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Install everything else:
pip install -r requirements.txt
```

## Sampling protein sequences
To design/generate all positions on the protein, run:
```bash
MODEL=trained_models/ProtEnT_backup.ckpt
OUTDIR=./sampled_sequences
PDB_DIR=data/proteins
python3 ProteinSequenceSampler.py  \
	--output_dir ${OUTDIR} \
	--model $MODEL \
	--from_pdb $PDB_DIR \
	--sample_temperatures 0.2,0.5 \
	--num_samples 100
```
The above command samples all sequences at 100% masking (i.e. only coord information is used by the model). You may sample at any other masking rate between 0-100% and the model will randomly select the positions to mask. For more options, run:

```bash
python3 ProteinSequenceSampler.py --help
```

## Sampling interface residues with partner context
To generate/design the interface residues for the first partner (order determined by partners.json), run:

```bash
MODEL=../trained_models/ProtPPIEnT_backup.ckpt
TEST_RESULTS_BASE=./sampled_ppi_sequences
PDB_DIR=data/ppis
PPI_PARTNERS_DICT=data/ppis/heteromers_partners_example.json
python3 PPIAbAgSequenceSampler.py  \
        --output_dir ${OUTDIR} \
        --model $MODEL \
        --from_pdb $PDB_DIR \
	--sample_temperatures 0.2,0.5 \
       	--num_samples 100 \
	--partners_json ${PPI_PARTNERS_DICT} \
	--partner_name p0

# to design interface residues on second partner use
# --partner_name p0
# to design interface residues on both partners use
# --partner_name both
```
