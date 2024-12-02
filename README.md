# MaskedProteinEnT
Code to sample sequences with a contextual Masked EnTransformer as described in  ["Contextual protein and antibody encodings from equivariant graph transformers"](https://pubmed.ncbi.nlm.nih.gov/37503113/).

![Self-supervised learning to transduce sequence labels for masked residues from those for unmasked residues by context matching on proteins.![image](https://github.com/Graylab/MaskedProteinEnT/assets/14285703/383bb634-1870-42ac-a82a-7563a3b90c82)
](./MainFigure_Model_downsized.png)

## Installation

For sampling, in your virtual environment, pip install as follows:
```sh
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Installation with Docker
`Dockerfile` is provided as example/demo of package use. Please see example command lines to use below. For production use you might need to mount host data dir as a subdir to `/code` dir where package code is located.
```
docker build -t masked-protein-ent .
docker run -it masked-protein-ent
```

## Test with Colab
Example Jupyter notebook for Colab is provided in MaskedProteinEnT-colab-example.ipynb. Please note that due to volatile nature of Colab platform it is difficult to ensure that in long term such notebook will be functionining so some edits might be required.
Alternatively, we provide a dockerfile for easy installation.

Sampling works well on CPUs and GPUs.
**Sampling is just as fast on cpus: <2min for 10000 sequences**


## Trained models
Download and extract trained models from [Zenodo](https://zenodo.org/records/8313466).
```
tar -xvzf model.tar.gz
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

## Sampling antibody sequences without partner context
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
	--num_samples 100 \
	--antibody \
	--mask_ab_indices 10,11,12
# To sample for a specific region
#	--mask_ab_region h3

```
The above command samples all sequences at 100% masking (i.e. only coord information is used by the model). You may sample at any other masking rate between 0-100% and the model will randomly select the positions to mask. For more options, run:

```bash
python3 ProteinSequenceSampler.py --help
```

## Sampling interface residues with partner context
To generate/design the interface residues for the first partner (order determined by partners.json), run:

```bash
MODEL=trained_models/ProtPPIEnT_backup.ckpt
OUTDIR=./sampled_ppi_sequences
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

## Sampling antibody interface residues with antigen context
```
MODEL=trained_models/ProtAbAgEnT_backup.ckpt
OUTDIR=./sampled_abag_sequences
PDB_DIR=data/abag/
PPI_PARTNERS_DICT=data/abag/1n8z_partners.json
python3 PPIAbAgSequenceSampler.py  \
        --output_dir ${OUTDIR} \
        --model $MODEL \
        --from_pdb $PDB_DIR \
	--sample_temperatures 0.2,0.5 \
       	--num_samples 100 \
	--partners_json ${PPI_PARTNERS_DICT} \
	--partner_name Ab \
        --antibody
# To specify sampling at a specific CDR loop:
# --mask_ab_region h3
# To specify sampling at a specific indices:
# --mask_ab_indices 10,11,12
```

## Training
### Installation
Model was trained with older versions of torch and pytorch_lightning. Newer versions are not backward compatible. The following instructions work for python 3.9 and cuda 11.1.
To train the model, you need to install torch and other dependencies as follows:
In your virtual env, run the following commands:
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements_torch191.txt
```
### Training
- The general training code is provided in `training_masked_model.py`. 
- For training the model under different settings, see `scripts/train_model_protein.sh` for training on the general protein dataset, `scripts/fine_tune_ppi-abag.sh` for fine-tuning on ppi-abag datasets.
- Training datasets are available under this [Zenodo link](https://zenodo.org/doi/10.5281/zenodo.13831402). See the table below for an overview and the methods section for detailed information of each dataset.
- The training script requires a [wandb entity](https://wandb.ai/site) for training logging. See `--wandb_entity` argument in `train_masked_model.py`.

| Description | File name | Download link | References |
|---|---|---|---|
| Training dataset identifiers | ids_train_casp12nr50_nr70Ig_nr40Others.fasta | [:arrow_down:](https://zenodo.org/records/13831403/files/ids_train_casp12nr50_nr70Ig_nr40Others.fasta) | n.a. |
| Training and validation datasets curated from the CASP12 version of Sidechainnet | sidechainnet_casp12_50.pkl | [:arrow_down:](https://zenodo.org/records/13831403/files/sidechainnet_casp12_50.pkl) | - [AlQuraishi, 2019](https://doi.org/10.1186/s12859-019-2932-0)<br> - [King & Koes, 2020](https://doi.org/10.48550/arxiv.2010.08162) |
| Training dataset on non-redundant heterodimer protein-protein interfaces curated from referenced work | ppi_trainset_5032_noabag_aug2022.h5 | [:arrow_down:](https://zenodo.org/records/13831403/files/ppi_trainset_5032_noabag_aug2022.h5) | [Gainza et al, 2020](https://doi.org/10.1038/s41592-019-0666-6) |
| Training dataset for antibody-antigen models curated from SAbDAb | __ | [:arrow_down:]() | [Dunbar et al, 2014](https://doi.org/10.1093/nar/gkt1043) |
| Training dataset for antibody-only models curated from SAbDAb | AbSCSAbDAb_trainnr90_bkandcbcoords_aug2022.h5 | [:arrow_down:](https://zenodo.org/records/13831403/files/AbSCSAbDAb_trainnr90_bkandcbcoords_aug2022.h5) | [Dunbar et al, 2014](https://doi.org/10.1093/nar/gkt1043) |
| Training dataset for antibodies with paired antibody structures generated with AlphaFold2 from a previous study | train_af_paired_nr70.h5 | [:arrow_down:](https://zenodo.org/records/13831403/files/train_af_paired_nr70.h5) | - [Dunbar et al, 2014](https://doi.org/10.1093/nar/gkt1043) <br> - [Ruffolo et al, 2023](https://doi.org/10.1038/s41467-023-38063-x) |
| Training dataset for antibodies with unpaired or single-chain antibody structures generated with AlphaFold2 from a previous study | train_af_unpaired_nr70.h5 | [:arrow_down:](https://zenodo.org/records/13831403/files/train_af_unpaired_nr70.h5) | - [Dunbar et al, 2014](https://doi.org/10.1093/nar/gkt1043) <br> - [Ruffolo et al, 2023](https://doi.org/10.1038/s41467-023-38063-x) |
| Test dataset curated from multiple sources |testset_rabd-dms-vhh_backboneandcb_oct2022.h5 | [:arrow_down:](https://zenodo.org/records/13831403/files/testset_rabd-dms-vhh_backboneandcb_oct2022.h5) | - [Li et al, 2014](https://doi.org/10.1002/prot.24620) <br> - [Gainza et al, 2020](https://doi.org/10.1038/s41592-019-0666-6) <br> - [Cho et al, 2003](https://doi.org/10.1038/nature01392) <br> - [Mason et al, 2021](https://doi.org/10.1038/s41551-021-00699-9) <br> - [Ruffolo, Gray & Sulam, 2021](https://doi.org/10.48550/arxiv.2112.07782.) |

## References
- EnTransformer code is based on [Phil Wang's implementation](https://github.com/lucidrains/En-transformer/tree/373efe752d0a9959fc0a61e2c6d5ca423c491682) of EGNN (Satorras et al. 2021) with equivariant transformer layers.
- Models and sequence recovery reported for Antibody CDRs with different models reported in Figure 2 available at https://zenodo.org/record/8313466.
- Please note that our protein training dataset is sourced from [SidechainNet](https://github.com/jonathanking/sidechainnet). You can download the dataset directly from their repository. We are providing it here solely for ease of access. We highly recommend visiting the linked repository above and referring to their publications for more detailed information. Please remember to cite SidechainNet in your work if you utilize their dataset.

If you use this repository to generate or score sequences, please cite:
```
Mahajan, S. P., Ruffolo, J. A., Gray, J. J., "Contextual protein and antibody encodings from equivariant graph transformers", biorxiv, 2023.
```
```
Mahajan, S. P., Davila-Hernandez, F.A., Ruffolo, J. A., Gray, J. J., "How well do contextual protein encodings learn structure, function, and evolutionary context?", 2023. Under Review.
```
