# author Sergey Lyskov
# demo/reference Dockerfile for https://github.com/Graylab/MaskedProteinEnT repository

FROM python:3.10

RUN apt-get -y update
RUN apt-get -y install --no-install-recommends git curl

RUN git clone https://github.com/Graylab/MaskedProteinEnT.git /code
WORKDIR /code

RUN pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

RUN curl -O https://zenodo.org/records/8313466/files/model.tar.gz && tar -xvzf model.tar.gz && rm model.tar.gz

# testing if code working as expected
RUN python3 ProteinSequenceSampler.py  \
    --output_dir ./sampled_sequences \
    --model models/ProtEnT_backup.ckpt \
    --from_pdb data/proteins \
    --sample_temperatures 0.2,0.5 \
    --num_samples 100

RUN test -d ./sampled_sequences
RUN ls ./sampled_sequences

ENTRYPOINT /bin/bash
