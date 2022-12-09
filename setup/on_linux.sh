#! /bin/bash
source "/home/$USER/miniconda3/etc/profile.d/conda.sh"

conda deactivate

conda activate base
mamba env remove --name thesis-core
mamba env create --file core_environment.yml --name thesis-core
conda activate thesis-core
python -c "import torch; print(torch.cuda.is_available())"
conda deactivate

mamba env remove --name thesis-embed
mamba env create --file embed_environment.yml --name thesis-embed
conda activate thesis-embed
pip install bio-embeddings[all] align
conda deactivate
