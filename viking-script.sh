#!/bin/bash
# Script to be run on compute clusters using Slurm.
# The Slurm parameters are set by the CLI. See `./cli.py slurm --help`.

echo "Loading CUDA 11.0"
module load system/CUDA/11.0.2-GCC-9.3.0

pushd $NOISY_DIR
[ -d .venv/ ] && . .venv/bin/activate
pip install -r requirements.txt
./cli.py train --checkpoint $NOISY_CHECKPOINT
popd
