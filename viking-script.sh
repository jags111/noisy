#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nls.forstner@gmail.com
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10gb
#SBATCH --output=./viking-logs/nora.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Checklist:
# - Create a config in ./configs/<name>-viking.yaml
# - Change the --output value above to the correct file
# - Change the NAME below to the correct value

NAME=nora
WD=./zoo/$NAME
CP=$WD/.latest

echo "Loading CUDA 11.0"
module load system/CUDA/11.0.2-GCC-9.3.0

pushd ~/scratch/noisy/
. .venv/bin/activate
pip install -r requirements.txt
./cli.py init \
    --workdir $WD \
    --config configs/$NAME-viking.yaml \
    --force true
./cli.py train \
    --checkpoint $CP
popd
