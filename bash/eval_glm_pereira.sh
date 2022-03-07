#!/bin/bash
#SBATCH --job-name=run-%j
#SBATCH --time=06:30:00
#SBACTH --ntasks=1
#SBATCH --output="run-%j.out"
#SBATCH --error="run-%j.err"
#SBATCH --mem=80G

source /om2/user/gretatu/anaconda/etc/profile.d/conda.sh
conda activate glmsingle

cd /om5/group/evlab/u/`whoami`/GLMsingle/evaluation_on_pereira/
python evaluate_GLMsingle_on_pereira.py --preproc "${1}" --pcstop "${2}" --fracs "${3}" --UID "${4}"
