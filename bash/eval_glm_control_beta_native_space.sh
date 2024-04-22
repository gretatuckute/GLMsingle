#!/bin/bash
#SBATCH --job-name=eval_glm_control_beta_native_space-%j
#SBATCH --time=18:00:00
#SBACTH --ntasks=1
#SBATCH --output="out/eval_glm_control_beta_native_space-%j.out"
#SBATCH --error="err/eval_glm_control_beta_native_space-%j.err"
#SBATCH --mem=120G
#SBATCH -p evlab

source /om2/user/gretatu/anaconda/etc/profile.d/conda.sh
conda activate glmsingle

cd /om5/group/evlab/u/`whoami`/GLMsingle/evaluation_on_control_beta/
python evaluate_GLMsingle_on_control_beta_native_space.py --preproc "${1}" --pcstop "${2}" --fracs "${3}" --UID "${4}"
