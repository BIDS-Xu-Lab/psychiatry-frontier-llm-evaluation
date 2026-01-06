#!/bin/bash

#SBATCH --job-name=anthropic
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mem=32G
#SBATCH --partition=day
#SBATCH --output=/gpfs/radev/project/xu_hua/kwj9/psychiatry_llm/evaluation_of_sota/jama_submission_12152025/psychiatry-frontier-llm-evaluation/code/top_5_accuracy/script_versions/slurm_logs/%x-%j.out

module reset
module load miniconda
conda activate mh-eval

python 3_generate_diagnoses_anthropic.py