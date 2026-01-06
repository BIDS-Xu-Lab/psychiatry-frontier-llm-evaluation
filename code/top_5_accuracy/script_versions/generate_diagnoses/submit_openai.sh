#!/bin/bash

#SBATCH --job-name=openai
#SBATCH --time=2-
#SBATCH --mail-type=ALL
#SBATCH --mem=32G
#SBATCH --partition=week
#SBATCH --output=/gpfs/radev/project/xu_hua/kwj9/psychiatry_llm/evaluation_of_sota/jama_submission_12152025/psychiatry-frontier-llm-evaluation/code/top_5_accuracy/script_versions/generate_diagnoses/slurm_logs/%x-%j.out

module reset
module load miniconda
conda activate mh-eval

python 2_generate_diagnoses_openai.py
