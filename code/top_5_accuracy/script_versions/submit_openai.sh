#!/bin/bash

#SBATCH --job-name=openai
#SBATCH --time=1-
#SBATCH --mail-type=ALL
#SBATCH --mem=32G
#SBATCH --partition=day
#SBATCH --output=/gpfs/radev/project/xu_hua/kwj9/psychiatry_llm/evaluation_of_sota/top_5_accuracy/jama/slurm_logs/%x-%j.out

module reset
module load miniconda
conda activate mh-eval

python generate_diagnoses_openai.py
