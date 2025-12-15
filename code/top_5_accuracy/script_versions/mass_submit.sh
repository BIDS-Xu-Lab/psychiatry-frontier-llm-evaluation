#!/bin/bash

# Hard-coded prefix for your Slurm scripts
PREFIX="submit_"
COUNT=0

# Find all .sh files with the given prefix and submit them
for script in ${PREFIX}*.sh; do
    # Check if the file exists and is not this script itself
    if [ -f "$script" ] && [ "$script" != "$0" ]; then
        echo "Submitting job: $script"
        sbatch "$script"
        ((COUNT++))
    fi
done

echo "Submitted $COUNT jobs with prefix '$PREFIX'."