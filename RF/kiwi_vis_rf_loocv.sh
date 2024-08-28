#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/asa30/4thYearImplementation/RF
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J kiwi-vis-rf-loocv
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=99:00:00
## Memory limit (in megabytes)
#SBATCH --mem=102400    # 100GB
## Specify partition
#SBATCH -p nodes

################# Part-2 Shell script ####################
#===============================
#  Activate Flight Environment
#-------------------------------
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

#===========================
#  Create results directory
#---------------------------
RESULTS_DIR="$(pwd)/${SLURM_JOB_NAME}-outputs/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

#===============================
#  Application launch commands
#-------------------------------
# Customize this section to suit your needs.

echo "Executing job commands, current working directory is $(pwd)"

# REPLACE THE FOLLOWING WITH YOUR APPLICATION COMMANDS
conda init 
conda activate tf-dmog 
python kiwi_vis_rf_loocv.py