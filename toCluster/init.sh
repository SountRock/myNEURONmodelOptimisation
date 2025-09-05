#!/bin/bash
#SBATCH --job-name=testHuber
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/sagalajev_lab/mathematical_models/SCS_mods/testHuber/output.txt
#SBATCH --error=/home/sagalajev_lab/mathematical_models/SCS_mods/testHuber/error.txt

echo "=== SLURM JOB STARTED ==="
echo "Date      : $(date '+%d-%m-%Y %H:%M:%S')"
echo "User      : $(whoami)"
echo "Host      : $(hostname)"
echo "Workdir   : $(pwd)"
echo "CPUs/task : ${SLURM_CPUS_PER_TASK}"
echo

module load anaconda/anaconda3/4.7.5
source activate neuron_env
nrnivmodl mod

echo "Python dir: $(which python)"
echo "Python ver: $(python --version)"
echo "NEURON    : $(python -c 'import neuron; print(neuron.__version__)')"
echo "NetPyNE   : $(python -c 'import netpyne; print(netpyne.__version__)')"
echo "DEAP      : $(python -c 'import deap; print(deap.__version__)')"
echo "Pymoo      : $(python -c 'import pymoo; print(pymoo.__version__)')"
echo

python /home/sagalajev_lab/mathematical_models/SCS_mods/testHuber/optimModel.py