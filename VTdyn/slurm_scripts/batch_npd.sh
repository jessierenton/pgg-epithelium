#!/bin/bash
#SBATCH --cpus-per-task=88
#SBATCH --array=0-6
#SBATCH --nodes=1
PARAMFILE=npd_params
number=$(($SLURM_ARRAY_TASK_ID+1))
p1=`(sed -n "$number"p $PARAMFILE) | awk '{print $1}'`

source /clusternfs/jrenton/anaconda2/my_anaconda.sh

python run_pgg_parallel.py $p1