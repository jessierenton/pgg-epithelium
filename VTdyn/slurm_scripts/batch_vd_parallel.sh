#!/bin/bash
#SBATCH --cpus-per-task=88
#SBATCH --array=0-5
#SBATCH --nodes=1
PARAMFILE=vd_params
number=$(($SLURM_ARRAY_TASK_ID+1))
p1=`(sed -n "$number"p $PARAMFILE) | awk '{print $1}'`
p2=`(sed -n "$number"p $PARAMFILE) | awk '{print $2}'`

source /clusternfs/jrenton/anaconda2/my_anaconda.sh

python run_vd_parallel.py $p1 $p2