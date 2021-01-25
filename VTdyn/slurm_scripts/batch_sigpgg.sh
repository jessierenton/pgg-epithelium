#!/bin/bash
#SBATCH --cpus-per-task=62
#SBATCH --array=2
#SBATCH --nodes=1
PARAMFILE=sigpgg_params/p10
number=$(($SLURM_ARRAY_TASK_ID+1))
p1=`(sed -n "$number"p $PARAMFILE) | awk '{print $1}'`
p2=`(sed -n "$number"p $PARAMFILE) | awk '{print $2}'`
p3=`(sed -n "$number"p $PARAMFILE) | awk '{print $3}'`
source /clusternfs/jrenton/anaconda2/my_anaconda.sh

python run_sigmoid_pgg_parallel.py $p1 $p2 $p3