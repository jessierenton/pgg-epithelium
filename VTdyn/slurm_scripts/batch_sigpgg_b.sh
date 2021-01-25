#!/bin/bash
#SBATCH --cpus-per-task=42
#SBATCH --array=0-4
#SBATCH --nodes=1
PARAMFILE=sigpgg_params_b
number=$(($SLURM_ARRAY_TASK_ID+1))
p1=`(sed -n "$number"p $PARAMFILE) | awk '{print $1}'`
p2=`(sed -n "$number"p $PARAMFILE) | awk '{print $2}'`
p3=`(sed -n "$number"p $PARAMFILE) | awk '{print $3}'`
p4=`(sed -n "$number"p $PARAMFILE) | awk '{print $4}'`
p5=`(sed -n "$number"p $PARAMFILE) | awk '{print $5}'`
p6=`(sed -n "$number"p $PARAMFILE) | awk '{print $6}'`
p7=`(sed -n "$number"p $PARAMFILE) | awk '{print $7}'`
source /clusternfs/jrenton/anaconda2/my_anaconda.sh

python run_sigmoid_pgg_parallel.py $p1 $p2 $p3 $p4 $p5 $p6 $p7