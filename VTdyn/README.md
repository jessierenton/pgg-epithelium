# Voronoi tessellation model

Built using Python 2.7.15

This folder contains all code for running public goods games simulations on the Voronoi tessellation (VT) model. 
Structure contains basic code for implementing VT model, libs/public_goods_lib.py has code for running simulations.

See run_pgg.py for example of running a single simulation.

## Parallel simulations to generate fixation probability data
If running on slurm can use bash scripts in slurm_scripts folder to run batch jobs. Reads parameters from files in parameter files.
Otherwise run with command line arguments (described below). For cooperator fixation set MUTANT_NUM = 1; for defector fixation set MUTANT_NUM=99.

### N-player prisoner's dilemma
run_NPD_parallel.py takes command line arguments: b1, b2... (can give as many b values as needed)

### Volunteer's dilemma
run_VD_parallel.py takes command line args: volunteer threshold, b1, b2...

### Sigmoid public good
run_sigmoid_pgg_parallel.py takes command line args: s, h, b1, b2...

Can find data from these simulations in ../Data/fixprobs

## Neutral simulation data
neighbour_data_neutral.py runs simulations for neutral mutants and generates data on cell neighbourhoods (i.e. how many neighbours for each cell, how many mutant neighbours etc.) Data can be found in ../Data/batch_* .
