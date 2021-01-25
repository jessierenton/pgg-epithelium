import numpy as np
# from pathos.multiprocessing import cpu_count
# from pathos.pools import ParallelPool as Pool
from multiprocessing import Pool,cpu_count
import libs.public_goods_lib as lib #library for simulation routines
import libs.data as data
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import sys,os
from itertools import product

NUMBER_SIMS = 10000
BATCH_SIZE = 1000
DELTA = 0.025
L = 10 # population size N=l*l
MUTANT_NUM = 1 # set to 1 for cooperator invaasion, N-1 for defector invasion
TIMEND = 10000. # simulation time (hours)
TIMESTEP = 12. # time intervals to save simulation history
INIT_TIME = 12.

threshold = int(sys.argv[1]) # first command line arg is threshold number volunteers
b_vals = np.array(sys.argv[2:],dtype=float) #succeeding CLA are b values

PARENTDIR = 'VD_fixprobs/'
PARENTDIR += '/threshold_%d/'%threshold
if not os.path.exists(PARENTDIR): # if the outdir doesn't exist create it
     os.makedirs(PARENTDIR)

game = lib.volunteers_dilemma
simulation = lib.simulation_decoupled_update

with open(PARENTDIR+'info',"w") as f:
    f.write('pop size = %3d\n'%(L*L))
    f.write('timestep = %.1f'%TIMESTEP)

def fixed(history,i,b):
    """returns 1/0 if cooperation/defection fixates, otherwise returns -1 (and saves mutant history)"""
    if 0 not in history[-1].properties['type']:
        fix = 1  
    elif 1 not in history[-1].properties['type']:
        fix = 0
    else: 
        fix = -1
        data.save_N_mutant(history,PARENTDIR+'/incomplete_b%.1f'%b,i)
    return fix

def run_single_unpack(args):
    return run_single(*args)

def run_single(i,b):
    """run a single simulation to fixation"""
    game_constants = (b,1.,threshold)
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,DELTA,game,game_constants,mutant_num=MUTANT_NUM,
                init_time=INIT_TIME,til_fix=True,save_areas=False,progress_on=False)
    fixation = fixed(history,i,b)
    with open(PARENTDIR+'b%.2ftime'%b,'a') as wfile:
        wfile.write('%5d    %5d    %d\n'%(i,history[-1].time,fixation))
    return fixation
    
def run_parallel(b_vals,number_sims,batch_size):
    """run simulations in parallel and save fixation data"""
    for b in b_vals:
        pool = Pool(cpu_count()-1,maxtasksperchild=1000)
        fixation = np.array([f for f in pool.imap(run_single_unpack,product(range(number_sims),[b]))]) 
        with open(PARENTDIR+'b%.2f'%b,'w') as wfile:    
            if number_sims%batch_size != 0: 
                batch_size=1
            fixation = fixation.reshape((number_sims/batch_size,batch_size))
            for fixation_batch in fixation:
                fixed = len(np.where(fixation_batch==1)[0])
                lost = len(np.where(fixation_batch==0)[0])
                incomplete = len(np.where(fixation_batch==-1)[0])
                wfile.write('%d    %d    %d\n'%(fixed,lost,incomplete))


run_parallel(b_vals,NUMBER_SIMS,BATCH_SIZE)  
