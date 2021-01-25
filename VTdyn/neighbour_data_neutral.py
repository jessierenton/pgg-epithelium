from multiprocessing import Pool  #parallel processing
import multiprocessing as mp
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth
import structure.initialisation as init
import sys
import os
import numpy as np
import libs.pd_lib_neutral as lib
import libs.data as data
from functools import partial
import pandas as pd

def distribution_data(history,mutant_id,i,all_types=False):
    """
    generates neighbour data for mutants (or all cells if all_types is True)
    cells are labelled by their ancestor. all cells with ancestor=mutant_id are type 1, all other cells type 0.
        returns list of dicts with keys: tissueid, time, n, k, j [, type] 
    n = # type 1 cells
    k = # neighbours
    j = # type 1 neighbours
    """
    if all_types:
        return [{'tissueid':i,'time':int(tissue.time),'n':sum(tissue.properties['ancestor']==mutant_id),'k':len(cell_neighbours),
                'j':sum((tissue.properties['ancestor']==mutant_id)[cell_neighbours]),'type': 1 if tissue.properties['ancestor'][idx]==mutant_id else 0} 
                for tissue in history if 1<=sum(tissue.properties['ancestor']==mutant_id)<100
                    for idx,cell_neighbours in enumerate(tissue.mesh.neighbours)]    
    else:
        return [{'tissueid':i,'time':int(tissue.time),'n':sum(tissue.properties['ancestor']==mutant_id),'k':len(cell_neighbours),
                'j':sum((tissue.properties['ancestor']==mutant_id)[cell_neighbours])} 
                for tissue in history if 1<=sum(tissue.properties['ancestor']==mutant_id)<100
                    for idx,cell_neighbours in enumerate(tissue.mesh.neighbours) if tissue.properties['ancestor'][idx]==mutant_id]     



def run_sim(all_types,i):
    """run a single simulation and save neighbour data for mutants (or all cells if all_types is True)"""
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=False,
                                    init_time=INIT_TIME,til_fix='exclude_final',save_areas=False)  
    mutant_id = np.argmax(np.bincount(history[-1].properties['ancestor']))
    return distribution_data(history,mutant_id,i,all_types)

L = 10 # population size N = l*l
INIT_TIME = 96. # initial simulation time to equilibrate 
TIMEND = 80000. # length of simulation (hours)
TIMESTEP = 12. # time intervals to save simulation history
SIM_RUNS = int(sys.argv[1]) # number of sims to run taken as command line arg

save_all_types = False # set to True to save data for cooperators AND defectors (otherwise just cooperator data saved)
simulation = lib.simulation_ancestor_tracking # tracks clones with common ancestor

outdir = 'coop_neighbour_distribution/'
if not os.path.exists(outdir): 
     os.makedirs(outdir)
savename = 'batch_tm'




# run simulations in parallel 
cpunum=mp.cpu_count()
pool = Pool(processes=cpunum-1,maxtasksperchild=1000)
df = pd.DataFrame(sum(pool.map(partial(run_sim,save_all_types),range(SIM_RUNS)),[]))
pool.close()
pool.join()
df.to_csv(outdir+savename,index=False)