import os
import sys
import numpy as np
import itertools
import structure
from structure.global_constants import T_D,dt,ETA,MU
from structure.cell import Tissue, BasicSpringForceNoGrowth
import structure.initialisation as init

def print_progress(step,N_steps):
    sys.stdout.write("\r %.2f %%"%(step*100./N_steps))
    sys.stdout.flush() 

def run(simulation,N_step,skip):
    """run a given simulation for N_step iterations
    returns list of tissue objects at intervals given by skip"""
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]

def run_generator(simulation,N_step,skip):
    """generator for running a given simulation for N_step iterations
    returns generator for of tissue objects at intervals given by skip"""
    return itertools.islice(simulation,0,N_step,skip)

def run_return_events(simulation,N_step):
    return [tissue.copy() for tissue in itertools.islice(simulation,N_step) if tissue is not None]

def run_return_final_tissue(simulation,N_step):
    return next(itertools.islice(simulation,N_step,None))

def run_til_fix(simulation,N_step,skip,include_fixed=True):
    return [tissue.copy() for tissue in generate_til_fix(simulation,N_step,skip,include_fixed=include_fixed)]
        
def fixed(tissue):
    try:
        return (1 not in tissue.properties['type'] or 0 not in tissue.properties['type'])
    except KeyError:
        return np.all(tissue.properties['ancestor']==tissue.properties['ancestor'][0])
    

def generate_til_fix(simulation,N_step,skip,include_fixed=True):
    for tissue in itertools.islice(simulation,0,N_step,skip):
        if not fixed(tissue):
            yield tissue
        else:
            if include_fixed:
                yield tissue
            break

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------ SIMULATION ROUTINES ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def simulation_no_division(tissue,dt,N_steps,rand):
    """run tissue simulation with no death or division"""
    yield tissue
    step = 1.
    while True:
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        tissue.update(dt)
        yield tissue

def simulation(tissue,dt,N_steps,stepsize,rand,eta=ETA,progress_on=False):
    yield tissue
    step = 1.
    while True:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        mesh.move_all(tissue.dr(dt,eta))
        if rand.rand() < (1./T_D)*N*dt:
            mother = rand.randint(N)
            tissue.add_daughter_cells(mother,rand)
            tissue.remove(mother,True)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        if progress_on: print_progress(step,N_steps)
        step += 1 
        yield tissue
        
def simulation_ancestor_tracking(tissue,dt,N_steps,stepsize,rand,eta=ETA,progress_on=False):
    """simulation loop for neutral process tracking ancestor ids"""
    tissue.properties['ancestor']=np.arange(len(tissue))
    return simulation(tissue,dt,N_steps,stepsize,rand,eta=eta,progress_on=progress_on)
    

def simulation_mutant_tracking(tissue,dt,N_steps,stepsize,rand,eta=ETA,progress_on=False,mutant_number=1,mutant_type=1):
    """simulation loop for neutral process tracking mutant ids"""
    tissue.properties['type'] = np.full(len(tissue),1-mutant_type,dtype=int)
    tissue.properties['type'][rand.choice(len(tissue),size=mutant_number,replace=False)]=mutant_type
    return simulation(tissue,dt,N_steps,stepsize,rand,eta=eta,progress_on=progress_on)

def initialise_tissue(N,dt,timend,timestep,rand,mu=MU,save_areas=False,save_cell_histories=False):  
    """initialise tissue and run simulation until timend returning final state"""              
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(mu),rand,save_areas=save_areas,save_cell_histories=save_cell_histories)
    if timend !=0: 
        tissue = run_return_final_tissue(simulation(tissue,dt,timend/dt,timestep/dt,rand),timend/dt)
        tissue.reset(reset_age=True)
    return tissue

def run_simulation(simulation,N,timestep,timend,rand,init_time=None,mu=MU,eta=ETA,dt=dt,til_fix=True,generator=False,save_areas=False,
                tissue=None,save_cell_histories=False,progress_on=False,**kwargs):
    """initialise tissue with NxN cells and run given simulation with given game and constants.
            starts with single cooperator
            ends at time=timend OR if til_fix=True when population all cooperators (type=1) or defectors (2)
        returns history: list of tissue objects at time intervals given by timestep
            """
    if tissue is None:
        tissue = initialise_tissue(N,dt,init_time,timestep,rand,mu=mu,save_areas=save_areas,save_cell_histories=save_cell_histories)
    if til_fix:
        include_fix = not (til_fix=='exclude_final')
        if generator:
            history = generate_til_fix(simulation(tissue,dt,timend/dt,timestep/dt,rand,eta=eta,progress_on=progress_on,**kwargs),timend/dt,timestep/dt,include_fix)
        else:
            history = run_til_fix(simulation(tissue,dt,timend/dt,timestep/dt,rand,eta=eta,progress_on=progress_on,**kwargs),timend/dt,timestep/dt)
    else:
        history = run(simulation(tissue,dt,timend/dt,timestep/dt,rand,eta=eta,progress_on=progress_on,**kwargs),timend/dt,timestep/dt)
    return history
