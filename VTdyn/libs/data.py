import os
import numpy as np
import functools
from itertools import product
import json

#library of functions for saving data from a history object (list of tissues)

def memoize(func):
    cache =  dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func

def counter(func):
    def wrapper(*args,**kwargs):
        wrapper.count += 1
        result = func(*args,**kwargs)
        print wrapper.count
        return result
    wrapper.count = 0
    return wrapper

def save_mean_area(history,outdir,index=0):
    """saves mean area of cells in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = "%s/area_mean_%03d"%(outdir,index)
    np.savetxt(filename,[np.mean(tissue.mesh.areas) for tissue in history])

def save_areas(history,outdir,index=0):
    """saves all areas of cells in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn"t exist create it
         os.makedirs(outdir)
    filename = "%s/areas_%03d"%(outdir,index)
    wfile = open(filename,"w")
    for tissue in history:
        for area in tissue.mesh.areas:        
            wfile.write("%.3e    "%area)
        wfile.write("\n")
    
def mean_force(history,outdir,index=0):
    """saves mean magnitude of force on cells in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn"t exist create it
         os.makedirs(outdir)
    wfile = open("%s/%s_%03d"%(outdir,"force",index),"w")
    for tissue in history:        
        wfile.write("%.3e \n"%np.mean(np.sqrt((tissue.Force(tissue)**2).sum(axis=1))))
    wfile.close() 

def save_neighbour_distr(history,outdir,index=0):
    """save neighbour distributions in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = "%s/%s_%03d"%(outdir,"neigh_distr",index) 
    np.savetxt(wfilename,neighbour_distribution(history),fmt=(["%d"]*18))


def save_N_cell(history,outdir,index=0):
    """save number of cells in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn"t exist create it
         os.makedirs(outdir)
    wfilename = "%s/%s_%03d"%(outdir,"N_cell",index)  
    np.savetxt(wfilename,population_size(history),fmt=("%d"))

def save_N_mutant(history,outdir,index=0):
    """saves number of mutants in each tissue given by "type" property"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = "%s/%s_%03d"%(outdir,"N_mutant",index)  
    np.savetxt(wfilename,number_mutants(history),fmt=("%d"))

def save_cell_histories(history,outdir,index=0):
    """saves ids, ages and time of extruded and divided cells"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    fdivided = "%s/%s_%03d"%(outdir,"divided_cells",index)  
    fextruded = "%s/%s_%03d"%(outdir,"extruded_cells",index)  
    try: np.savetxt(fdivided,history[-1].divided_cells,fmt=["%5d","%.3f","%.3f"])
    except AttributeError: pass
    try: np.savetxt(fextruded,history[-1].extruded_cells,fmt=["%5d","%.3f","%.3f"])
    except AttributeError: pass

def save_cycle_phases(history,outdir,index=0):
    "saves number of cells in each cycle phase"
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = "%s/%s_%03d"%(outdir,"cycle_phase",index)
    np.savetxt(filename,cycle_phases(history),fmt=("%3d")) 
    
def save_cell_density(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = "%s/%s_%03d"%(outdir,"cell_density",index)
    np.savetxt(filename,cell_density(history),fmt="%.6f")

def save_tension_area_product(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = "%s/%s_%03d"%(outdir,"tension_area_product",index)
    np.savetxt(filename,mean_tension_area_product(history,std=True),fmt="%.6f")

def division_history(history):
    return history[-1].divided_cells

def extrusion_history(history):
    return history[-1].extruded_cells
    
def number_mutants(history):
    return [sum(tissue.properties["type"]) for tissue in history]

def population_size(history):
    return [len(tissue) for tissue in history]

def neighbour_distribution(history):
    return [np.bincount([len(tissue.mesh.neighbours[i]) for i in range(len(tissue))],minlength=18) for tissue in history]

def cell_cycle_lengths(history,start_time=0.0,ids=None):
    cell_histories_ = cell_histories(history,start_time)
    return [age for age,divided in zip(cell_histories_['age'],cell_histories_['divided']) if divided]

def extrusion_ages(history,start_time=0.0):
    cell_histories_ = cell_histories(history,start_time)
    return [age for age,divided in zip(cell_histories_['age'],cell_histories_['divided']) if not divided]

def transition_ages(history,start_time=0.0):
    cell_histories_ = cell_histories(history,start_time)
    return [trans_age for trans_age in cell_histories_['transition_age'] if trans_age>=0]
    
def cycle_phases(history):
    return [(sum(tissue.properties["cycle_phase"]),sum(1-tissue.properties["cycle_phase"])) for tissue in history]

def cycle_phase_proportion(history,phase):
    return [sum(tissue.properties["cycle_phase"]==phase)/float(len(tissue)) for tissue in history] 

def cell_density(history):
    return [len(tissue)/(tissue.mesh.geometry.width*tissue.mesh.geometry.height) for tissue in history]
    
def mean_force(history,std=True):
    if std:
        return [(np.mean(np.sqrt((tissue.Force(tissue)**2).sum(axis=1))),np.std(np.sqrt((tissue.Force(tissue)**2).sum(axis=1)))) for tissue in history]
    else:
        return [np.mean(np.sqrt((tissue.Force(tissue)**2).sum(axis=1))) for tissue in history]

def forces(history):
    return [np.sqrt((tissue.Force(tissue)**2).sum(axis=1)).tolist() for tissue in history]

def cell_histories(history,start_time=0.0):
    cell_histories_ = history[-1].cell_histories
    if start_time>0.:
        time = np.array(cell_histories_['time'])
        start = np.where(time>start_time)[0][0]
        cell_histories_ = {key:valist[start:] for key,valist in cell_histories.iteritems()}
    for key,value in cell_histories_.iteritems():
        if isinstance(value,np.ndarray):
            cell_histories_[key] = value.tolist()
    return cell_histories_

def mean_tension_area_product(history,std=True):
   t_a_p = [[tissue.tension_area_product(i) for i in range(len(tissue))] for tissue in history]
   if std: return [(np.mean(t),np.std(t)) for t in t_a_p]
   else: return [np.mean(t) for t in t_a_p]

def mean_cell_seperation(history,std=True):
    cs = [[d for neighbour_distances in tissue.mesh.distances for d in neighbour_distances] 
            for tissue in history]
    if std: return [(np.mean(d),np.std(d)) for d in cs]
    else: return [np.mean(d) for d in cs]

def mean_area(history,std=True):
    if std: 
        return [(np.mean(tissue.mesh.areas),np.std(tissue.mesh.areas)) for tissue in history]
    else: 
        return [np.mean(tissue.mesh.areas) for tissue in history]

def areas(history):
    return [tissue.mesh.areas.tolist() for tissue in history]

def ages(history):
    return [tissue.age.tolist() for tissue in history]

@memoize
def get_local_density(mesh):
    return mesh.local_density()

def save_info(history,outdir,index=0,**kwargs):
    """saves import info and parameters for a simulation"""
    timestep = history[1].time-history[0].time
    timend = history[-1].time
    with open(outdir+"/info_%03d"%index, "w") as f:
        f.write("timestep = %.2f, simulation length = %.1f (hours)\n"%(timestep,timend))
        f.write("initial population size = %d \n"%len(history[0]))
        for key,(val,fmt) in kwargs.iteritems():
            f.write(key+" = "+fmt%val+"\n")
    
def save_ages(history,outdir,index=0):
    """saves all cell ages for each tissue in history"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = "%s/ages_%03d"%(outdir,index)
    wfile = open(filename,"w")
    for tissue in history:
        for age in tissue.age:        
            wfile.write("%.3e    "%age)
        wfile.write("\n")    
    
def save_mean_age(history,outdir,index=0):
    """save mean age of cells for each tissue in history"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = "%s/age_mean_%03d"%(outdir,index)
    np.savetxt(filename,[np.mean(tissue.age) for tissue in history])

def save_mean_stress(history,outdir,index=0):
    """save mean stress on cells for each tissue in history"""
    if not os.path.exists(outdir): # if the folder doesn"t exist create it
         os.makedirs(outdir)
    filename = "%s/stress_mean_%03d"%(outdir,index)
    np.savetxt(filename,[np.mean([tissue.cell_stress(i) for i in range(len(tissue))]) for tissue in history])

def save_stress(history,outdir,index=0):
    """save stress on each cell for each tissue in history"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = "%s/stress_%03d"%(outdir,index)
    with open(filename,"w") as f:
        for tissue in history:
            for i in range(len(tissue)):
                f.write("%.5e    "%tissue.cell_stress(i))
            f.write("\n")

def save_var_to_mean_ratio_all(history,outdir,s,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    initial_types = np.unique()
    if type is None: filename = "%s/var_to_mean_ratio_%03d"%(outdir,index)
    else: filename = "%s/var_to_mean_ratio_type_%03d_%03d"%(outdir,type_,index) 
    np.savetxt(filename,[_calc_var_to_mean(tissue,s,type_) for tissue in history],fmt="%.4f")
    
def _calc_var_to_mean(width,height,centres,s):
    # width,height,centres = tissue.mesh.geometry.width,tissue.mesh.geometry.height,tissue.mesh.centres
    range_=((-width/2.,width/2),(-height/2,height/2))
    n_array=np.histogram2d(centres[:,0],centres[:,1],s,range_)[0]
    return np.var(n_array)/np.mean(n_array)

def calc_density(width,height,centres,s):
    range_=((-width/2.,width/2),(-height/2,height/2))
    return np.histogram2d(centres[:,0],centres[:,1],s,range_)[0]
  
def save_all(history,outdir,index=0):
    save_N_cell(history,outdir,index)
    save_N_mutant(history,outdir,index)
    save_ages(history,outdir,index)
    save_neighbour_distr(history,outdir,index)
    save_force(history,outdir,index)
    try: save_areas(history,outdir,index)
    except: pass
    save_age_of_death(history,outdir,index)
    
def save_as_json(history,outfile,fields,parameters=None,index=0):
    data = {f:FIELDS_DICT[f](history) for f in fields}
    for key,value in data.iteritems():
        if isinstance(value,np.ndarray):
            data[key] = value.tolist()
    if parameters is not None: data["parameters"]=parameters
    with open(outfile+'_%03d.json'%index,"w") as f:        
        json.dump(data,f,indent=4)
    
    
FIELDS_DICT = {"pop_size":population_size,"mutants":number_mutants,"neighbours":neighbour_distribution,
                "cycle_lengths":cell_cycle_lengths,"extrusion_ages":extrusion_ages,"cycle_phases":cycle_phases,
                "density":cell_density,"energy":mean_tension_area_product,"cell_seperation":mean_cell_seperation,
                "cell_histories":cell_histories,"transition_ages":transition_ages,"mean_area":mean_area,"areas":areas,
                "mean_force":mean_force,"forces":forces,"ages":ages} 
