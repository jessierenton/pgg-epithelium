import numpy as np
import libs.public_goods_lib as lib #library for simulation routines
import libs.plot as vplt #plotting library
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import pandas as pd

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = 10 # simulation time (hours)
timestep = 1.0 # time intervals to save simulation history
init_time = 12.
rand = np.random.RandomState()

simulation = lib.simulation_decoupled_update  #simulation routine imported from lib
DELTA = 0.025
MUTANT_NUM = 1
b,c = 5.,1.
s,h = 10,0.5

# game = lib.volunteers_dilemma
# game_constants = (b,c,threshold)

game = lib.sigmoid_game
game_constants = (b,c,s,h)

history = lib.run_simulation(simulation,l,timestep,timend,rand,DELTA,game,game_constants,mutant_num=MUTANT_NUM,
                init_time=init_time,til_fix=True,save_areas=False,progress_on=True,return_events=True)
