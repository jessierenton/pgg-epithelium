import numpy as np
import mesh, cell

def hex_centres(N_across,N_up,noise,rand,multiplier=1):
    """generate an NxN hexagonal lattice of points (with noise is not zero)
    return (NxN,2) array of centre positions and the width and height of domain  """
    assert(N_up % 2 == 0)  # expect even number of rows    
    width, height = float(N_across), float(N_up)*np.sqrt(3)/2
    x = np.arange(-width/2.,width/2.,width/N_across)
    y = np.arange(-height/2.,height/2.,height/(N_up/2))
    centres = np.zeros((N_across, N_up/2, 2, 2))
    centres[:, :, 0, 0] += x[:, np.newaxis]
    centres[:, :, 1, 0] += (x+1./2)[:, np.newaxis]
    centres[:, :, 0, 1] += y[np.newaxis, :]
    centres[:, :, 1, 1] += (y+np.sqrt(3)/2)[np.newaxis,:]
    centres = centres.reshape(-1, 2) + np.array([0.25,3**0.5/4])
    centres += (rand.rand(N_up*N_across, 2)-0.5)*noise 
    
    return centres*multiplier, width*multiplier, height*multiplier

    
def init_mesh_torus(N_cell_across,N_cell_up,noise,rand,multiplier=1,save_areas=False):
    """generate a mesh object with NxN cells and periodic bcs"""
    centres,width,height = hex_centres(N_cell_across,N_cell_up,noise,rand,multiplier)
    if save_areas: return mesh.Mesh(centres,mesh.Torus(width,height))
    else: return mesh.MeshNoArea(centres,mesh.TorusNoArea(width,height))
    return mesh.Mesh(centres,geometry)
    
def init_tissue_torus(N_cell_across,N_cell_up,noise,force,rand,save_areas=False,save_cell_histories=False):
    """generate a tissue object with NxN cells and given force object and periodic bcs"""
    N = N_cell_across*N_cell_up
    return cell.Tissue(init_mesh_torus(N_cell_across,N_cell_up,noise,rand,save_areas=save_areas),force,np.arange(N),
                N,np.zeros(N,dtype=float),np.full(N,-1,dtype=int),save_cell_histories=save_cell_histories)
    
def init_tissue_torus_with_multiplier(N_cell_across,N_cell_up,noise,force,rand,multiplier,ages=None,save_areas=False,save_cell_histories=False):
    """generate a tissue object with NxN cells and given force object and periodic bcs for density dep. sims"""
    N = N_cell_across*N_cell_up
    if ages is None: ages = np.zeros(N,dtype=float)
    return cell.Tissue(init_mesh_torus(N_cell_across,N_cell_up,noise,rand,multiplier,save_areas=save_areas),force,np.arange(N),
                N,ages,np.full(N,-1,dtype=int),save_cell_histories=save_cell_histories)
    
    
    
    