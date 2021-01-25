import numpy as np
import copy
from functools import partial
import global_constants as gc
from global_constants import EPS, L0, MU, ETA, T_M
              
class Tissue(object):    
    
    """Defines a tissue comprised of cells which can move, divide and be extruded"""
    
    def __init__(self,mesh,force,cell_ids,next_id,age,mother,properties=None,save_cell_histories=False,cell_histories=None,time=0.):
        """ Parameters:
        mesh: Mesh object
            defines cell locations and neighbour connections
        force: Force object
            defines force law between neighbouring cells
        cell_ids: (N,) array ints
             unique id for each cell (N is number of cells)
        next_id: int
            next available cell id
        age: (N,) array floats
            age of each cell
        mother: (N,) array ints
            id of mother for each cell (-1 for initial cells)
        properties: dict or None
            dictionary available for any other cell properties
            
        """
        self.mesh = mesh
        self.Force = force
        self.cell_ids = cell_ids
        self.next_id = next_id
        self.age = age
        self.mother = mother
        self.properties = properties or {}
        self.save_cell_histories = save_cell_histories
        if save_cell_histories:
            self.cell_histories = cell_histories or {}
        self.time=time
        
        
    def __len__(self):
        return len(self.mesh)
    
    def reset(self,reset_age=True):
        N = len(self)
        self.cell_ids = np.arange(N,dtype=int)
        if reset_age: self.age = np.zeros(N,dtype=float)
        self.next_id = N
        self.mother = -np.ones(N,dtype=int)
        self.time = 0.
        if self.save_cell_histories: self.cell_histories = {}
        
    
    def copy(self):
        """create a copy of Tissue"""
        if self.save_cell_histories:
            return Tissue(self.mesh.copy(),self.Force,self.cell_ids.copy(),self.next_id,self.age.copy(), self.mother.copy(),copy.deepcopy(self.properties),self.save_cell_histories, self.cell_histories,self.time)
        else: return Tissue(self.mesh.copy(),self.Force,self.cell_ids.copy(),self.next_id,self.age.copy(), self.mother.copy(),copy.deepcopy(self.properties),time=self.time)
    
    def mesh_id(self,cell_id):
        return np.where(self.mesh.ids==cell_id)[0]
             
    def update(self,dt):
        self.mesh.update()
        self.age += dt      
        self.time += dt
    
    def get_neighbour_cell_ids(self,idx_list,aslists=False):
        if aslists:
            try: 
                return self.cell_ids[self.mesh.neighbours[idx_list]].tolist()
            except TypeError:
                return [self.cell_ids[self.mesh.neighbours[i]].tolist() for i in idx_list]
        else:
            try: 
                return self.cell_ids[self.mesh.neighbours[idx_list]]
            except TypeError:
                return [self.cell_ids[self.mesh.neighbours[i]] for i in idx_list]
    
    def get_next_nearest_neighbour_cell_ids(self,idx_list,aslists=False):
        if aslists: 
            try:
               return self.cell_ids[self.mesh.next_nearest_neighbours(idx_list)].tolist()
            except TypeError:
               return [self.cell_ids[self.mesh.next_nearest_neighbours(i)].tolist() for i in idx_list]
        else:
            try:
                return self.cell_ids[self.mesh.next_nearest_neighbours(idx_list)]
            except TypeError:
                return [self.cell_ids[self.mesh.next_nearest_neighbours(i)] for i in idx_list]
    
    def update_cell_histories(self,idx_list,divided,position=True,neighbour_data=True,distance_data=True):
        if self.cell_histories == {}:
            self.cell_histories.update({'time':[],'cell_ids':[],'age':[],'divided':[]})
            try: self.cell_histories.update({'area':[]}) 
            except KeyError: pass 
            if position:
                self.cell_histories.update({'position':None})
            if neighbour_data:
                self.cell_histories.update({'nn':[]})
                self.cell_histories.update({'nextnn':[]})
                self.cell_histories.update({'mother':[]})
            if distance_data:
                self.cell_histories.update({'mean_separation':[]})
                self.cell_histories.update({'mean_distance':[]})
            self.cell_histories.update({key:[] for key in self.properties.keys()})
        try: 
            len(idx_list)
            idx_list = np.array(idx_list)
        except TypeError:
            pass            
        for key,valist in self.cell_histories.iteritems():
            if key == 'time':
                try:
                    valist.extend([self.time]*len(idx_list))
                except TypeError:
                    valist.append(self.time)
            elif key == 'cell_ids':
                _add_to_list(valist,self.cell_ids[idx_list])
            elif key == 'age':
                _add_to_list(valist,self.age[idx_list])
            elif key == 'divided':
                _add_to_list(valist,divided)
            elif key == 'area':
                _add_to_list(valist,self.mesh.areas[idx_list])
            elif key == 'mother':
                _add_to_list(valist,self.mother[idx_list])
            elif key == 'nn':
                _add_lists_to_list(valist,self.get_neighbour_cell_ids(idx_list,True))
            elif key == 'nextnn':
                _add_lists_to_list(valist,self.get_next_nearest_neighbour_cell_ids(idx_list,True))
            elif key == 'mean_separation':
                _add_to_list(valist,self.mesh.mean_cell_separation())
            elif key == 'mean_distance':
                _add_to_list(valist,self.mesh.mean_cell_distance())
            elif key == 'position':
                if valist is None:
                    self.cell_histories['position'] = self.mesh.centres[idx_list]
                else:
                    self.cell_histories['position'] = np.vstack((valist,self.mesh.centres[idx_list]))
            else:
                _add_to_list(valist,self.properties[key][idx_list])
        
    def update_extruded_divided_lists(self,idx_list,mother):
        if isinstance(idx_list,int):
            if mother:
                self.divided_cells.append((self.cell_ids[idx_list],self.age[idx_list],self.time))
            else:
                self.extruded_cells.append((self.cell_ids[idx_list],self.age[idx_list],self.time))  
        else:
            if mother is True:
                divided_cells = [(cid,age,self.time) for cid,age in zip(self.cell_ids[idx_list],self.age[idx_list])]
                self.divided_cells.extend(divided_cells)
            elif mother is False: 
                extruded_cells = [(cid,age,self.time) for cid,age in zip(self.cell_ids[idx_list],self.age[idx_list])]
                self.extruded_cells.extend(extruded_cells)    
            else:
                divided_cells = [(cid,age,self.time) for cid,age in zip(self.cell_ids[idx_list[mother]],self.age[idx_list[mother]])]
                extruded_cells = [(cid,age,self.time) for cid,age in zip(self.cell_ids[idx_list[~mother]],self.age[idx_list[~mother]])]
                self.divided_cells.extend(divided_cells)
                self.extruded_cells.extend(extruded_cells)
    
    def remove(self,idx_list,divided=None):
        """remove a cell (or cells) from tissue. if storing dead cell ids need arg mother=True if cell is being removed
        following division, false otherwise. can be list."""
        if self.save_cell_histories:
             self.update_cell_histories(idx_list,divided)
        self.mesh.remove(idx_list)
        self.cell_ids = np.delete(self.cell_ids,idx_list)
        self.age = np.delete(self.age,idx_list)
        self.mother = np.delete(self.mother,idx_list)
        for key,val in self.properties.iteritems():
            self.properties[key] = np.delete(val,idx_list)
        
    def add_daughter_cells(self,i,rand,daughter_properties=None):
        """add pair of new cells after a cell division. copies properties dictionary from mother unless alternative values
        are specified in the daughter_properties argument"""
        angle = rand.rand()*np.pi
        dr = np.array((EPS*np.cos(angle),EPS*np.sin(angle)))
        new_cen1 = self.mesh.centres[i] + dr
        new_cen2 = self.mesh.centres[i] - dr
        self.mesh.add([new_cen1,new_cen2])
        self.cell_ids = np.append(self.cell_ids,[self.next_id,self.next_id+1])
        self.age = np.append(self.age,[0.0,0.0])
        self.mother = np.append(self.mother,[self.cell_ids[i]]*2)
        self.next_id += 2
        for key,val in self.properties.iteritems():
            if daughter_properties is None or key not in daughter_properties: 
                self.properties[key] = np.append(self.properties[key],[self.properties[key][i]]*2)
            else: 
                self.properties[key] = np.append(self.properties[key],daughter_properties[key])  
    
    def add_many_daughter_cells(self,idx_list,rand):
         if len(idx_list)==1: self.add_daughter_cells(idx_list[0],rand)
         else:
             for i in idx_list:
                 self.add_daughter_cells(i,rand)
        
    def dr(self,dt,eta=ETA): 
        """calculate distance cells move due to force law in time dt"""  
        return (dt/eta)*self.Force(self)
    
    def cell_stress(self,i):
        """calculates the stress p_i on a single cell i according to the formula p_i = sum_j mag(F^rep_ij.u_ij)/l_ij
        where F^rep_ij is the repulsive force between i and j, i.e. F^rep_ij=F_ij if Fij is positive, 0 otherwise;
        u_ij is the unit vector between the i and j cell centres and l_ij is the length of the edge between cells i and j"""     
        edge_lengths = self.mesh.edge_lengths(i)
        repulsive_forces = self.Force.force_ij(self,i)
        repulsive_forces[repulsive_forces<0]=0
        return sum(repulsive_forces/edge_lengths) 
    
    def tension_area_product(self,i):
        distances = self.mesh.distances[i]
        forces = self.Force.force_ij(self,i)
        return -0.25*sum(forces*distances)
        
class Force(object):
    """Abstract force object"""
    def force(self):
        """returns (N,2) array floats giving vector force on each cell"""
        raise NotImplementedError()
    
    def magnitude(self,tissue):
        """returns (N,) array floats giving magnitude of force on each cell"""
        return np.sqrt(np.sum(np.sum(self.force(tissue),axis=0)**2))
    
    def __call__(self, tissue):
        return self.force(tissue)
        
class BasicSpringForceTemp(Force):
    
    def __init__(self,mu=MU):
        self.mu=mu
    
    def force(self,tissue):
        return np.array([self.force_i(tissue,i) for i in range(len(tissue))])
    
    def force_i(self):
        """returns force on cell i"""
        raise Exception('force law undefined')

class BasicSpringForceNoGrowth(BasicSpringForceTemp):
    
    def __init__(self,mu=MU,T_m=T_M):
        BasicSpringForceTemp.__init__(self,mu)
        self.T_m=T_m
        if T_m is None:
            self.force_i = self.force_i_no_T_m
            self.force_ij = self.force_ij_no_T_m
    
    def force_i(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        if tissue.age[i] >= self.T_m or tissue.mother[i] == -1: pref_sep = L0
        else: pref_sep = (tissue.mother[n_list]==tissue.mother[i])*((L0-EPS)*tissue.age[i]/self.T_m+EPS-L0) +L0
        return (-self.mu*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1)).sum(axis=0)
    
    def force_i_no_T_m(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        return (-self.mu*vecs*np.repeat((distances-L0)[:,np.newaxis],2,axis=1)).sum(axis=0)
    
    def force_ij(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        if tissue.age[i] >= self.T_m or tissue.mother[i] == -1: pref_sep = L0
        else: pref_sep = (tissue.mother[n_list]==tissue.mother[i])*((L0-EPS)*tissue.age[i]/self.T_m+EPS-L0) +L0
        forces = -self.mu*(distances-pref_sep) 
        return forces
    
    def force_ij_no_T_m(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        forces = -self.mu*(distances-L0) 
        return forces

class BasicSpringForceGrowth(BasicSpringForceTemp):

    def force_i(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        pref_sep = RHO+0.5*GROWTH_RATE*(tissue.age[n_list]+tissue.age[i])
        return (-self.mu*vecs*np.repeat((distances-pref_sep)[:,np.newaxis],2,axis=1)).sum(axis=0)

class SpringForceVariableMu(BasicSpringForceTemp):

    def __init__(self,delta,mu=MU):
        BasicSpringForceTemp.__init__(self,mu)
        self.delta = delta

    def force_i(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        pref_sep = RHO+0.5*GROWTH_RATE*(tissue.age[n_list]+tissue.age[i])
        MU_list = -self.mu*(1-0.5*self.delta*(tissue.properties['mutant'][n_list]+tissue.properties['mutant'][i]))
        return (vecs*np.repeat((MU_list*(distances-pref_sep))[:,np.newaxis],2,axis=1)).sum(axis=0)

class MutantSpringForce(BasicSpringForceTemp):

    def __init__(self,alpha,mu=MU):
        BasicSpringForceTemp.__init__(self,mu)
        self.alpha = alpha

    def force_i(self,tissue,i):
        distances,vecs,n_list = tissue.mesh.distances[i],tissue.mesh.unit_vecs[i],tissue.mesh.neighbours[i]
        pref_sep = RHO+0.5*GROWTH_RATE*(tissue.age[n_list]+tissue.age[i])
        alpha_i = tissue.properties['mutant'][i]*(self.alpha-1)+1
        return (vecs*np.repeat((-self.mu/alpha_i*(distances-pref_sep))[:,np.newaxis],2,axis=1)).sum(axis=0)
        

def _add_to_list(list_1,to_add):
    try: 
        list_1.extend(to_add)
    except TypeError:
        list_1.append(to_add)
        
def _add_lists_to_list(list_1,to_add):
    if isinstance(to_add[0],int):
        list_1.append(to_add)
    else: list_1.extend(to_add)

