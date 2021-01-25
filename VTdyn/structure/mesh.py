import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, ConvexHull
import copy
import os

def polygon_area(points):
    n_p = len(points)
    return 0.5*sum(points[i][0]*points[(i+1)%n_p][1]-points[(i+1)%n_p][0]*points[i][1] for i in range(n_p))

def circumcenter(A,B,C):
    """calculate circumcenter from coords of triangle vertices taken as a (3,2)-array"""
    D = 2*(A[0]*(B[1]-C[1])+B[0]*(C[1]-A[1])+C[0]*(A[1]-B[1]))
    Px = 1./D*((A[0]**2+A[1]**2)*(B[1]-C[1])+(B[0]**2+B[1]**2)*(C[1]-A[1])+(C[0]**2+C[1]**2)*(A[1]-B[1]))
    Py = 1./D*((A[0]**2+A[1]**2)*(C[0]-B[0])+(B[0]**2+B[1]**2)*(A[0]-C[0])+(C[0]**2+C[1]**2)*(B[0]-A[0]))
    return Px,Py
    
class Geometry(object):
    """Abstract Geometry object needed for Mesh."""
    
    def periodise(self,r):
        """returns coordinate r accounting for bc's"""
        raise NotImplementedError()
    
    def periodise_list(self,r):
        """returns list of coords accounting for bc's"""
        raise NotImplementedError()
         
    def retriangulate(self,centres,N_mesh):
        """Takes coordinates of set of points (centres) and number of points as arguments
        Performs Voronoi Tessellation (if calculating cell areas) or Delaunay Triangulation.
        Returns: 
            neighbours: N-dim list of (k,) arrays giving neighbour ids of each cell (k=neighbour number), 
            distances: N-dim list of (k,) arrays giving distances between each cell and its neighbours,
            sep_vector: N-dim list of (k,2) arrays giving unit vectors between each cell and its neighbours, 
            (areas: (N,) array giving area of each cell)
        """
        raise NotImplementedError()
    
    def distance(self,r0,r1):   
        """returns distance between two points"""
        raise NotImplementedError()
    

#
# class Plane(Geometry):
#
#     def retriangulate(self,centres,N_mesh):
#         vor = Voronoi(centres)
#         pairs = vor.ridge_points
#         neighbours = [pairs[loc[0],1-loc[1]] for loc in (np.where(pairs==k) for k in xrange(N_mesh))]
#         sep_vectors = [centres[i]-centres[n_cell] for i,n_cell in enumerate(neighbours)]
#         distances = [np.sqrt((cell_vectors*cell_vectors).sum(axis=1)) for cell_vectors in sep_vectors]
#         sep_vectors = [cell_vectors/np.repeat(cell_distances[:,np.newaxis],2,axis=1) for cell_distances,cell_vectors in zip(distances,sep_vectors)]
#         neighbours = [n_set for n_set in neighbours]
#         areas = np.abs([polygon_area(vor.vertices[polygon]) for polygon in np.array(vor.regions)[vor.point_region]])
#         return neighbours, distances, sep_vector, areas

class Torus(Geometry):
    """Square domain with periodic boundary conditions"""
    
    def __init__(self,width,height):
        """width and height of periodicity"""
        self.width = width
        self.height = height
    
    def __str__(self):
        return 'torus: width=%.5f, height=%.5f'%(self.width,self.height)
        
    def periodise(self,coords):
        half_width, half_height = self.width/2., self.height/2.
        for i,L in enumerate((half_width,half_height)):
            if coords[i] >= L: coords[i] -= L*2
            elif coords[i] < -L: coords[i] += L*2
        return coords
        
    def periodise_list(self,coords):
        half_width, half_height = self.width/2., self.height/2.
        for i,L in enumerate((half_width,half_height)):
            coords[np.where(coords[:,i] >= L)[0],i] -= L*2
            coords[np.where(coords[:,i] < -L)[0],i] += L*2
        return coords
    
    def retriangulate(self,centres,N_mesh):
        width,height = self.width, self.height
        centres_3x3 = np.reshape([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]],(9*N_mesh,2))
        vor = Voronoi(centres_3x3)
        pairs = vor.ridge_points
        neighbours = [pairs[loc[0],1-loc[1]] for loc in (np.where(pairs==k) for k in xrange(4*N_mesh,5*N_mesh))]
        sep_vectors = [centres[i]-centres_3x3[n_cell] for i,n_cell in enumerate(neighbours)]
        distances = [np.sqrt((cell_vectors*cell_vectors).sum(axis=1)) for cell_vectors in sep_vectors]
        sep_vectors = [cell_vectors/np.repeat(cell_distances[:,np.newaxis],2,axis=1) for cell_distances,cell_vectors in zip(distances,sep_vectors)]
        neighbours = [n_set%N_mesh for n_set in neighbours] 
        areas = np.abs([polygon_area(vor.vertices[polygon]) for polygon in np.array(vor.regions)[vor.point_region][4*N_mesh:5*N_mesh]])
        return neighbours, distances, sep_vectors, areas
    
    def distance(self,r0,r1):
        delta = np.abs(r0-r1)
        delta[:,0] = np.min((delta[:,0],self.width-delta[:,0]),axis=0)
        delta[:,1] = np.min((delta[:,1],self.height-delta[:,1]),axis=0)
        return np.sqrt((delta ** 2).sum(axis=1))
        
    def distance_squared(self,r0,r1):
        delta = np.abs(r0-r1)
        delta[:,0] = np.min((delta[:,0],self.width-delta[:,0]),axis=0)
        delta[:,1] = np.min((delta[:,1],self.height-delta[:,1]),axis=0)
        return (delta ** 2).sum(axis=1)
        
    def tri_area(self,triangle):
        sides = self.distance(triangle,np.roll(triangle,1,axis=0))
        p = 0.5*np.sum(sides)
        return np.sqrt(p*(p-sides[0])*(p-sides[1])*(p-sides[2]))
          

class TorusNoArea(Torus):
    """same as Torus geometry but does not calculate cell areas (overides retriangulate)"""           
    def retriangulate(self,centres,N_mesh):
        width,height = self.width,self.height
        centres_3x3 = np.reshape([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]],(9*N_mesh,2))
        vnv = Delaunay(centres_3x3).vertex_neighbor_vertices
        neighbours = [vnv[1][vnv[0][k]:vnv[0][k+1]] for k in xrange(4*N_mesh,5*N_mesh)]
        sep_vectors = [centres[i]-centres_3x3[n_cell] for i,n_cell in enumerate(neighbours)]
        distances = [np.linalg.norm(cell_vectors,axis=1) for cell_vectors in sep_vectors]
        sep_vectors = [cell_vectors/np.repeat(cell_distances[:,np.newaxis],2,axis=1) for cell_distances,cell_vectors in zip(distances,sep_vectors)]
        neighbours = [n_set%N_mesh for n_set in neighbours] 

        return neighbours,distances,sep_vectors
        
# class Cylinder(Geometry):
#     def __init__(self,width):
#         self.width = width
#
#     def periodise(self,coords):
#         half_width = self.width/2.
#         if coords[0] >= half_width: coords[i] -= half_width*2
#         elif coords[0] < -half_width: coords[i] += half_width*2
#         return coords
#
#     def periodise_list(self,coords):
#         half_width = self.width/2.
#         coords[np.where(coords[:,0] >= half_width)[0],i] -= half_width*2
#         coords[np.where(coords[:,0] < -half_width)[0],i] += half_width*2
#         return coords


class Mesh(object):
    
    """ 
    keeps track of cell positions and neighbour relations and defines methods for moving cells and updating
    Attributes: N_cells = number of cells 
                centres = array of (x,y) values for both cell and ghost node positions
                geometry = Geometry object, e.g. Torus
                neighbours, distances, unit_vecs, areas (see Geometry class)
    """
   
    def __init__(self,centres,geometry):
        """Parameters:
        centres: (N,2) array floats
            positions of cells
        geometry: Geometry object 
        """
        self.N_mesh = len(centres)
        self.centres = centres
        self.geometry = geometry
        self.neighbours,self.distances,self.unit_vecs, self.areas = self.retriangulate()
    
    def __len__(self):
        return self.N_mesh
    
    def write(self,outdir):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        np.savetxt(outdir+'/centres',self.centres)
        with open(outdir+'/geometry','w') as f:
            f.write(str(self.geometry))
    
    def copy(self):
        """create a copy of Mesh object"""
        meshcopy = copy.copy(self)
        meshcopy.centres = copy.copy(meshcopy.centres)
        return meshcopy
    
    def next_nearest_neighbours(self,i):
        return np.array(list(set([k for j in self.neighbours[i] for k in self.neighbours[j]])))
    
    def update(self):
        """recalculate and define mesh attributes"""
        self.N_mesh = len(self.centres)
        self.neighbours, self.distances, self.unit_vecs, self.areas = self.retriangulate()
        
    def retriangulate(self):
        return self.geometry.retriangulate(self.centres,self.N_mesh)
        
    def move(self, i, dr):
        """move cell i by dr"""
        self.centres[i] = self.geometry.periodise(self.centres[i]+dr)
    
    def move_all(self, dr_array):
        """move all N cells by vectors given by (N,2) dr_array """
        self.centres = self.geometry.periodise_list(self.centres + dr_array)
    
    def add(self,pos):
        """add new cell centre"""
        self.centres = np.append(self.centres,pos,0)
    
    def remove(self,i):
        """remove cell centre"""
        self.centres = np.delete(self.centres,i,0)
        
    def voronoi(self):
        return Voronoi(self.centres)
    
    def convex_hull(self):
        return ConvexHull(self.centres())  

    def delaunay(self):
        return Delaunay(self.centres)
        
    def distances(self,i):
        """get distances between cell i and its neighbours"""
        return self.geometry.distance(self.centres[i],self.centres)
        
    def local_density(self):
        return 1./self.areas + np.array([sum(1./self.areas[neighbours]) for neighbours in self.neighbours])
    
    def cell_local_density_radius(self,R,i):
        return np.sum(self.geometry.distance_squared(self.centres,self.centres[i])<R**2)/(np.pi*R**2)
    
    def local_density_radius(self,R):
        return [self.cell_local_density(R,i) for i in range(self.N_mesh)]

    def triangle_areas(self,triples):
        """returns list of areas of each triangle in DT"""
        return [self.geometry.tri_area(self.centres[triple]) for triple in triples]
                
    def triples(self):
        """returns list of triples corresponding to triangles in DT"""
        triples = np.array([sorted([i,j,k]) for i in range(self.N_mesh) for j in self.neighbours[i] for k in self.neighbours[j] if (k!=i and k in self.neighbours[i])]) 
        return np.unique(triples,axis=0)
        
    def edge_lengths(self,i):
        """returns list of edge lengths (corresponding to interface between cells in neighbour list) for a Voronoi cell i"""
        neighbour_pairs = np.array([sorted([j,k]) for j in self.neighbours[i] for k in self.neighbours[j] if (k!=i and k in self.neighbours[i])])
        neighbour_pairs = np.unique(neighbour_pairs,axis=0)
        vertices = [circumcenter(self.centres[i],*self.centres[pair]) for pair in neighbour_pairs]
        edge_lengths = np.array([self.edge_length_ij(j,neighbour_pairs,vertices) for j in self.neighbours[i]])
        return edge_lengths
    
    def voronoi_vertices(self,i):
        neighbour_pairs = np.array([sorted([j,k]) for j in self.neighbours[i] for k in self.neighbours[j] if (k!=i and k in self.neighbours[i])])
        neighbour_pairs = np.unique(neighbour_pairs,axis=0)
        vertices = [circumcenter(self.centres[i],*self.centres[pair]) for pair in neighbour_pairs]
        return vertices
        
    def edge_length_ij(self,j,neighbour_pairs_of_i,vertices_of_i):
        vertex_id_locations = np.where(neighbour_pairs_of_i==j)
        edge_vector = np.array(vertices_of_i[vertex_id_locations[0][0]])-np.array(vertices_of_i[vertex_id_locations[0][1]])
        return (edge_vector[0]**2+edge_vector[1]**2)**0.5  

    def mean_cell_separation(self):
        return np.mean([np.mean(distance) for distance in self.distances])
    
    def mean_cell_distance(self):
        return np.mean([np.mean(self.geometry.distance(centre,np.delete(self.centres,i,0))) for i,centre in enumerate(self.centres)])
       
        
class MeshNoArea(Mesh):
    def __init__(self,centres,geometry):
        self.N_mesh = len(centres)
        self.centres = centres
        self.geometry = geometry
        self.neighbours,self.distances,self.unit_vecs = self.retriangulate()
    
    def update(self):
        self.N_mesh = len(self.centres)
        self.neighbours, self.distances, self.unit_vecs = self.retriangulate()

    def retriangulate(self):
        return self.geometry.retriangulate(self.centres,self.N_mesh)
        
    def local_density(self):
        triples = self.triples()
        triangle_areas = self.triangle_areas(triples)
        local_density = np.zeros(self.N_mesh)
        for triple,area in zip(triples,triangle_areas):
            for i in triple:
                local_density[i]+=1./area
        return local_density        

        