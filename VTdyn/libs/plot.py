import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, PatchCollection
import matplotlib.patches as patches
import seaborn as sns
from shapely.ops import polygonize
from shapely.geometry import LineString, MultiPolygon, Polygon, MultiPoint, Point, LinearRing
from scipy.spatial import Voronoi, Delaunay
from descartes.patch import PolygonPatch
import os

A0 = np.sqrt(3)/2
beige = '#F4EDD6'
DEFAULT_PALETTE = np.array((beige,'#688A87','#FF70C3'))
DELTA = 0.025

#functions for plotting and animating tissue objects/histories

def set_heatmap_colours(heat_map,palette=None,return_palette_only=False):
    if 'bins' in heat_map: bins = heat_map['bins']
    else: bins = 100
    if palette is None:
            palette = np.array(sns.cubehelix_palette(bins,start=.5,rot=-.2))
    if return_palette_only: 
        return palette
    else:
        data = heat_map['data']
        if 'lims' in heat_map: dmin,dmax = heat_map['lims']
        else: dmin,dmax = np.min(data),np.max(data)
        bin_bounds = np.arange(bins)*(dmax-dmin)/bins+dmin
        data_bins = np.digitize(data,bin_bounds)-1
        return palette[data_bins],palette,bin_bounds

def set_key_palette(key,history=None,tissue=None,palette=None):
    if palette is not None: return palette
    try:
        if history is not None:  
            key_max = max((max(tissue.properties[key]) for tissue in history))
        elif tissue is not None:
            key_max = max(tissue.properties[key])
        else: 
            raise TypeError('history or tissue argument must be given')
    except KeyError:
        key_max = len(DEFAULT_PALETTE)
    if key_max>len(DEFAULT_PALETTE):
        palette = np.array(sns.color_palette("husl", key_max+1))
        np.random.shuffle(palette)
    else: palette = DEFAULT_PALETTE
    return palette

def plot_colour_bar(fig,palette,bin_bounds):
    ax=fig.add_axes((0.2,0.1,0.6,0.03))
    ax.set_ylim([0,1])
    ax.set_xlim([bin_bounds[0],bin_bounds[-1]])
    ax.set_yticks([])
    n=len(palette)
    dx=bin_bounds[1]-bin_bounds[0]
    rects = (patches.Rectangle((x,0.), dx, 1.,
                      color=c) for x,c in zip(bin_bounds[:-1],palette))
    for r in rects:
        ax.add_patch(r)
            
def plot_tri_torus(tissue,fig=None,ax=None,time = None,label=False,node_colour='k',line_colour='k',lw=2):
    """plot Delaunay triangulation of a tissue object (i.e. cell centres and neighbour connections) with torus geometry"""
    width, height = tissue.mesh.geometry.width, tissue.mesh.geometry.height 
    if ax is None: 
        if fig is None: fig = plt.figure()
        ax = fig.add_subplot(111)
        minx, miny, maxx, maxy = -width/2,-height/2,width/2,height/2
        w, h = maxx - minx, maxy - miny
        ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
        ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
        ax.set_aspect(1)
        ax.set_axis_off()
    centres = tissue.mesh.centres
    centres_3x3 = np.vstack([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]])
    tri = Delaunay(centres_3x3).simplices  
    plt.triplot(centres_3x3[:,0], centres_3x3[:,1], tri.copy(),color=line_colour,lw=lw)
    plt.plot(centres_3x3[:,0], centres_3x3[:,1], 'o',color = node_colour)
    if label:
        for i, coords in enumerate(tissue.mesh.centres):
            plt.text(coords[0],coords[1],str(i))
    if time is not None:
        lims = plt.axis()
        plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time)
    return ax

def plot_centres(tissue,ax=None,time = None,label=False,palette=DEFAULT_PALETTE):
    width, height = tissue.mesh.geometry.width, tissue.mesh.geometry.height 
    if ax is None: 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        minx, miny, maxx, maxy = -width/2,-height/2,width/2,height/2
        w, h = maxx - minx, maxy - miny
        ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
        ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
        ax.set_aspect(1)
        ax.set_axis_off()
    centres = tissue.mesh.centres
    plt.plot(centres[:,0], centres[:,1], '.',color = 'black')
    if label:
        for i, coords in enumerate(tissue.mesh.centres):
            plt.text(coords[0],coords[1],str(i))
    if time is not None:
        lims = plt.axis()
        plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time)
    plt.show()

def create_axes(tissue,figsize=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    set_ax(tissue,ax)
    return fig,ax

def set_ax(tissue,ax):
    width, height = tissue.mesh.geometry.width, tissue.mesh.geometry.height 
    minx, miny, maxx, maxy = -width/2,-height/2,width/2,height/2
    w, h = maxx - minx, maxy - miny
    ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
    ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
    ax.set_aspect(1)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

def multi_torus_plot(tissue_list,nrows,ncols,param_name=None,param_vals=None,fmt='%.1f',figsize=None):
    fig,axes = plt.subplots(nrows,ncols,figsize=figsize)
    for ax,tissue,param in zip(axes.flatten(),tissue_list,param_vals):
        set_ax(tissue,ax)
        torus_plot(tissue,ax=ax,lw=1.5)
        if param_name is not None and param_vals is not None:
            ax.set_title(param_name+fmt%param)
    plt.subplots_adjust(left=0.0,right=1.0,bottom=0.0,top=1.0,wspace=0.0,hspace=0.0)

def plot_neighbours_of_most_recent_deaths(tissue,n,ax,palette=None):
    if tissue.cell_histories != {}:
        if palette is None:
            palette = sns.cubehelix_palette(n, start=0.4,dark=0,rot=0,light=0.5,reverse=True)
        centres=tissue.mesh.centres
        divided = np.flip(tissue.cell_histories['divided'])
        times = np.flip(tissue.cell_histories['time'])
        nn = tissue.cell_histories['nn'][::-1]
        nn = [neighbours for neighbours,is_divided,time in zip(nn,divided,times) if not is_divided and time < tissue.time][:n]
        for neighbours,color in zip(nn,palette):
            neighbour_ids = [np.where(tissue.cell_ids==n)[0][0] for n in neighbours if n in tissue.cell_ids]
            centres_to_plot = centres[np.array(neighbour_ids)]
            plt.plot(centres_to_plot[:,0], centres_to_plot[:,1], 'o',color=color)  

def plot_recent_divisions(tissue,n,ax,palette=None):
    if tissue.cell_histories != {}:
        if palette is None:
            palette = sns.cubehelix_palette(n, start=0.4,dark=0,rot=0,light=0.5,reverse=True)
        centres=np.flip(tissue.mesh.centres,axis=0)
        ages = np.flip(tissue.age)
        for color,centre1,centre2,age1,age2 \
                in zip(palette,centres[::2],np.roll(centres,-1,axis=0)[::2],ages[::2],np.roll(ages,-1)[::2]):
            if age1!=age2:
                break
            else:
                dx,dy = centre2-centre1
                ax.arrow(centre1[0],centre1[1],dx,dy,head_width=0.25,head_length=0.2,overhang=0.7,color=color,length_includes_head=True)
                ax.arrow(centre2[0],centre2[1],-dx,-dy,head_width=0.25,head_length=0.2,overhang=0.7,color=color,length_includes_head=True)
    
def torus_plot(tissue,palette=None,key=None,key_label=None,ax=None,fig=None,show_centres=False,cell_ids=False,mesh_ids=False,areas=False,boundary=False,colours=None,animate=False,
                heat_map=None,plot_vals=None,textcolor='black',figsize=None,lw=2.5,edgecolor='k',time=False,threshold_area=None,
                neighbours_of_recent_deaths=None,recent_divisions=None,fitness=False,game=None,game_constants=None):
    """plot tissue object with torus geometry
    args
    ---------------
    tissue: to plot,r'
    palette: array of colors
    key: (str) color code cells by key. must match a tissue.properties key, e.g. type, ancestors
    key_label: (bool) plot key vals if True
    ax: matplotlib axes
    show_centres: (bool) plot cell centres if True, or specific ids if list
    cell_ids/mesh_ids: (bool) label with cell/mesh ids if True
    areas: (bool) label with cell areas if True
    boundary: (bool) if True plot square boundary over which tissue is periodic
    colours: (array) provide colours for specific key values (if key not None) OR colour of each cell 
    animate: (bool) True if plot is part of animation
    heat_map: (dict) 
        heat_map['data'] is array of floats with data val for each cell
        options to provide 'palette', 'bins':(int) and 'lims':(float,float)
    plot_vals: (array,str) array gives data val for each cell, str gives format to convert to text
    fitness: (bool) is True must provide game and game_constants to calculate fitness
    """
    width, height = tissue.mesh.geometry.width, tissue.mesh.geometry.height 
    centres = tissue.mesh.centres 
    centres_3x3 = np.vstack([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]])
    N = tissue.mesh.N_mesh
    mask = np.full(9*N,False,dtype=bool)
    mask[4*N:5*N]=True
    vor = Voronoi(centres_3x3)
    
    mp = MultiPolygon([Polygon(vor.vertices[region])
                for region in (np.array(vor.regions)[np.array(vor.point_region)])[mask]])
    
    if ax is None: 
        fig,ax=create_axes(tissue,figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    if fitness:
        heat_map = {'data':'fitness'}
    if key is None and colours is None and heat_map is None:
        if palette is not None: c = palette[0]
        else: c = beige
        ax.add_collection(PatchCollection([PolygonPatch(p,linewidth=lw,edgecolor=edgecolor,facecolor=c) for p in mp],match_original=True))
    else:
        if key is not None:
            if key == 'CIP':
                tissue.properties['CIP'] = (tissue.mesh.areas>threshold_area)*1
            if palette is None: palette = set_key_palette(key,tissue=tissue)
            colours = palette[tissue.properties[key]]                     
        elif heat_map is not None:
            if heat_map['data']=='fitness':
                heat_map['data'] = recalculate_fitnesses(tissue.mesh.neighbours,tissue.properties['type'],DELTA,game,game_constants)
            colours,palette,bin_bounds = set_heatmap_colours(heat_map,palette)
            if 'show_cbar' in heat_map:
                if heat_map['show_cbar']:
                    plot_colour_bar(fig,palette,bin_bounds)
            elif not animate: plot_colour_bar(fig,palette,bin_bounds)
            
        coll = PatchCollection([PolygonPatch(p,facecolor = c,linewidth=lw,edgecolor=edgecolor) for p,c in zip(mp,colours)],match_original=True)
        ax.add_collection(coll) 
    if show_centres: 
        if show_centres is True:
            plt.plot(centres[:,0], centres[:,1], 'o',color='black')
        else:
            centres_to_plot = centres[np.array(show_centres)]
            plt.plot(centres_to_plot[:,0], centres_to_plot[:,1], 'o',color='black')
    if cell_ids:
        ids = tissue.cell_ids
        for i, coords in enumerate(tissue.mesh.centres):
            ax.text(coords[0],coords[1],str(ids[i]),color=textcolor,ha='center',va='center')
    if mesh_ids:
        for i, coords in enumerate(tissue.mesh.centres):
            ax.text(coords[0],coords[1],str(i),color=textcolor,ha='center',va='center')
    if areas:
        for area, coords in zip(tissue.mesh.areas,tissue.mesh.centres):
            ax.text(coords[0],coords[1],'%.2f'%area,color=textcolor,ha='center',va='center')
    if key_label is True:
        for id, coords in zip(tissue.properties[key],tissue.mesh.centres):
            ax.text(coords[0],coords[1],str(id),color=textcolor,ha='center',va='center')
    if plot_vals is not None:
        data,fmt = plot_vals[0],plot_vals[1]
        for val,coords in zip(data,tissue.mesh.centres):
            ax.text(coords[0],coords[1],fmt%val,color=textcolor,ha='center',va='center')
    if boundary: 
        ax.add_patch(patches.Rectangle((-width/2,-height/2),width,height,fill=False,linewidth=1.5))
    if time:
        ax.text(0.5,0.,r'$t=%5.1f$ hours'%(tissue.time),transform=ax.transAxes,ha="center")
    if neighbours_of_recent_deaths is not None:
        plot_neighbours_of_most_recent_deaths(tissue,neighbours_of_recent_deaths,ax)
    if recent_divisions is not None:
        plot_recent_divisions(tissue,recent_divisions,ax)
    if not animate: return ax

def animate_torus(history, key = None, heat_map=None, savefile=None, index=None, delete_images=True,imagedir='images',pause=0.001,palette=None,time=False,**kwargs):
    """view animation of tissue history with torus geometry
    args: 
    --------------- 
    history: list of tissue objects
    key: (str) color code cells by key. must match a tissue.properties key, e.g. type, ancestors
    """
    if not os.path.exists(imagedir): # if the outdir doesn't exist create it
         os.makedirs(imagedir)
    width,height = history[0].mesh.geometry.width,history[0].mesh.geometry.height
    plt.ion()
    fig,ax = create_axes(history[0])
    ax.set_autoscale_on(False)
    plot = []
    if savefile is not None:
        frames=[]
        i = 0
    if key is not None:
        palette = set_key_palette(key,history=history,palette=palette)
    elif heat_map is not None: 
        palette = set_heatmap_colours(heat_map,return_palette_only=True)
    heat_map_multi = heat_map
    for i,tissue in enumerate(history):
        if heat_map_multi is not None:
            heat_map = {key:val if key!='data' else val[i] for key,val in heat_map_multi.iteritems()}
            if heat_map['show_cbar'] is not None or i>=1: heat_map['show_cbar']=None
        try:
            for c in ax.collections:
                c.remove()
        except TypeError: pass
        try:
            for t in ax.texts:
                t.remove()
        except TypeError: pass
        torus_plot(tissue,palette,key=key,heat_map=heat_map,ax=ax,fig=fig,animate=True,time=time,**kwargs)
        if savefile is not None:
            frame="%s/image%05i.png" % (imagedir,i)
            fig.savefig(frame,dpi=500)
            frames.append(frame)
            i+=1
        else: plt.pause(pause)
    if savefile is not None:
        if index is not None: savefile += str(index)
        os.system("mencoder 'mf://%s/image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o %s.mpg" %(imagedir,savefile))
        if delete_images:
            for frame in frames: os.remove(frame)

def plot_spring(r1,r2,ax,colour='k',lw=1.5):
    x,y=r2-r1
    x0,y0 = r1
    L = np.sqrt(x**2+y**2)
    spring_radius, number_turns = 0.05,12
    Np = 1000 #number points to plot
    pad = 100
    w = np.linspace(0,L,Np)
    xp=np.zeros(Np)
    xp[pad:-pad] = spring_radius * np.sin(2*np.pi*number_turns*w[pad:-pad]/L)
    R = np.array([[y/L,-x/L],[x/L,y/L]]).T
    xs,ys = np.matmul(R,np.vstack((xp,w)))
    xs += x0
    ys += y0
    ax.plot(xs,ys,c=colour,lw=lw)

def plot_springs(tissue,ax,color='k'):
    mesh = tissue.mesh
    for i in range(len(tissue)):
        r1 = mesh.centres[i]
        for j,r2 in zip(mesh.neighbours,mesh.centres):
            if i<j:
                plot_spring(r1,r2,ax,color)
                

######## FITNESS FUNCTIONS ##########


def prisoners_dilemma_averaged(cell_type,neighbour_types,b,c):
    """calculate average payoff for single cell"""
    return -c*cell_type+b*np.sum(neighbour_types)/len(neighbour_types)

def prisoners_dilemma_accumulated(cell_type,neighbour_types,b,c):
    """calculate accumulated payoff for single cell"""
    return -c*cell_type*len(neighbour_types)+b*np.sum(neighbour_types)

def N_person_prisoners_dilemma(cell_type,neighbour_types,b,c):
    return -c*cell_type + b*(np.sum(neighbour_types)+cell_type)/(len(neighbour_types)+1)

def volunteers_dilemma(cell_type,neighbour_types,b,c,M):
    return -c*cell_type +b*((np.sum(neighbour_types)+cell_type)>=M)

def benefit_function_game(cell_type,neighbour_types,benefit_function,benefit_function_params,b,c):
    return -c*cell_type + b*benefit_function(np.sum(neighbour_types)+cell_type,len(neighbour_types)+1,*benefit_function_params)

def logistic_benefit(j,Ng,s,k):
    return (logistic_function(j,Ng,s,k)-logistic_function(0,Ng,s,k))/(logistic_function(Ng,Ng,s,k)-logistic_function(0,Ng,s,k))

def logistic_function(j,Ng,s,k):
    return 1./(1.+np.exp(s*(k-j)/Ng))

def get_fitness(cell_type,neighbour_types,DELTA,game,game_constants):
    """calculate fitness of single cell"""
    return 1+DELTA*game(cell_type,neighbour_types,*game_constants)

def recalculate_fitnesses(neighbours_by_cell,types,DELTA,game,game_constants):
    """calculate fitnesses of all cells"""
    return np.array([get_fitness(types[cell],types[neighbours],DELTA,game,game_constants) 
                        for cell,neighbours in enumerate(neighbours_by_cell)])

################################

#plotting functions for a PLANAR geometry tissue object IN DEVELOPMENT   
    
# def finite_plot(tissue,palette=DEFAULT_PALETTE,key=None,key_label=None,ax=None,show_centres=False,cell_ids=False,mesh_ids=False):
#     centres = tissue.mesh.centres
#     vor = tissue.mesh.voronoi()
#     center = vor.points.mean(axis=0)
#     ptp_bound = vor.points.ptp(axis=0)
#
#     regions = [Polygon(vor.vertices[region]) if -1 not in region else
#                 Polygon(get_region_for_infinite(region,vor,center,ptp_bound))
#                 for region in np.array(vor.regions)[np.array(vor.point_region)] if len(region)>=2
#                 ]
#     convex_hull = MultiPoint([Point(i) for i in centres]).convex_hull
#     mp = MultiPolygon(
#         [poly.intersection(convex_hull) for poly in regions])
#
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         minx, miny, maxx, maxy = mp.bounds
#         w, h = maxx - minx, maxy - miny
#         ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
#         ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
#         ax.set_aspect(1)
#
#     if key is None: ax.add_collection(PatchCollection([PolygonPatch(p) for p in mp]))
#     else:
#         colours = palette[tissue.by_mesh(key)]
#         coll = PatchCollection([PolygonPatch(p,facecolor = c) for p,c in zip(mp,colours)],match_original=True)
#         # coll.set_facecolors(palette[tissue.by_mesh(key)])
#         ax.add_collection(coll)
#
#     if show_centres:
#         plt.plot(centres[:,0], centres[:,1], 'o',color='black')
#     if cell_ids:
#         ids = tissue.by_mesh('id')
#         for i, coords in enumerate(tissue.mesh.centres):
#             plt.text(coords[0],coords[1],str(ids[i]))
#     if mesh_ids:
#         for i, coords in enumerate(tissue.mesh.centres):
#             plt.text(coords[0],coords[1],str(i))
#     if key_label is not None:
#         ids = tissue.by_mesh(key_label)
#         for i, coords in enumerate(tissue.mesh.centres):
#             plt.text(coords[0],coords[1],str(ids[i]))
#
#
# def get_vertices_for_inf_line(vor,i,pointidx,center,ptp_bound):
#     t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
#     t /= np.linalg.norm(t)
#     n = np.array([-t[1], t[0]])  # normal
#
#     midpoint = vor.points[pointidx].mean(axis=0)
#     direction = np.sign(np.dot(midpoint - center, n)) * n
#     far_point = vor.vertices[i] + direction * ptp_bound.max()
#
#     return far_point
#
# def get_region_for_infinite(region,vor,center,ptp_bound):
#     infidx= region.index(-1)
#     finite_end1 = region[infidx-1]
#     finite_end2 = region[(infidx+1)%len(region)]
#
#     try: pointidx1 = vor.ridge_points[vor.ridge_vertices.index([-1,finite_end1])]
#     except ValueError: pointidx1 = vor.ridge_points[vor.ridge_vertices.index([finite_end1,-1])]
#     far_point1 = get_vertices_for_inf_line(vor,finite_end1,pointidx1,center,ptp_bound)
#
#     try: pointidx2 = vor.ridge_points[vor.ridge_vertices.index([-1,finite_end2])]
#     except ValueError: pointidx2 = vor.ridge_points[vor.ridge_vertices.index([finite_end2,-1])]
#     far_point2 = get_vertices_for_inf_line(vor,finite_end2,pointidx2,center,ptp_bound)
#     region_vertices = []
#     for pt in region:
#         if pt != -1: region_vertices.append(vor.vertices[pt])
#         else: region_vertices.append(far_point1); region_vertices.append(far_point2)
#     return np.array(region_vertices)
#
#
# def plot_cells(tissue,DEFAULT_PALETTE=DEFAULT_PALETTE,key=None,ax=None,label=False,time = False,colors=None,centres=True):
#     ghosts = tissue.by_mesh('ghost')
#     fig = plt.Figure()
#     if ax is None:
#         ax = plt.axes()
#         plt.axis('scaled')
#         xmin,xmax = min(tissue.mesh.centres[:,0]), max(tissue.mesh.centres[:,0])
#         ymin,ymax = min(tissue.mesh.centres[:,1]), max(tissue.mesh.centres[:,1])
#         ax.set_xlim(xmin,xmax)
#         ax.set_ylim(ymin,ymax)
#         ax.xaxis.set_major_locator(plt.NullLocator())
#         ax.yaxis.set_major_locator(plt.NullLocator())
#     ax.cla()
#     vor = tissue.mesh.voronoi()
#     cells_by_vertex = np.array(vor.regions)[np.array(vor.point_region)]
#     verts = [vor.vertices[cv] for cv in cells_by_vertex[~ghosts]]
#     if colors is not None:
#         coll = PolyCollection(verts,linewidths=[2.],facecolors=colors)
#     elif key is not None:
#         colors = np.array(DEFAULT_PALETTE)[tissue.by_mesh(key)]
#         coll = PolyCollection(verts,linewidths=[2.],facecolors=colors)
#     else: coll = PolyCollection(verts,linewidths=[2.])
#     ax.add_collection(coll)
#     if label:
#         ids = tissue.by_mesh('id')
#         for i, coords in enumerate(tissue.mesh.centres):
#             if ~ghosts[i]: plt.text(coords[0],coords[1],str(ids[i]))
#     if time:
#         lims = plt.axis()
#         plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time)
#     if centres:
#         real_centres = tissue.mesh.centres[~ghosts]
#         plt.plot(real_centres[:,0], real_centres[:,1], 'o',color='black')
#
#
# def animate_finite(history, key = None, timestep=None):
#     xmin,ymin = np.amin([np.amin(tissue.mesh.centres,axis=0) for tissue in history],axis=0)*1.5
#     xmax,ymax = np.amax([np.amax(tissue.mesh.centres,axis=0) for tissue in history],axis=0)*1.5
#
#     plt.ion()
#     fig = plt.figure()
#     ax = plt.axes()
#     plt.axis('scaled')
#     ax.set_xlim(xmin,xmax)
#     ax.set_ylim(ymin,ymax)
#     fig.set_size_figsize(6, 6)
#     ax.set_autoscale_on(False)
#     plot = []
#     if key is not None:
#         key_max = max((max(tissue.by_mesh(key)) for tissue in history))
#         palette = np.array(sns.color_palette("husl", key_max+1))
#         np.random.shuffle(palette)
#         for tissue in history:
#             ax.cla()
#             finite_plot(tissue,palette,key,ax=ax)
#             plt.pause(0.001)
#     else:
#         for tissue in history:
#             ax.cla()
#             finite_plot(tissue,ax=ax)
#             plt.pause(0.001)
#
#
# def animate_video_mpg(history,name,index,facecolours='Default'):
#     v_max = np.max((np.max(history[0].mesh.centres), np.max(history[-1].mesh.centres)))
#     if key: key_max = np.max(history[0].properties[key])
#     size = 2.0*v_max
#     outputdir="images"
#     if not os.path.exists(outputdir): # if the folder doesn't exist create it
#         os.makedirs(outputdir)
#     fig = plt.figure()
#     ax = plt.axes()
#     plt.axis('scaled')
#     lim = [-0.55*size, 0.55*size]
#     ax.set_xlim(lim)
#     ax.set_ylim(lim)
#     frames=[]
#     i = 0
#     for cells in history:
#         plot_cells(cells,key,ax)
#         i=i+1
#         frame="images/image%04i.png" % i
#         fig.savefig(frame,dpi=500)
#         frames.append(frame)
#     os.system("mencoder 'mf://images/image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o " + "%s%0.3f.mpg" %(name,index))
#     for frame in frames: os.remove(frame)
#