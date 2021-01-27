import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product
from scipy.special import binom
import coop_neighbour_stats as cns

PAGEWIDTH = 5
DELTA = 0.025
Z=100

def get_coop_coop_neighbour_dist(df):
    """return probability distribution for number of coop neighbours for a coop/defector (type 1/0) cell as dataframe.
            depends on k (# neighbours) and n (total coop cells in population)"""
    j_dist = df.groupby(['n','k','type'])['j'].value_counts(normalize=True).sort_index()
    return j_dist.reset_index(name='j_freq')

def compare_sim_type0(j_dist):
    type0_sim = j_dist.set_index(['type','n','k','j'])['j_freq'].loc[0].to_frame().assign(type='sim')
    type0_pred = j_dist[j_dist.type==1].drop('type',axis=1).assign(type='pred')
    type0_pred['n'] = Z - type0_pred['n']
    type0_pred['j'] = type0_pred['k'] - type0_pred['j']
    type0_pred = type0_pred.set_index(['n','k','j'])
    return pd.concat([type0_pred,type0_sim]).reset_index().sort_values(['n','k','j']).reset_index()

def plot_colorbar(cmap,jmax,savename):
    fig,ax=plt.subplots(figsize=(1.5,0.4))
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,jmax),cmap=cmap),
                    cax=ax,orientation='horizontal')
    cbar.set_ticks([0,jmax])
    cbar.set_ticklabels(['j=0',jmax])
    plt.subplots_adjust(bottom=0.6)
    plt.savefig(savename,tight_layout=True)

####### FIGURE 3 (bottom) ############
    
palette = sns.cubehelix_palette(as_cmap=True)
df = pd.read_csv('batch_alltypes') #read from data file
j_dist = get_coop_coop_neighbour_dist(df) 
j_dist0_compare = compare_sim_type0(j_dist)
width,aspect = PAGEWIDTH/3,0.6
g = sns.relplot(data=j_dist0_compare[j_dist0_compare.k.isin([5,6,7])],x='n',y='j_freq',hue='j',style='type',kind='line',
                col='k',height=width/aspect,aspect=aspect,legend='brief',palette=sns.cubehelix_palette(as_cmap=True))
g.set_ylabels('$f_j^B(n,k)$')
g.savefig('coop_neighbour_dist_B.pdf',tight_layout=True)

plot_colorbar(palette,7,'colorbar_j.pdf')
