import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product
import coop_neighbour_stats as cns

PAGEWIDTH = 5
DELTA = 0.025
Z=100

def plot_degree_dist_with_sd(df,k_min=None,k_max=None,savename=None):
    """plot degree distribution for given neighbour data (df) with standard deviation"""
    if k_min is not None: df = df[df.k>=k_min]
    if k_max is not None: df = df[df.k<=k_max]
    degree_dist = df.groupby(['time','tissueid'])['k'].value_counts(normalize=True)
    k_vals = sorted(df.k.unique())
    index = pd.MultiIndex.from_tuples([(t,idx,k) for t,idx,_k in degree_dist.index for k in k_vals],names=('time','tissueid','k'))
    degree_dist = degree_dist.reindex(index).fillna(0)   
    degree_dist = degree_dist.reset_index(name='Frequency')
    g = sns.catplot(x='k',y='Frequency',data=degree_dist,ci='sd',kind='bar',height=PAGEWIDTH/2,aspect=1.2)
    g.set_xlabels(r'Number of neighbours, $k$')
    g.set_ylabels(r'Frequency, $g_k$')
    if savename is not None:
        plt.savefig(savename+'.pdf')

def plot_degree_distribution(df):
    """plot degree distribution for given neighbour data (df)"""
    g = sns.distplot(df.k,kde=False,axlabel='degree',norm_hist=True,bins=[4,5,6,7,8,9,10],hist_kws={'rwidth':0.9,'alpha':0.9})
    return g
    
def plot_compare_degree_dist(df,k_min,k_max,savename=None,width=PAGEWIDTH/1.4,aspect=1.2,nvals=None,col_wrap=None):
    if nvals is not None:
        col = 'n'
        df = df.loc[df.n.isin(nvals)]
    else:
        col = None
    df = df.loc[df.k.isin([4,5,6,7,8])]
    replace_dict = {0:'D',1:'C'}
    df = df.replace({'type':replace_dict})
    g = sns.displot(data=df,x='k',kind='hist',stat='probability',common_norm=False,bins=[4,5,6,7,8],shrink=0.8,alpha=0.9,hue='type',discrete=True,hue_order=['C','D'],
            multiple='dodge',palette=sns.color_palette('colorblind')[:2],height=width/aspect,aspect=aspect,edgecolor='white',col=col,col_wrap=col_wrap)
    g.legend.set_title(None)
    g.set_xlabels(r'$k$')
    g.set_ylabels(r'$g_k$')
    g.set(xticks=[4,6,8])
    g.set_xticklabels(['4','6','8'])
    if savename is not None:
        plt.savefig(savename+'.pdf')
    return g
    
def plot_coop_coop_neighbour_dist_with_n(df,k_min=0,k_max=np.inf,col_wrap=3,savename=None,width=PAGEWIDTH/3,aspect=1,legend=False):
    """plot distribution (f^A_j(n,k)) of number coop neighbours (j) for coop cells given k total neighbours and n coop cells in population.
        calculated from neighbour data in df"""
    j_dist = cns.get_coop_coop_neighbour_dist(df)
    j_dist = j_dist[(j_dist.k>=k_min)&(j_dist.k<=k_max)]
    g = sns.relplot(x='n',y='j_freq',hue='j',data=j_dist,kind='line',col='k',col_wrap=col_wrap,height=width/aspect,aspect=aspect,legend=legend)
    g.set_ylabels('$f_j^A(n,k)$')
    if savename is not None:
        plt.savefig(savename+'.pdf')
    return g

# def plot_coop_coop_neighbour_dist_given_n(df,n,k_min=0,k_max=None,norm=True):
#     if k_max is None:
#         k_max = df.k.max()
#     df = df[(df.k>=k_min)&(df.k<=k_max)]
#     try:
#         g = sns.FacetGrid(data=df[df.n.isin(n)],col='k',row='n')
#         g = g.map(sns.distplot,'j',kde=False,hist=True,bins=np.arange(k_max+2),norm_hist=norm)
#     except TypeError:
#         g = sns.FacetGrid(data=df[df.n==n],col='k',col_wrap=3)
#         g = g.map(sns.distplot,'j',kde=False,bins=np.arange(k_max+2),norm_hist=norm)
#     return g

def plot_compare_sigma(sigma_vt,sigma_wm,k_min=None,k_max=None,col_wrap=2,savename=None,width=PAGEWIDTH/3,aspect=1,palette=None):
    """plot structure coefficients for well-mixed and VT model"""
    sigma_vt = sigma_vt.assign(type='VT')
    sigma_wm = sigma_wm.assign(type='WM')
    sigma = pd.concat([sigma_vt,sigma_wm])
    if k_min is not None: sigma = sigma[sigma.k>=k_min]
    if k_max is not None: sigma = sigma[sigma.k<=k_max]
    g = sns.relplot(data=sigma,x='j',y='sigma',style='type',hue='type',col='k',markers=['d','o'],
                        height=width/aspect,aspect=aspect,col_wrap=col_wrap,edgecolor='k',palette=['k','w'],legend=False)
    g.set_ylabels(r'$\sigma_{j,k}$')
    # g._legend.texts[0].set_text('')
    handles = [mpl.lines.Line2D([0],[0],marker='d',markeredgecolor='k',markerfacecolor='k',color='w',markersize=5),
                                mpl.lines.Line2D([0],[0],marker='o',markeredgecolor='k',markerfacecolor='w',color='w',markersize=5)]
    labels = ['VT','WM']
    g.add_legend(handles=handles,labels=labels,loc='center right',title='')
    if savename is not None:
        g.savefig(savename+'.pdf',tight_layout=True)
    return g


if __name__ == '__main__':    
    # df = cns.load_data('batch_tm')
    df_all = cns.load_data('batch_alltypes',None)
    Z=100
    
    #FIGURE 2
    g = plot_degree_dist_with_sd(df,4,8,'degree_dist')
    
    #FIGURE 12
    g = plot_compare_degree_dist(df_all,4,8,'degree_dist_compare',nvals=[1,10,20,40,60,80,90,99],col_wrap=4,width=PAGEWIDTH/4,aspect=0.7)

    # #FIGURE 3 (top)
    g = plot_coop_coop_neighbour_dist_with_n(df,5,7,None,'coop_neighbour_dist',aspect=0.6,legend=False)
    
    # #FIGURE 4
    sigma_vt = cns.structure_coeffs(df)
    sigma_wm = cns.structure_coeffs_well_mixed(Z,cns.get_degree_distribution(df))
    g = plot_compare_sigma(sigma_vt,sigma_wm,5,7,None,'sigma',aspect=0.8)
