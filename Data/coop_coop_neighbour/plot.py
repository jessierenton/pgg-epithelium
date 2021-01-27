import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import coop_neighbour_stats as cns

READFILE = 'batch_tm'
PAGEWIDTH = 5
DELTA = 0.025
PALETTE = sns.color_palette('colorblind')

Z=100

def plot_sigmoid_critical_heatmap(data,hvals,svals,Z,ax,savename=None,btoc_type=1,show_yaxis=True,**heatmap_kws):  
    """takes data as sigmoid_df for btoc_type=1 or tuple (theta_A,theta_B) for btoc_type=0""" 
    if btoc_type==1:
        sigmoid_critical = [[cns.critical_benefit_to_cost1(data,Z,cns.sigmoid_benefit,h,s) 
                            for h in hvals] for s in svals]
    elif btoc_type == 0:
        sigmoid_critical = [[cns.critical_benefit_to_cost0(data[0],data[1],Z,cns.sigmoid_benefit,h,s) 
                            for h in hvals] for s in svals]
    sigmoid_critical = np.array(sigmoid_critical)
    sns.heatmap(sigmoid_critical,ax=ax,xticklabels=False,yticklabels=False,rasterized=True,**heatmap_kws)
    ax.invert_yaxis()
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel('h')
    ax.set_xticks([0,int(len(hvals)/2),len(hvals)])
    ax.set_xticklabels([0,0.5,1])
    if show_yaxis:
        ax.set_yticks([0,int(len(svals)/2),len(svals)])
        ax.set_ylabel('s')
        ax.set_yticklabels([svals[0],int(svals[len(svals)//2]),int(svals[-1])])
    if savename is not None:
        plt.savefig(savename+'.pdf',bbox_inches='tight')
    return ax

def plot_heatmaps(sigma_vt,theta_A,theta_B,hvals,svals):
    fig,axes = plt.subplots(ncols=3,figsize=[PAGEWIDTH,PAGEWIDTH/2.2],gridspec_kw={'width_ratios':[12,12,1]})
    vmin,vmax = 1.5,6.5
    norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
    heatmap_kws = {'norm':norm,'vmin':vmin,'vmax':vmax,'cmap':'mako_r'}
    plot_sigmoid_critical_heatmap((theta_A,theta_B),hvals,svals,Z,ax=axes[0],btoc_type=0,cbar=False,**heatmap_kws)
    plot_sigmoid_critical_heatmap(sigma_vt,hvals,svals,Z,ax=axes[1],btoc_type=1,cbar=True,cbar_ax=axes[2],show_yaxis=False,**heatmap_kws)
    axes[1].collections[0].colorbar.set_ticks([2,3,4,5,6])  
    axes[1].collections[0].colorbar.set_ticklabels([2,3,4,5,6]) 
    axes[0].set_title(r'$(b/c)^*_0$',pad=10) 
    axes[1].set_title(r'$(b/c)^*_1$',pad=10) 
    plt.savefig('criticals_heatmap.pdf',bbox_inches='tight')

def fill_between(ax,sigmoid_btoc,s,colors,alpha):
    df0 = sigmoid_btoc[sigmoid_btoc.s==s]
    btoc0,btoc1 = df0[df0.type==0]['b_to_c_star'].values,df0[df0.type==1]['b_to_c_star'].values
    hvals = df0[df0.type==0]['h'].values
    n = len(btoc0)//2+1
    hvals0,hvals1 = hvals[:n],hvals[n-1:]
    btoc00,btoc01,btoc10,btoc11 = btoc0[:n],btoc0[n-1:],btoc1[:n],btoc1[n-1:] 
    ax.fill_between(hvals,np.ceil(sigmoid_btoc.b_to_c_star.max()),np.append(btoc10,btoc01[1:]),facecolor=colors[0],edgecolor=None,alpha=alpha)
    ax.fill_between(hvals0,btoc10,btoc00,facecolor=colors[1],edgecolor=None,alpha=alpha)
    ax.fill_between(hvals1,btoc01,btoc11,facecolor=colors[2],edgecolor=None,alpha=alpha)
    ax.fill_between(hvals,np.append(btoc00,btoc11[1:]),np.floor(sigmoid_btoc.b_to_c_star.min()*2)/2  ,facecolor=colors[3],edgecolor=None,alpha=alpha)
    return ax


def plot_sigmoid_critical(sigmoid_df,x,hue=None,col=None,hvals=None,svals=None,savename=None,btoc_type=1,**relplot_kws):
    if hvals is not None: sigmoid_df = sigmoid_df[sigmoid_df.h.isin(hvals)]
    if svals is not None: sigmoid_df = sigmoid_df[sigmoid_df.s.isin(svals)]
    # if hue == 's':
    #     norm = mpl.colors.LogNorm(vmin=sigmoid_df.s.min(), vmax=sigmoid_df.s.max())
    # elif hue == 'h':
    #     norm = mpl.colors.Normalize(vmin=sigmoid_df.h.min(),vmax=sigmoid_df.h.max())
    # else:
    #     norm = None
    g = sns.relplot(data=sigmoid_df,x=x,y='b_to_c_star',hue=hue,col=col,kind='line',**relplot_kws)
    ylabel = r'$b/c$' if btoc_type is None or col=='type' else r'$(b/c)^*_{}$'.format(btoc_type) 
    g.set_ylabels(ylabel)
    if col=='type':
        g.axes[0].set_title(r'$(b/c)^*_0$')
        g.axes[1].set_title(r'$(b/c)^*_1$')
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight')
    return g
    
def plot_compare(sigmoid_btoc,colors,alpha,savename=None,width=PAGEWIDTH/3,aspect=1,title=None):
    g = plot_sigmoid_critical(sigmoid_btoc,'h',col='s',hvals=None,svals=None,savename=None,legend=False,style='type',
                                    col_wrap=2,color='k',height=width/aspect,aspect=aspect)
    g.set_ylabels(r'$b/c$')
    yintmin,yintmax = np.ceil(sigmoid_btoc.b_to_c_star.min()/2)*2,np.round(sigmoid_btoc.b_to_c_star.max()/2)*2
    g.set(yticks=[yintmin,(yintmax+yintmin)/2,yintmax])  
    for s,ax in zip(sigmoid_btoc.s.values,g.axes):
        fill_between(ax,sigmoid_btoc,s,colors,alpha)
    if title is not None:
        g.axes.flatten()[0].text(0.02,0.95,title,weight='bold',transform=g.fig.transFigure)
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight')
    return g
   
def plot_criticals_vs_s(sigma_df,theta_A,theta_B,hvals,svals):
    criticals = pd.concat([
                        cns.sigmoid_critical_df0(theta_A,theta_B,hvals,svals).assign(type=0),
                        cns.sigmoid_critical_df1(sigma_df,hvals,svals).assign(type=1)])
    g = sns.relplot(data=criticals,x='s',y='b_to_c_star',style='type',col='h',hue='h',kind='line',height=PAGEWIDTH/3,aspect=3/5,
                        color='k')
    g.set_ylabels(r'$b/c$')

def plot_criticals0_compare(theta_A1,theta_B1,theta_A2,theta_B2,hvals,svals,label1,label2,savename=None,width=PAGEWIDTH/3,aspect=1):
    criticals = pd.concat([cns.sigmoid_critical_df0(theta_A1,theta_B1,hvals,svals).assign(type=label1),
                            cns.sigmoid_critical_df0(theta_A2,theta_B2,hvals,svals).assign(type=label2)])
    g = sns.relplot(data=criticals,x='h',y='b_to_c_star',style='type',col='s',height=width/aspect,aspect=aspect,kind='line',
                    col_wrap=3,color='k',legend='full')
    g.set_ylabels(r'$(b/c)^{*}_0$')
    g._legend.texts[0].set_text('') 
    g._legend.set_bbox_to_anchor([1.0,0.5])
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight')                        
                            
def plot_gos(gos,palette,width=PAGEWIDTH/3,aspect=1,savename=None,title=None,subplot_kwargs=None):
    g = sns.relplot(data=gos,x='n',y='gos',hue='b',col='h',row='s',kind='line',
                        height=width/aspect,aspect=aspect,facet_kws={'margin_titles':True},legend=False)
    g.set_ylabels(r'$G(n)$')
    g.set(yticks=[0,0.005])
    if title is not None:
        g.axes.flatten()[0] .text(0.03,0.95,title,weight='bold',transform=g.fig.transFigure,ha='left',va='top')
    if savename is not None:
        if subplot_kwargs is None:
            plt.savefig(savename,bbox_inches='tight')
        else:
            plt.subplots_adjust(**subplot_kwargs)
            plt.savefig(savename)
    return g
    

def plot_colorbar(cmap,bmin,bmax,savename,bminlabel=None,orientation='horizontal'):
    fig,ax = plt.subplots(figsize=(0.8,0.4)) if orientation == 'horizontal' else plt.subplots(figsize=(0.45,0.8))
        
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(bmin,bmax),cmap=cmap),
                    cax=ax,orientation=orientation)
    cbar.set_ticks([bmin,bmax])
    plt.tick_params(axis = "both", which = "both", bottom = False, top = False)
    if orientation == 'horizontal':
        cbar.set_ticks([bmin,bmax])
        cbar.set_ticklabels([bminlabel,bmax])
        plt.subplots_adjust(bottom=0.6,left=0.27,right=0.86)
    else:
        cbar.set_ticks([])
        ax.set_title(bmax,fontsize='medium')
        ax.set_xlabel(bminlabel,fontsize='medium')
        plt.subplots_adjust(bottom=0.22,left=0.33,right=0.64,top=0.58)
    plt.savefig(savename,tight_layout=True)

def find_fp(df): 
    fp = [] 
    for n,g in zip(df.n,df.gos): 
        if n==0: 
            fp.append([0,-np.sign(df.iloc[1].gos)]) 
        elif n==100: 
            fp.append([100,np.sign(df.iloc[-2].gos)])     
        elif g0*g<0: 
            if np.abs(g0)<np.abs(g): 
                fp.append([n0,np.sign(g0)]) 
            else: 
                fp.append([n,np.sign(g0)]) 
        n0,g0=n,g 
    return fp 

def find_fp_vary_h(df,s,b,color='k',ax=None,figsize=(PAGEWIDTH/5,PAGEWIDTH/5),linestyles=['--','-']):
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    df = df[(df.s==s)&(df.b==b)].sort_values(['h','n'])
    fixed_points = [find_fp(df[df.h==h]) for h in df.h.unique()]
    nfp = [len(f) for f in fixed_points]
    bifs=[0]+[i+1 for i,(n0,n1) in enumerate(zip(nfp[:-1],nfp[1:])) if n0!=n1]+[None]
    hvals = [df.h.unique()[i:j] for i,j in zip(bifs[:-1],bifs[1:])]
    fixed_points = [np.array(fixed_points[i:j],dtype=float) for i,j in zip(bifs[:-1],bifs[1:])]
    for harr,fp in zip(hvals,fixed_points):
        fparr = fp.T[0]
        stability = np.array((fp.T[1,:,0]+1)/2,dtype=int)
        for stab,fpvals in zip(stability,fparr):
            ax.plot(harr,fpvals,color=color,ls=linestyles[stab])
    return ax
        
if __name__ == "__main__":    
    df = pd.read_csv(READFILE)
    sigma_vt = cns.structure_coeffs(df)
    theta_A,theta_B = cns.theta_AB(df)
    theta_wm_A,theta_wm_B = cns.theta_AB_wm(Z,degree_dist=None)
    sigma_wm = cns.structure_coeffs_well_mixed(Z,degree_dist=None)

    ########### FIGURE 6 ########################
    hvals = [i/100. for i in range(101)]
    svals = [i/2. for i in range(1,101)]

    plot_heatmaps(sigma_vt,theta_A,theta_B,hvals,svals)

    ########### FIGURE 10 ########################
    hvals = [i/100. for i in range(101)]
    svals = [1,5,10,20]
    svals = [20,50,100,500]

    sigmoid_btoc_vt = pd.concat([cns.sigmoid_critical_df0(theta_A,theta_B,hvals,svals).assign(type=0),
                                cns.sigmoid_critical_df1(sigma_vt,hvals,svals).assign(type=1)])
    sigmoid_btoc_wm = pd.concat([cns.sigmoid_critical_df0(theta_wm_A,theta_wm_B,hvals,svals).assign(type=0),
                                cns.sigmoid_critical_df1(sigma_wm,hvals,svals).assign(type=1)])

    colors = (PALETTE[0],PALETTE[2],PALETTE[4],PALETTE[3])
    g = plot_compare(sigmoid_btoc_vt,colors,alpha=0.3,savename=None,title='VT')
    g = plot_compare(sigmoid_btoc_wm,colors,alpha=0.3,savename=None,title='WM')

    # ########### FIGURE ... ########################
    # hvals = [0.,0.2,0.5,0.8,1.0]
    # svals = [i/10 for i in range(1,500)]
    # plot_criticals_vs_s(sigma_vt,theta_A,theta_B,hvals,svals)

    ######### FIGURE 11 #########################
    # Gradient of selection
    hvals = [0.0,0.2,0.5,0.8,1.0]
    svals = [1,5,10]
    bvals_wm = [3,5,7,9,11]
    bvals_vt = [1.5,2,2.5,3,3.5]
    palette=sns.cubehelix_palette(as_cmap=True)
    gos_wm=cns.sigmoid_gos_wm(Z,bvals_wm,hvals,svals).reset_index()
    gos_vt=cns.sigmoid_gos(df,bvals_vt,hvals,svals).reset_index()
    subplot_kwargs = {'left':0.13,'bottom':0.14,'right':0.95,'top':0.92,'wspace':0.37,'hspace':0.21}
    g = plot_gos(gos_vt,palette,width=PAGEWIDTH/4,aspect=0.9,savename=None,title='VT',subplot_kwargs=subplot_kwargs)
    g = plot_gos(gos_wm,palette,width=PAGEWIDTH/4,aspect=0.9,savename=None,title='WM',subplot_kwargs=subplot_kwargs)
    plot_colorbar(palette,1.5,3.5,'gos_vt_cb.pdf','b=1.5')
    plot_colorbar(palette,3,11,'gos_wm_cb.pdf','b=3')
    plot_colorbar(palette,1.5,3.5,'gos_vt_cbv.pdf','b=1.5',orientation='vertical')
    plot_colorbar(palette,3,11,'gos_wm_cbv.pdf','b=3',orientation='vertical')

    ######### FIGURE 9 #########################
    svals = [1,5,10]
    hvals = [i/100. for i in range(101)]
    plot_criticals0_compare(theta_A,theta_B,theta_wm_A,theta_wm_B,hvals,svals,'VT','WM',savename='compare_criticals_0.pdf',aspect=0.8)


