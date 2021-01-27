import numpy as np
from scipy.special import binom
import pandas as pd
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt

PAGEWIDTH = 5
Z = 100

def structure_coefficient(j,k):
    sc = (k-2)**(k-1-j)/((k+2)*(k+1)*k**2)
    sc *= sum((k-l)*((k**2-(k-2)*l)*nu(l,j,k) + (2*k+(k-2)*l)*tau(l,j,k)) for l in range(0,k))
    return sc
    
def nu(l,j,k):
    return binom((k-1-l),(k-1-j))*1/(k-1)**(k-1-l)+binom(l,k-j)*(k-2)/(k-1)**l

def tau(l,j,k):
    return binom(k-1-l,k-j)*(k-2)/(k-1)**(k-1-l) + binom(l,k-1-j)*1./(k-1)**l
    
def NPD_benefit(j,G):
    return j/G

def VD_benefit(j,G,M):
    return 1*(j>=M)
    
def sigmoid_benefit(j,G,h,s):
    return (logistic_function(j/G,h,s)-logistic_function(0,h,s))/(logistic_function(1,h,s)-logistic_function(0,h,s))

def logistic_function(x,h,s):
    return 1./(1+np.exp(s*(h-x)))

def critical_benefit_to_cost(sigma,k,benefit_function,*params):
    return 1/(sum(sigma_j*(benefit_function(j+1,k+1,*params)-benefit_function(k-j,k+1,*params)) 
                for j,sigma_j in enumerate(sigma)))

def plot_compare(hex_crit,vt_crit,x,col=None,hvals=None,svals=None,col_wrap=2,sints=True,savename=None):
    df = pd.concat([hex_crit.assign(type='HL'),vt_crit.assign(type='VT')])
    if hvals is not None: df = df[df.h.isin(hvals)]
    if svals is not None: df = df[df.s.isin(svals)]
    if sints:
        df['s'] = df['s'].astype(int)
    wm_crit = pd.DataFrame([{'h':h,'s':s,'crit':7*(Z-1)/(Z-7)} for h,s in product(df.h.unique(),df.s.unique())])
    df = pd.concat([wm_crit.assign(type='WM'),df])
    g = sns.relplot(data=df,x=x,y='crit',style='type',col=col,height=PAGEWIDTH/3,aspect=1,kind='line',
                    col_wrap=col_wrap,color='k',style_order=['VT','HL','WM'])
    g.set_ylabels(r'$(b/c)^{*}_1$')
    g._legend.texts[0].set_text('') 
    g._legend.set_bbox_to_anchor([1.0,0.5])
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight')
    return g

###### FIGURE 5 ###################

k = 6
sigma_6 = np.array([structure_coefficient(j,k) for j in range(k+1)])
NPD_crit = critical_benefit_to_cost(sigma_6,6,NPD_benefit)
VD_crit = [critical_benefit_to_cost(sigma_6,6,VD_benefit,M) for M in range(0,8)]
hvals = [i/100. for i in range(101)]
svals = [i/2. for i in range(101)]
sig_crit = pd.DataFrame([{'h':h,'s':s,'crit':critical_benefit_to_cost(sigma_6,6,sigmoid_benefit,h,s)} 
                for h,s in product(hvals,svals)])
sig_crit_vt = pd.read_csv('sigmoid_critical_prediction').rename({'b_to_c_star':'crit'},axis=1)
g = plot_compare(sig_crit,sig_crit_vt,'h','s',svals=[1,5,10,20],savename='compare_btoc.pdf')
