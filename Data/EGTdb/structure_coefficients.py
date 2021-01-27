import numpy as np
from scipy.special import binom
import pandas as pd
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt

PAGEWIDTH = 5
Z = 100
PALETTE = sns.color_palette("colorblind")

def structure_coefficient(j,k):
    sc = (k-2)**(k-1-j)/((k+2)*(k+1)*k**2)
    sc *= sum((k-l)*((k**2-(k-2)*l)*nu(l,j,k) + (2*k+(k-2)*l)*tau(l,j,k)) for l in range(0,k))
    return sc

def structure_coefficients(k):
    return np.array([structure_coefficient(j,k) for j in range(k+1)])

def delta_structure_coefficients(k):
    sc = structure_coefficients(k)
    return np.roll(sc,-1)[:-1]-sc[:-1]

def nu(l,j,k):
    return binom((k-1-l),(k-1-j))*1/(k-1)**(k-1-l)+binom(l,k-j)*(k-2)/(k-1)**l

def tau(l,j,k):
    return binom(k-1-l,k-j)*(k-2)/(k-1)**(k-1-l) + binom(l,k-1-j)*1./(k-1)**l
    
def plot_structure_coefficients_range(kvals,ax=None,figsize=None,legend=False):
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    for k in kvals:
        jvals = np.arange(0,k+1)
        ax.scatter(jvals/(k+1),structure_coefficients(k),label="%d"%k,marker='.')
    if legend:    
        plt.legend(title="k",bbox_to_anchor=(1.05,1),loc="upper left",frameon=False)
    ax.set_xlabel("j/(k+1)")
    ax.set_xticks([0,0.5,1])
    ax.set_ylabel(r'$\sigma_j$')
    ax.set_yticks([0,0.1,0.2])
        
def plot_delta_structure_coefficients_range(kvals,ax=None,figsize=None,legend=False):
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    for k in kvals:
        plt.scatter(np.arange(k)/(k+1),delta_structure_coefficients(k),marker='.')
    if legend:
        plt.legend(title="k")
    ax.set_xlabel("j/(k+1)")
    ax.set_xticks([0,1])
    ax.set_ylabel(r"$\Delta\sigma_j$")
    ax.set_yticks([-0.05,0,0.05])
    
def plot_main(kvals):
    fig,axes = plt.subplots(1,2,figsize=(PAGEWIDTH,PAGEWIDTH/1.5))
    plot_structure_coefficients_range(kvals1,axes[0])
    plot_delta_structure_coefficients_range(kvals1,axes[1])
    plt.legend(title='k')
    
####### FIGURE 13 #################################

kvals1 = [5,10,20,40,80]
plot_structure_coefficients_range(kvals1,figsize=(PAGEWIDTH/1.5,PAGEWIDTH/2),legend=True)
plt.tight_layout()
plt.savefig("sigmas_kreg.pdf")



# plot_structure_coefficients_range(kvals2)
    