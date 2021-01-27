import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import linregress,sem 
from itertools import product

PAGEWIDTH = 5
PALETTE = sns.color_palette('colorblind')  
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 5



def read_data(dirname,b_vals):
    df = pd.concat([pd.read_csv(dirname+f'/b{b:.2f}',delimiter='    ',names=['fixed','lost','incomplete']).assign(b=b) 
                        for b in b_vals])
    df = df.reset_index(drop=True)
    if df.incomplete.nunique()==1:
        df = df.drop('incomplete',axis=1) 
    return df.reindex(columns=['b','fixed','lost'])  
    
def get_sig_fixprobs(dirname,b_vals_list,sigfunc_params,percent=True):
    df = pd.concat([read_data(dirname+f'/h{h:.3f}_s{s:.1f}',b_vals).assign(h=h,s=s) for (h,s),b_vals in zip(sigfunc_params,b_vals_list)])
    df = df.assign(fixprob=df.fixed/(df.fixed+df.lost))
    if percent:
        df.fixprob*=100
    return pd.DataFrame(df,columns=['h','s','b','fixprob'])

def get_VD_fixprobs(dirname,b_vals_list,threshold_vals,percent=True):
    df = pd.concat([read_data(dirname+f'/threshold_{M:d}',b_vals).assign(M=M) for M,b_vals in zip(threshold_vals,b_vals_list)])
    df = df.assign(fixprob=df.fixed/(df.fixed+df.lost))
    if percent:
        df.fixprob*=100
    return pd.DataFrame(df,columns=['M','b','fixprob'])

def get_NPD_fixprobs(dirname,b_vals,percent=True):
    df = read_data(dirname,b_vals)
    df = df.assign(fixprob=df.fixed/(df.fixed+df.lost))
    if percent:
        df.fixprob*=100
    return pd.DataFrame(df,columns=['b','fixprob'])

def get_mean_fixprobs(df,groupby_params='b',funcs=['sem']):
    if funcs is not None:
        funcs = ['mean'] + funcs
        df = df.groupby(groupby_params).agg(funcs)
        df.columns = df.columns.map('_'.join)
    else: 
        df = df.groupby(groupby_params).mean()
    return df.reset_index()

def plot_fixprobs(df,hue=None,col=None,row=None,savename=None):
    g = sns.lmplot(data=df,x='b',y='fixprob',x_estimator=np.mean,x_ci='sd',hue=hue,col=col,row=row,height=PAGEWIDTH/2,aspect=4/3,ci=None)
    axes = g.axes.flatten()
    for ax in axes:    
        ax.plot(df.b.unique(),df.b.nunique()*[1.0],ls=':',color='grey')
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight')

def get_sig_fixprobs_compare(dirname,b_vals_list,sigfunc_params):
    df_sig_compare = pd.concat([get_sig_fixprobs(dirname+'/mutantC',b_vals_list,sigfunc_params,percent=True).assign(type='C'),
                            get_sig_fixprobs(dirname+'/mutantD',b_vals_list,sigfunc_params,percent=True).assign(type='D')])
    df_sig_compare.loc[df_sig_compare.type=='D','fixprob'] = 100 - df_sig_compare[df_sig_compare.type=='D']['fixprob']
    return df_sig_compare

def calc_critical_b_to_c(df,M=None,h=None,s=None):
    if M is not None:
        df = df[df.M==M].drop('M',axis=1) 
    if h is not None and s is not None:
        df = df[(df.h==h)&(df.s==s)].drop(['s','h'],axis=1) 
    slope, intercept, r_value, p_value, std_err = linregress(get_mean_fixprobs(df,'b',None))
    return (1-intercept)/slope

def calc_critical_intercept(df,M=None,h=None,s=None):
    if M is not None:
        df = df[df.M==M].drop('M',axis=1) 
    if h is not None and s is not None:
        df = df[(df.h==h)&(df.s==s)].drop(['s','h'],axis=1)
    slopeC, interceptC, r_valueC, p_valueC, std_errC = linregress(get_mean_fixprobs(df[df.type=='C'],'b',None))
    slopeD, interceptD, r_valueD, p_valueD, std_errD = linregress(get_mean_fixprobs(df[df.type=='D'],'b',None))
    return (interceptD-interceptC)/(slopeC-slopeD)    

def plot_sig_critical_intercepts(df,x,hue=None,col=None,theory_file=None,savename=None,height=PAGEWIDTH/2,aspect=2/3,color=PALETTE[1]):
    critical_label = r'$(b/c)^*_1$'
    criticals = [{'h':hval,'s':sval,critical_label:calc_critical_intercept(df,h=hval,s=sval)}
                    for (hval,sval) in df.set_index(['h','s']).index.unique()]
    criticals = pd.DataFrame(criticals)
    _plot_sigmoid_criticals(criticals,x,critical_label,hue=hue,col=col,theory_file=theory_file,savename=savename,height=height,aspect=aspect,color=color)
    

def plot_criticals_sigmoid(df,x,hue=None,col=None,theory_file=None,savename=None,height=PAGEWIDTH/2,aspect=2/3,color=PALETTE[1]):
    critical_label = r'$(b/c)^*_0$'
    criticals = [{'h':hval,'s':sval,critical_label:calc_critical_b_to_c(df,h=hval,s=sval)}
                    for (hval,sval) in df.set_index(['h','s']).index.unique()]
    criticals = pd.DataFrame(criticals)
    _plot_sigmoid_criticals(criticals,x,critical_label,hue=hue,col=col,theory_file=theory_file,savename=savename,height=height,aspect=aspect,color=color)

def _plot_sigmoid_criticals(criticals,x,critical_label,hue=None,col=None,theory_file=None,savename=None,height=PAGEWIDTH/2,aspect=2/3,color=PALETTE[1]):
    if hue is not None:    
        palette = set_palette(len(df[hue].unique())) 
        legend = 'full'
        g = sns.relplot(data=criticals,x=x,y=critical_label,hue=hue,col=col,palette=palette,height=height,aspect=aspect,legend=legend,)
    else:
        legend = False  
        g = sns.relplot(data=criticals,x=x,y=critical_label,col=col,edgecolor=color,facecolor='white',height=height,aspect=aspect,legend=legend,marker='o')
    g.set(xticks=[0,1])
    if theory_file is not None:
        criticals_theory = pd.read_csv(theory_file)
        criticals_theory = pd.DataFrame(criticals_theory[criticals_theory.s.isin(criticals.s.values)])  
        criticals_theory = criticals_theory.rename({'b_to_c_star':critical_label},axis=1)
        criticals_theory = criticals_theory.set_index('s')
        for s,ax in zip(criticals.s.unique(),g.axes.flatten()):
            ax.plot(criticals_theory.loc[s,'h'],criticals_theory.loc[s,critical_label],color=color,zorder=0)
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight')

def set_palette(ncol):
     if ncol > len(PALETTE):
         return sns.color_palette("hls",ncol)
     else:
         return PALETTE[:ncol]
  
#
# df_PD = get_NPD_fixprobs('PD_fixprobs',np.arange(2,3.75,0.25))
# plot_fixprobs(df_PD,savename='PD_fixprobs.pdf')
# print('critical PD: ', calc_critical_b_to_c(df_PD))
#
# df_NPD = get_NPD_fixprobs('NPD_fixprobs',np.arange(1.5,3.25,0.25))
# plot_fixprobs(df_NPD,savename='NPD_fixprobs.pdf')
# print('critical NPD: ', calc_critical_b_to_c(df_NPD))
#
# df_VD = get_VD_fixprobs('VD_fixprobs',[np.arange(1.5,3,0.25),[1.25,1.5,1.75,2.5]],[2,3])
# plot_fixprobs(df_VD,savename='VD_fixprobs.pdf',hue='M')
# for M in df_VD.M.unique():
#     print(f'critical VD, M = {M}: ', calc_critical_b_to_c(df_VD,M=M))
    
if __name__ == "__main__":
    
    sigfunc_params = [[0,1],[0.5,1],[1,1],
                        [0,5],[0.2,5],[0.5,5],[0.8,5],[1,5],
                        [0,10],[0.2,10],[0.5,10],[0.8,10],[1,10]]
    b_vals_list = [np.arange(1.5,3.2,0.25)]*3 \
                    +[np.arange(1.4,2.3,0.2)]+[np.arange(1.6,2.5,0.2)]+[np.arange(1.4,2.5,0.2)]+[np.arange(2.0,3.1,0.2)]+[np.arange(2.6,3.5,0.2)]\
                    +[np.arange(1.6,2.5,0.2)]+[np.arange(1.6,2.5,0.2)]+[np.arange(1.4,2.3,0.2)]+[np.arange(2.8,4.5,0.4)]+[np.arange(4.2,5.3,0.2)]


    df_sig = get_sig_fixprobs('SIG_fixprobs/mutantC',b_vals_list,sigfunc_params,percent=True)
    plot_fixprobs(df_sig,savename='sig_fixprobs.pdf',hue='s',col='h')
    for h,s in sigfunc_params:
        print(f'critical sigmoid, h = {h:.1f}, s = {s:.0f}: ', calc_critical_b_to_c(df_sig,h=h,s=s))
    
    
    #FIGURE 8
    plot_criticals_sigmoid(df_sig,x='h',col='s',theory_file='sigmoid_critical_prediction0',savename='sigmoid_criticals0.pdf',height=PAGEWIDTH/2.5,aspect=2.5/3,color='k')



    sigfunc_params = [[h,s] for s in [5,10] for h in [0,0.2,0.5,0.8,1.0]]
    b_vals_list = np.vstack([np.arange(2.0,2.9,0.2),np.arange(1.8,2.7,0.2),np.arange(1.8,2.7,0.2),np.arange(1.8,2.7,0.2),np.arange(2.0,2.9,0.2),
                                np.arange(2.4,3.3,0.2),np.arange(2.0,2.9,0.2),np.arange(1.6,2.5,0.2),np.arange(2.0,2.9,0.2),np.arange(2.4,3.3,0.2)
                            ])
    df_sig_compare = get_sig_fixprobs_compare('SIG_fixprobs',b_vals_list,sigfunc_params)
    
    #FIGURE 7
    plot_fixprobs(df_sig_compare,hue='type',col='h',row='s',savename=None)
    plot_sig_critical_intercepts(df_sig_compare,'h',hue=None,col='s',color='k',theory_file='sigmoid_critical_prediction1',savename='sigmoid_criticals1.pdf',height=PAGEWIDTH/2.5,aspect=2.5/3)