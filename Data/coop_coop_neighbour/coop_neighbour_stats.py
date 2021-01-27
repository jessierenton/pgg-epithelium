import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product
from scipy.special import binom

READFILE = 'batch_tm' #data file 
DELTA = 0.025
Z=100

def load_data(READFILE,load_type=1):
    """load neighbour stats data from file into dataframe.
    if file has type column either: if load_type is None, load all data; or if load_type = 0 or 1, load given type"""
    df = pd.read_csv(READFILE)
    if 'type' in df.columns and load_type is not None:
        df = pd.DataFrame(df[df.type==1])
        df = df.drop('type',axis=1)
    return df

def get_x_dist(df,coop=True):
    if coop: df['x']=(df.j+1)/(df.k+1)
    else: df['x']=df.j/(df.k+1)
    x_dist = df.groupby(['n'])['x'].value_counts(normalize=True).sort_index() 
    return x_dist.reset_index(name='freq')

def get_degree_distribution(df):
    """return probability distribution for number of neighbours"""
    degree_dist = df.k.value_counts(normalize=True).sort_index()
    return degree_dist
    
def get_coop_coop_neighbour_dist(df):
    """return probability distribution for number of coop neighbours for a coop cell as dataframe.
            depends on k (# neighbours) and n (total coop cells in population)"""
    j_dist = df.groupby(['n','k'])['j'].value_counts(normalize=True).sort_index()
    return j_dist.reset_index(name='j_freq')

def hypergeometric(Z,n,k,j):
    return binom(Z-1,k)**-1*binom(n-1,j)*binom(Z-n,k-j)

def get_coop_coop_neighbour_dist_well_mixed(Z,kvals):
    """return probability distribution for number of coop neighbours for a coop cell as dataframe for well-mixed population
    of size Z, for given range of neighbour numbers (kvals)"""
    return pd.DataFrame([{'n':n,'k':k,'j':j,'j_freq':hypergeometric(Z,n,k,j)} for n in range(1,Z) for k in kvals
                                for j in range(k+1)])
    

def get_f_jk(j_dist,degree_dist=None):
    """return probabilities that a coop cell has j coop neighbours AND k neighbours total"""
    if degree_dist is not None:
        for k,freq in zip(degree_dist.index, degree_dist.values):
            j_dist.loc[j_dist.k==k,'j_freq'] *= freq
    return j_dist

def get_f_jkAB(j_dist,degree_dist):
    """return probabilities that a coop (A) and defector (B) cell has j coop neighbours AND k neighbours total"""
    f_jk = get_f_jk(j_dist,degree_dist)
    f_jk = f_jk.rename({'j_freq':'A'},axis=1)
    f_jk_temp = f_jk.copy()
    f_jk_temp['j'] = f_jk['k']- f_jk['j']
    f_jk_temp['n'] = Z - f_jk['n']
    f_jk = f_jk.set_index(['n','k','j'])
    f_jk_temp = f_jk_temp.set_index(['n','k','j'])
    f_jk_temp = f_jk_temp.rename({'A':'B'},axis=1)
    return pd.concat([f_jk,f_jk_temp],axis=1).fillna(0).reset_index()


def sample_tissues(df,size):
    """generate a sample of unique tissues from the dataframe of given size"""
    tissues_df = pd.DataFrame(df,columns=['tissueid','time','n']).drop_duplicates()
    sample = tissues_df.groupby('n').apply(lambda x: x.sample(size))  
    index = pd.MultiIndex.from_frame(sample) 
    df = df.assign(i=[i for n in df.n for i in range(n)]) 
    df.set_index(['tissueid','time','i'])[index]   

def calc_lambda_cc(df):
    """calculate lambda_CC = expected proportion coop neighbours for a coop"""
    degree_dist =  get_degree_distribution(df)
    j_dist = get_coop_coop_neighbour_dist(df)
    for k,freq in zip(degree_dist.index, degree_dist.values):
        j_dist.loc[j_dist.k==k,'j_freq'] *= freq
    grouped = j_dist.groupby('n')
    return grouped.apply(lambda x: sum(x.j/x.k*x.j_freq))
    
     
def structure_coeffs(df):
    """compute structure coefficients from the neighbour data in df"""
    j_dist = get_coop_coop_neighbour_dist(df) 
    # j_dist = j_dist.append([{'n':1,'k':k,'j':0, 'j_freq':1} for k in range(j_dist.k.min(),j_dist.k.max())])
    sigma_df = j_dist.groupby(['k','j'])['j_freq'].sum().reset_index(name='sigma')
    degree_dist = get_degree_distribution(df) 
    for k,freq in zip(degree_dist.index, degree_dist.values):
        sigma_df.loc[sigma_df.k==k,'sigma'] *= freq
    return sigma_df

def structure_coeffs_well_mixed(Z,degree_dist=None):
    """compute structure coefficients for well_mixed population. If no degree_dist assume k=6"""
    if degree_dist is None: degree_dist = pd.Series([1],index=[6]) 
    df = pd.DataFrame([{'k':k, 'j':j, 'sigma':degree_dist.loc[k]*sigma_j_wm(Z,k,j)} 
                for k in degree_dist.index 
                    for j in range(k+1)] )   
    return df

def sigma_j_wm(Z,k,j):
    """return j-th structure coefficient for a well-mixed population with group size k+1"""
    if j < k:
        return Z/(k+1)
    else:
        return (Z-k-1)/(k+1)
    

def gradient_of_selection(f_jk,benefit_function,b,c=1,*params):
    """compute gradient of selection from coop/defector neighbour distribution for coop neighbours"""
    f_jk['gos'] = f_jk.A*(b*benefit_function(f_jk.j+1,f_jk.k+1,*params)-c) - f_jk.B*b*benefit_function(f_jk.j,f_jk.k+1,*params)
    gos = f_jk.groupby('n')['gos'].sum()  
    gos = gos*(Z-gos.index.values)*gos.index.values/(Z**2)*DELTA
    gos.loc[0]=gos.loc[100]=0
    return gos

def NPD_gos(df,bvals,c=1):
    """returns gradient of selection for an NPD game calculated from neighbour data in df"""
    j_dist = get_coop_coop_neighbour_dist(df) 
    degree_dist = get_degree_distribution(df)
    f_jk = get_f_jkAB(j_dist,degree_dist)
    return pd.concat([gradient_of_selection(f_jk,NPD_benefit,b,c) for b in bvals],keys=bvals,names='b')
    
def sigmoid_gos(df,bvals,hvals,svals,c=1):
    """returns gradient of selection for a sigmoid game calculated from neighbour data in df"""
    j_dist = get_coop_coop_neighbour_dist(df) 
    degree_dist = get_degree_distribution(df)
    f_jk = get_f_jkAB(j_dist,degree_dist)
    return pd.concat([gradient_of_selection(f_jk,sigmoid_benefit,b,c,h,s) 
                for b,h,s in product(bvals,hvals,svals)],keys=list(product(bvals,hvals,svals)),names=['b','h','s'])

def sigmoid_gos_wm(Z,bvals,hvals,svals,c=1,degree_dist=None):
    """returns gradient of selection for a sigmoid game calculated for a well-mixed population with given
        degree_dist (if None assume 6 neighbours)"""
    kvals = degree_dist.index if degree_dist is not None else [6]
    j_dist = get_coop_coop_neighbour_dist_well_mixed(Z,kvals)
    f_jk = get_f_jkAB(j_dist,degree_dist)
    return pd.concat([gradient_of_selection(f_jk,sigmoid_benefit,b,c,h,s) 
                for b,h,s in product(bvals,hvals,svals)],keys=list(product(bvals,hvals,svals)),names=['b','h','s'])

def get_theta_AB(j_dist,degree_dist=None):
    """compute theta^A/B_j,k from given coop prob. distribution for coop neighbours"""
    f_jk = get_f_jkAB(j_dist,degree_dist)
    thetas = pd.concat([f_jk.loc[:,['j','k']],f_jk.groupby(['j','k'])[['A','B']].transform(pd.Series.cumsum)],axis=1)  
    theta_A = thetas.groupby(['j','k'])['A'].sum().reset_index(name='theta_A')
    theta_B = thetas.groupby(['j','k'])['B'].sum().reset_index(name='theta_B')
    return theta_A,theta_B

def theta_AB(df):
    """compute theta^A/B_j,k for neighbour data in df"""
    j_dist = get_coop_coop_neighbour_dist(df) 
    degree_dist = get_degree_distribution(df) 
    return get_theta_AB(j_dist,degree_dist)

def theta_AB_wm(Z,degree_dist=None):
    """compute theta^A/B_j,k for well-mixed population with given degree_dist (if None assume 6 neighbours)"""
    kvals = degree_dist.index if degree_dist is not None else [6]
    j_dist = get_coop_coop_neighbour_dist_well_mixed(Z,kvals)
    return get_theta_AB(j_dist,degree_dist)


def critical_benefit_to_cost0(theta_A,theta_B,Z,benefit_function,*params):
    """calculate (b/c)*_0 from given theta^A_jk, theta^B_jk for a given benefit function/params
        this is threshold b/c at which rho_A=1/Z
    """
    return Z*(Z-1)/(2*(sum(theta_A.theta_A*benefit_function(theta_A.j+1,theta_A.k+1,*params))
                            -sum(theta_B.theta_B*benefit_function(theta_B.j,theta_B.k+1,*params))))

def critical_benefit_to_cost1(sigma_df,Z,benefit_function,*params):
    """calculate (b/c)*_1 from given sigma_df (structure coefficients) for a given benefit function/params
        this is threshold b/c at which rho_A=rho_B
    """
    return sum(sigma_df.sigma)/(sigma_df.sigma*(benefit_function((sigma_df.j+1),(sigma_df.k+1),*params)-benefit_function((sigma_df.k-sigma_df.j),(sigma_df.k+1),*params))).sum()
    
def mean_payoffs(dist,benefit_function,b,c,*params):
    """return expected payoffs for A and B cells against n as dataframe"""
    dist['A_pay'] = (b*benefit_function(dist.j+1,dist.k+1,*params)-c)*dist.A
    dist['B_pay'] = (b*benefit_function(dist.j,dist.k+1,*params))*dist.B
    return dist.groupby('n')[['A_pay','B_pay']].sum()

def payoff_difference(dist,benefit_function,b,c,*params):
    """return difference in expected payoffs for A and B cells against n as series"""
    payoffs = mean_payoffs(dist,benefit_function,b,c,*params)
    return (payoffs['A_pay']-payoffs['B_pay']).rename("pdiff")

#-----------------------DEFINE BENEFIT FUNCTIONS FOR VARIOUS GAMES -----------------------------------------------------
def NPD_benefit(j,N):
    """N-player prisoners dilemma"""
    return j/N

def VD_benefit(j,N,M):
    """Volunteer's dilemma"""
    return 1*(j>=M)
    
def sigmoid_benefit(j,N,h,s):
    return (logistic_function(j/N,h,s)-logistic_function(0,h,s))/(logistic_function(1,h,s)-logistic_function(0,h,s))

def logistic_function(x,h,s):
    return 1./(1+np.exp(s*(h-x)))

#-------------------BENEFIT-TO-COST RATIOS FOR VARIOUS BENEFIT FUNCTIONS ------------------------------------------------
def NPD_critical_b_to_c(sigma_df,Z):
    """return (b/c*_1) for an NPD game with structure coefficients in sigma_df"""
    return (Z-1)/sum(sigma_df.sigma*(2*sigma_df.j+1-sigma_df.k)/(sigma_df.k+1))
    
def sigmoid_critical_df1(sigma_df,hvals,svals):
    """calculate (b/c*_1) for a sigmoid game with structure coefficients in sigma_df
        returns dataframe with (b/c*_1) for given h and s values"""
    return pd.DataFrame([{'h':h,'s':s,'b_to_c_star':critical_benefit_to_cost1(sigma_df,Z,sigmoid_benefit,h,s)} 
                            for h in hvals for s in svals])

def sigmoid_critical_df0(theta_A,theta_B,hvals,svals):
    """calculate (b/c*_0) for a sigmoid game with given theta_A and theta_B
        returns dataframe with (b/c*_0) for given h and s values"""
    return pd.DataFrame([{'h':h,'s':s,'b_to_c_star':critical_benefit_to_cost0(theta_A,theta_B,Z,sigmoid_benefit,h,s)} 
                            for h in hvals for s in svals])
    
def sigmoid_payoff_difference(df,hvals,svals,bvals,c=1):
    """return difference in expected A and B payoffs against n for given b, h and s values as dataframe"""
    f_jk = get_f_jkAB(get_coop_coop_neighbour_dist(df),get_degree_distribution(df))
    return pd.concat([payoff_difference(f_jk,sigmoid_benefit,b,c,h,s).assign(h=h,s=s,b=b)
                for h,s,b in product(hvals,svals,bvals)]).reset_index()  

def sigmoid_payoffs(df,hvals,svals,bvals,c=1):
    """return expected A and B payoffs against n for given b, h and s values as dataframe"""
    f_jk = get_f_jkAB(get_coop_coop_neighbour_dist(df),get_degree_distribution(df))
    df = pd.concat([mean_payoffs(f_jk,sigmoid_benefit,b,c,h,s).assign(h=h,s=s,b=b)
                for h,s,b in product(hvals,svals,bvals)]).reset_index() 
    df = df.set_index(['n','h','s','b'])
    return pd.concat([df['A_pay'].rename('payoff').to_frame().assign(type='A'),df['B_pay'].rename('payoff').to_frame().assign(type='B')]).reset_index()
    

if __name__ == '__main__':    
    df = load_data(READFILE)
    Z=100
    
    theta_A,theta_B = theta_AB(df)
    sigma_vt = structure_coeffs(df)
    
    # save critcial benefit-to-cost ratios for beneficial (0) and favoured (1) cooperations with
    # logistic benefit function and given values for s and h
    hvals = [i/100. for i in range(101)]
    svals = [1,5,10]
    sigmoid_df0 = sigmoid_critical_df0(theta_A,theta_B,hvals,svals)
    sigmoid_df0.to_csv('sigmoid_critical_prediction0',index=False)
    sigmoid_df1 = sigmoid_critical_df0(sigma_vt,hvals,svals)
    sigmoid_df1.to_csv('sigmoid_critical_prediction1',index=False)
    
    # print critical b-to-c for favoured cooperation with NPD game for VT model and well-mixed population
    degree_dist = get_degree_distribution(df)
    sigma_wm = structure_coeffs_well_mixed(Z,degree_dist=degree_dist)
    print('well mixed NPD: {:.3f}'.format(critical_benefit_to_cost1(sigma_wm,Z,NPD_benefit)))
    print('vt model NPD: {:.3f}'.format(critical_benefit_to_cost1(sigma_vt,Z,NPD_benefit)))
    