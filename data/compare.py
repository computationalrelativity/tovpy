# This file is here for comparison with MATLAB data and from traditional method of assuming an r_c
# when running this code, the current directory should be at tovpy.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pylab
from astropy import constants as const
from numpy import log10 as lg
from scipy.interpolate import interp1d as sp_interp1d
from numpy import pi
from scipy.integrate import ode
from tovpy import Units
import os
uts = Units()

G_cgs=const.G.cgs.value
c_cgs=const.c.cgs.value
Ms_cgs=const.M_sun.cgs.value
hbar_cgs=const.hbar.cgs.value
m_n_cgs=const.m_n.cgs.value

params = {'text.usetex': True}
pylab.rcParams.update(params)
FS = 22
matplotlib.rcParams['figure.figsize'] = (10., 10./1.618)
matplotlib.rcParams['figure.titlesize'] = FS
matplotlib.rcParams['figure.subplot.wspace'] = 0.
matplotlib.rcParams['figure.subplot.hspace'] = 0.2
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
matplotlib.rcParams['xtick.labelsize'] = FS
matplotlib.rcParams['ytick.labelsize'] = FS
matplotlib.rcParams['axes.labelsize'] = FS
matplotlib.rcParams['legend.fontsize'] = FS
matplotlib.rcParams['legend.borderpad'] = 0.1
matplotlib.rcParams['legend.loc'] = 'best'
matplotlib.rcParams['legend.borderaxespad'] = 0.2
matplotlib.rcParams['legend.fancybox'] = True
matplotlib.rcParams['axes.titlesize'] = FS

eos_name = input("Enter EOS name: ")
# eos name to be chosen from: 
# 2B.rns, 2H.rns, ALF2.rns,BHB_lp_25-Sept-2017.rns,BLQ_30-Jan-2020.rns,BLh_gibbs_180_0.35_10-Jul-2019.rns,DD2_22-Jun-2018.rns,
# DD2_25-Sept-2017.rns,ENG.rns,H4.rns,LS1800B0.rns,LS220B0.rns,LS220B0v2.rns,LS_220_25-Sept-2017.rns,MPA1.rns,MS1.rns,MS1b.rns,
# NL3_05-Oct-2017.rns,SFHo+BL_01-Apr-2019.rns,SFHo_09-Feb-2019.rns,SFHo_25-Sept-2017.rns,SLy.rns,SLy4.rns,SLy4_15-Jun-2018.rns,
# TM1_05-Oct-2017.rns,TMA_05-Oct-2017.rns,eosA,eosAPR,eosAPR_fit,eosAU,eosB,eosC,eosF,eosFP,eosFPS,eosFPS_fit,eosFPS_old,eosG,
# eosL,eosN,eosNV,eosO,eosSLy,eosSLy_fit,eosUU,eosWNV,eosWS,eos_DD2_adb.rns,eos_SFHo_adb.rns,eos_SFHx_adb.rns
# To note that LS1800B0.rns, eosAPR, eosFP, eosFPS, eosSLy, eosWNV are not included in the list above
# and that typing '.rns' is optional

try:
    data_tovpy = np.loadtxt('data/'+eos_name+'.txt', delimiter='\t', skiprows=1)
    data_eos=np.loadtxt('eos/'+eos_name, skiprows=1)

except FileNotFoundError:
    try:
        data_tovpy = np.loadtxt('data/'+eos_name+'.rns.txt', delimiter='\t', skiprows=1)
        data_eos=np.loadtxt('eos/'+eos_name+'.rns', skiprows=1)
    except FileNotFoundError:
        raise FileNotFoundError("File not found in 'data'")

M_tovpy = data_tovpy[:,1]
R_tovpy = data_tovpy[:,2]

N = 200   

M = np.zeros(N)
R = np.zeros(N)

def tov(r,y):
    m, p=y[0], y[1] 
    
    rho = p2e(p) 
    
    dmdr=4*np.pi * r**2 * rho
    
    dpdr=-G_cgs*(rho+p/c_cgs**2)*(m+4*pi*r**3*p/c_cgs**2)/r**2/(1-2*G_cgs*m/c_cgs**2/r)
    
    return [dmdr, dpdr] 



def solout(r,y):
    if y[1]<=min_p or y[1]>=max_p:
        return -1
    else:
        return 0

e, p =data_eos[:, 0], data_eos[:, 1]
    
min_p = np.min(p)
max_p = np.max(p)
min_e = np.min(e)
max_e = np.max(e)
                
lgp, lge= lg(p), lg(e)
lgp2lge = sp_interp1d(lgp, lge)
lge2lgp = sp_interp1d(lge, lgp) 
    
  
def p2e(p):
    if p < min_p or p>max_p:
        return 0.0
    else:
        return 10.0**lgp2lge(lg(p))
       
def e2p(e):
    if e<min_e or e>max_e:
        return 0.0
    else:
        return 10.0**lge2lgp(lg(e))
            
a, b = 14,  lg(max_e)
rhoc=10**(np.linspace(a,b,N))
    
r0=1
r1=2*10**7
    
    
for i in range (len(rhoc)): 
        
    y=np.zeros(4)
    y[0]=4*pi*r0**3/3*rhoc[i]  
    y[1]=e2p(rhoc[i])
    y0=[y[0], y[1]]
    
    solver=ode(tov).set_integrator('dopri5')
    solver.set_solout(solout)
    solver.set_initial_value(y0,r0)
    results=solver.integrate(r1)
    M[i]=results[0]
    R[i]=solver.t
plt.plot(R[:-1]/10**5,M[:-1]/Ms_cgs, label=str(eos_name)+' traditional',linewidth=2, linestyle='solid')
plt.plot(R_tovpy, M_tovpy, label=str(eos_name)+' tovpy',linestyle='dotted', linewidth=3)
plt.axhline(y = 2.01, color = 'k', ls = '--', label= 'J0348+0432')
plt.fill_between([0,100],y1 = 2.01-0.04, y2 = 2.01+0.04, color = 'grey', alpha = 0.5)
plt.axhline(y = 2.14, color = 'k', ls = '-.', label = 'J0740+6620')
plt.fill_between([0,100],y1 = 2.14-0.09, y2 = 2.14+0.1, color = 'grey', alpha = 0.5)
plt.xlabel('Radius (km)')
plt.ylabel(r'Mass (M$_\odot$)')
plt.xlim(0, 20)
plt.ylim(0, 3)
plt.legend()
module_dir = os.path.dirname(__file__)
save_dir = os.path.join(module_dir, 'figures')
plt.savefig(os.path.join(save_dir, 'compare_'+eos_name+'.pdf'), bbox_inches='tight')
plt.show()