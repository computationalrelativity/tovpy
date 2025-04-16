"""
Copyright (C) 2024 Sebastiano Bernuzzi and others

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import sys, os, shutil
import numpy as np
from numpy import log, exp
import scipy as sp
from scipy.integrate import solve_ivp, odeint
from scipy.special import factorial2, gamma, factorial2, hyp2f1, poch
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, bisect
from math import comb, prod

from . import eos
from . import units
from .eos import EOS
from .units import Units

class TOV(object):
    
    """ 

    Class to solve the Tolman-Oppenheimer-Volkov stellar structure
    equations together with even/odd parity stationary bartropic perturbations
    
    Lindblom , Astrophys. J. 398 569. (1992) 
    Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
    
    Work in geometric units

    Reference codes:
    * https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_neutron_star_t_o_v_8c_source.html
    * https://bitbucket.org/bernuzzi/tov/src/master/TOVL.m
    * https://lscsoft.docs.ligo.org/bilby/_modules/bilby/gw/eos/tov_solver.html

    """

    def __init__(self,
                 eos        = None, # EOS instance
                 leven      = [], # multipole indexes of even perturbations 
                 lodd       = [], # multipole indexes of odd perturbations 
                 dhfact     = -1e-12, # ODE step
                 ode_method = 'DOP853',
                 ode_atol   = 1e-9,
                 ode_rtol   = 1e-9): 

        if not eos:
            raise ValueError("Must provide a EOS")
        self.eos = eos

        leven = np.array(leven)
        lodd  = np.array(lodd)

        # Solve perturbation equations for these indexes
        self.leven = leven[leven>1]
        self.lodd  = lodd[lodd>1]

        # Build variable list
        var = self.__buildvars()
        self.nvar = len(var)
        self.var  = dict(zip(var, range(self.nvar)))
        self.ivar = {v: k for k, v in self.var.items()}

        # ODE solver options
        if dhfact > 0.:
            raise ValueError("ODE timestep must be negative")
        self.dhfact = dhfact
        self.ode_method = ode_method        
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol

        
    def __buildvars(self):
        """
        List of varnames
        """
        v = ['r','m','nu']
        for l in self.leven:
            v.append('H{}'.format(l))
            v.append('dH{}'.format(l))
        for l in self.lodd:
            v.append('Psi{}'.format(l))
            v.append('dPsi{}'.format(l))
        return v
    
    def __pert_even(self,ell,m,r,p,e,dedp,dnu_dr=[]):
        """
        Eq.(27-29) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
        https://arxiv.org/abs/0906.0096
        Note only C0 depends on ell: return an array of values
        """
        r2       = r**2
        r3       = r * r2
        div_r    = 1.0/r
        div_r2   = div_r**2
        exp_lam  = 1.0 / (1.0 - 2.0 * m * div_r ) # Eq. (18)
        if not dnu_dr:
            dnu2 = (2.0 * (m + 4.0 * np.pi * r3 * p) / (r * (r - 2.0 * m)))**2
        else:
            dnu2 = dnu_dr**2
        C1 = 2.0/r + exp_lam * ( 2*m*div_r2 + 4*np.pi*r*(p-e) ) 
        C0 = np.zeros(max(ell)+1)
        for l in ell:
            Lam = l*(l+1)
            C0[l] = -dnu2
            C0[l] += exp_lam * ( -Lam*div_r2 + 4*np.pi*( 5*e + 9*p + (e + p) * dedp ) ) 
        return C1, C0
                
    def __pert_odd(self,ell,m,r,p,e,dedp):
        """
        Eq.(31) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
        https://arxiv.org/abs/0906.0096
        Note only C0 depends on ell: return an array of values
        """
        r2 = r**2
        r3 = r * r2
        div_r = 1.0/r
        div_r2 = div_r**2
        div_r3 = div_r*div_r2
        exp_lam = 1.0 / (1.0 - 2.0 * m * div_r ) # Eq. (18)
        C1 = exp_lam * ( 2*m + 4*np.pi*r3*(p-e) ) * div_r2
        C0 = np.zeros(max(ell)+1)
        for l in ell:
            Lam = l*(l+1)
            C0[l] = exp_lam*( -Lam*div_r2 + 6*m*div_r3 - 4*np.pi*(e-p) )
        return C1, C0

    def __tov_rhs(self,h,y):
        """
        ODE r.h.s. for TOV equations with pseudo-enthalpy independent variable.
        Implements Eqs. (5) and (6) of Lindblom, Astrophys. J. 398, 569 (1992).
        Also uses Eqs. (7) and (8) [ibid] for inner boundary data, and
        Eqs. (18), (27), (28) of Damour & Nagar, Phys. Rev. D 80, 084035 (2009)
        for the metric perturbation used to obtain the Love number. 
        """
        dy = np.zeros_like(y)
        # Unpack y
        r  = y[self.var['r']]
        m  = y[self.var['m']]
        # EOS call
        p    = self.eos.Pressure_Of_PseudoEnthalpy(h)
        e    = self.eos.EnergyDensity_Of_PseudoEnthalpy(h)
        dedp = self.eos.EnergyDensityDeriv_Of_Pressure(p)
        # print(f'vars: r:{r} m:{m} h:{h} p:{p} e:{e}')
        # TOV
        dr_dh  = -r * (r - 2.0 * m)/(m + 4.0*np.pi*r**3*p)
        dm_dh  = 4.0 * np.pi * r**2 * e * dr_dh 
        dnu_dr =  2.0 * (m + 4.0 * np.pi * r**3 * p) / (r * (r - 2.0 * m))
        dy[self.var['r']] = dr_dh
        dy[self.var['m']] = dm_dh
        dy[self.var['nu']] = dnu_dr * dr_dh        
        # print('derivs:', dr_dh, dm_dh, dnu_dr)
        # Even perturbations
        if len(self.leven) != 0:
            C1,C0 = self.__pert_even(self.leven,m,r,p,e,dedp,dnu_dr)
            for l in self.leven:
                H = y[self.var['H{}'.format(l)]]
                dH = y[self.var['dH{}'.format(l)]]
                dH_dh = dH * dr_dh
                ddH_dh = -(C0[l] * H + C1 * dH) * dr_dh
                dy[self.var['H{}'.format(l)]] = dH_dh
                dy[self.var['dH{}'.format(l)]] = ddH_dh    
        # Odd perturbations
        if len(self.lodd) != 0:
            C1,C0 = self.__pert_odd(self.lodd,m,r,p,e,dedp)
            for l in self.lodd:
                Psi  = y[self.var['Psi{}'.format(l)]]
                dPsi = y[self.var['dPsi{}'.format(l)]]
                dPsi_dh = dPsi * dr_dh
                ddPsi_dh = -(C0[l] * Psi + C1 * dPsi) * dr_dh
                dy[self.var['Psi{}'.format(l)]] = dPsi_dh
                dy[self.var['dPsi{}'.format(l)]] = ddPsi_dh   
        return dy

    def __initial_data(self,pc,dh_fact=-1e-12,verbose=False):
        """
        Set initial data for the solution of TOV equations using the pseudo-enthalpy formalism introduced in:
        Lindblom (1992) "Determining the Nuclear Equation of State from Neutron-Star Masses and Radii", Astrophys. J. 398 569.
        * input the central pressure
        """
        y = np.zeros(self.nvar)
        # Central values 
        ec     = self.eos.EnergyDensity_Of_Pressure(pc)
        hc     = self.eos.PseudoEnthalpy_Of_Pressure(pc)
        dedp_c = self.eos.EnergyDensityDeriv_Of_Pressure(pc)
        dhdp_c = 1.0 / (ec + pc)
        dedh_c = dedp_c / dhdp_c
        dh = -1e-12 * hc
        h0 = hc + dh
        h1 = 0.0 - dh
        r0 = np.sqrt(-3.0 * dh / (2.0 * np.pi * (ec + 3.0 * pc)))
        m0 = 4.0 * np.pi * r0**3 * ec / 3.0
        # Series expansion for the initial core 
        r0 *= 1.0 + 0.25 * dh * (ec - 3.0 * pc  - 0.6 * dedh_c) / (ec + 3.0 * pc) # second factor Eq. (7) of Lindblom (1992) 
        m0 *= 1.0 + 0.6 * dh * dedh_c / ec # second factor of Eq. (8) of Lindblom (1992) 
        y[self.var['r']]  = r0
        y[self.var['m']]  = m0
        y[self.var['nu']] = 0.0
        #  Initial data for the ell-perturbation
        a0 = 1.0
        if len(self.leven)!= 0:
            for l in self.leven:
                y[self.var['H{}'.format(l)]] = a0 * r0**l
                y[self.var['dH{}'.format(l)]] = a0 * l * r0**(l-1)
        if len(self.lodd)!= 0:
            for l in self.lodd:
                y[self.var['Psi{}'.format(l)]] = a0 * r0**(l+1)
                y[self.var['dPsi{}'.format(l)]] = a0 * (l+1) * r0**l
        # if verbose:
        #     print("pc = {:.8e} hc = {:.8e} dh = {:.8e} h0  = {:.8e}".format(pc,hc,dh,h0))
        #     print(y, self.ivar)
        return y, h0, h1
    
    def solve(self,pc):
        """
        Solves the Tolman-Oppenheimer-Volkov stellar structure equations using the pseudo-enthalpy formalism introduced in:
        Lindblom (1992) "Determining the Nuclear Equation of State from Neutron-Star Masses and Radii", Astrophys. J. 398 569.
        """
        # Initial data
        y, h0, h1 = self.__initial_data(pc, dh_fact=self.dhfact, verbose=True)
        # Integrate
        # print("Integrating TOV equations")
        # print("h0 = {:.8e} h1 = {:.8e}".format(h0,h1))
        sol = solve_ivp(self.__tov_rhs, [h0, h1], y,
                        first_step = abs(self.dhfact),
                        method = self.ode_method,
                        rtol = self.ode_rtol,
                        atol = self.ode_atol)
    
        # Take one final Euler step to get to surface 
        y  = sol.y[:,-1]
        dy = self.__tov_rhs(sol.t[-1],y)
        y[:] -= dy[:] * h1
        np.append(sol.y, y)
        # Mass, Radius & Compactness
        M,R,C = self.__compute_mass_radius(y)
        # Match to Schwarzschild exterior
        sol.y[self.var['nu'],:] += np.log(1.0-(2.*M)/R) - sol.y[self.var['nu'],-1]
        # Even Love number k2 (for testing purposes only)
        #yyl = R*y[self.ivar['dH2']]/y[self.ivar['H2']]
        #k2 = self.__compute_Love_even_ell2(self, self.C,yyl)
        # Even Love numbers

        self.sol = sol
        if len(self.leven) != 0:
            k, h = {}, {}
            for l in self.leven:
                yyl = R * y[self.var['dH{}'.format(l)]] / y[self.var['H{}'.format(l)]]
                k[l] = self.__compute_Love_even(l,C,yyl)   
                h[l] = self.__compute_shape(l,C,yyl)
        if len(self.lodd) != 0:                
        # Odd Love numbers
            j = {}
            for l in self.lodd:
                yyl = R*y[self.var['dPsi{}'.format(l)]]/y[self.var['Psi{}'.format(l)]]
                j[l]= self.__compute_Love_odd(l,C,yyl)

        if len(self.leven)!= 0 and len(self.lodd)!= 0:
            return M,R,C,k,h,j
        elif len(self.leven)!= 0 and len(self.lodd)== 0:
            return M,R,C,k,h
        elif len(self.leven)== 0 and len(self.lodd)!= 0:
            return M,R,C,j
        else:
            return M,R,C

    def __compute_legendre(self, c, l):
        """
        Computes Legendre function values returning Pl2(x), Ql2(x) and their derivatives at x = 1/c -1
        """
        x = 1/c -1
        L = np.linspace(0,l-1,l)
        nP = -prod((2*l-1)/2-L)/gamma(l) * 2**l * l*(l-1)
        nQ = gamma(l)/factorial2(2*l+1)*(l+1)*(l+2)

        Pl2 = 0
        dPl2 = 0
        for i in np.linspace(2,l,l-2+1,dtype=int):
            Pl2 = Pl2 + gamma(i)/gamma(i-2) * comb(l,i) * prod((l+i-1)/2-L) / gamma(l) * x**(i-2)
            dPl2 = dPl2 + gamma(i)/gamma(i-2) * comb(l,i) * prod((l+i-1)/2-L) / gamma(l) * (i-2) * x**(i-3)
        
        dPl2 = 2**l*(-2*x)*Pl2/nP + 2**l*(1-x**2)*dPl2/nP
        Pl2  = 2**l*(1-x**2)*Pl2/nP

        Ql2  = 1/nQ * np.sqrt(np.pi)/2**(l+1) * gamma(l+3)/gamma(l+3/2) * (x**2-1)/x**(l+3) * hyp2f1((l+3)/2, (l+4)/2,l+3/2,1/x**2)
        dQl2 = 1/nQ * np.sqrt(np.pi)/2**(l+1) * gamma(l+3)/gamma(l+3/2) * (2*x**(-2 - l)*hyp2f1((l+3)/2, (l+4)/2,l+3/2,1/x**2) +\
                                                            (-3 - l)*x**(-4 - l)*(-1 + x**2)*hyp2f1((l+3)/2, (l+4)/2,l+3/2,1/x**2) -\
                                                            (2*((l+3)/2)*((l+4)/2)*x**(-6 - l)*(-1 + x**2)*hyp2f1((l+3)/2+1, (l+4)/2+1,l+3/2+1,1/x**2)/(l+3/2)))
        return Pl2,dPl2,Ql2,dQl2
    
    def __compute_psi(self, c, l):
        x = 1/c
        CoefficientP = poch(5, l-2) / poch (2-l, l-2) / poch(3+l, l-2) * gamma(l-2) * 2 ** (l-2)
        CoefficientQ = -1 / (l+2)
        psiP = x**3 * hyp2f1(2-l, 3+l, 5, x/2) * CoefficientP
        psiQ = - (l+2) * x**(-1-l) * ((1+l) * x * hyp2f1(-1+l,2+l,2+2*l,2/x) + (-1+l)*hyp2f1(l,3+l,3+2*l,2/x) )/(1+l) * CoefficientQ
        dPsiP = 3 * x**2 * hyp2f1(2-l, 3+l, 5, x/2) - 1/10 * (-6 + l + l**2) * x**3 * hyp2f1(3-l, 4+l, 6, x/2)
        dPsiP = dPsiP * CoefficientP
        dPsiQ = 1/(1+l)/(3+2*l) * (2+l) * x**(-3-l) * (
            l*(3+5*l+2*l**2)*x**2*hyp2f1(-1+l, 2+l, 2+2*l, 2/x) +
            (-1+l)*(
                (3+2*l)**2*x*hyp2f1(l, 3+l, 3+2*l, 2/x) +
                2*l*(3+l)*hyp2f1(1+l, 4+l, 4+2*l, 2/x)
            )
        )
        dPsiQ = dPsiQ * CoefficientQ
        return psiP, dPsiP, psiQ, dPsiQ

    def __compute_mass_radius(self, y):
        """
        Compute mass, radius, & compactness
        """
        R = y[self.var['r']]
        M = y[self.var['m']]
        return M,R,M/R

    def Compute_baryon_mass(self, sol):
        """
        Compute baryon mass
        """
        r = sol.y[self.var['r'],:]
        m = sol.y[self.var['m'],:]
        # e = self.EOSEnergyDensityOfPseudoEnthalpyGeometerized(sol.t,self.eos)
        e = np.array([self.eos.EnergyDensity_Of_PseudoEnthalpy(sol.t[i]) for i in range(len(sol.t))])
        return np.trapz( 4*np.pi*r**2.*e/np.sqrt(1-2*m/r), r )

    def Compute_proper_radius(self, sol):
        """
        Compute baryon mass
        """
        r = sol.y[self.var['r'],:]
        m = sol.y[self.var['m'],:]
        return np.trapz( r, 1./np.sqrt((1-2*m/r)), r )
        
    def __compute_Love_odd(self,ell,c,y):
        """
        Compute odd parity Love numbers given 
        * the multipolar index ell
        * the compactness c
        * the ratio y = R Psi(R)'/Psi(R) 
        Eq.(61) of Damour & Nagar, Phys. Rev. D 80 084035 (2009)
        """
        c2 = c**2
        c3 = c*c2
        c4 = c*c3
        c5 = c*c4
        j = 0.
        if ell == 2:
            nj =  96*c5*(-1 + 2*c)*(-3 + y)
            dj =  5.*(2*c*(9 + 3*c*(-3 + y) + 2*c2*(-3 + y) + 2*c3*(-3 + y) - 3*y + 12*c4*(1 + y)) + 3*(-1 + 2*c)*(-3 + y)*log(1 - 2*c))
            j = nj/dj
        else:
            PsiP, dPsiP, PsiQ, dPsiQ = self.__compute_psi(c,ell)
            factor =  - c ** (2 * ell + 1)
            j = factor * (dPsiP - c * y * PsiP) / (dPsiQ - c * y * PsiQ)
        return j
    
    def __compute_Love_even(self,ell,c,y):
        """
        Compute even parity Love numbers given 
        * the multipolar index ell
        * the compactness c
        * the ratio y = R H(R)'/H(R) 
        Eq.(49) of Damour & Nagar, Phys. Rev. D 80 084035 (2009)
        """
        c2 = c**2
        c3 = c*c2
        c4 = c*c3
        c5 = c*c4
        c6 = c*c5
        c7 = c*c6
        c8 = c*c7
        c9 = c*c8
        c10 = c*c9
        c11 = c*c10
        c13 = c2*c11
        c15 = c2*c13
        c17 = c2*c15
        k = 0.
        if ell < 2: return k
        if ell == 2:
            nk = (1-2*c)**2*(2+2*c*(y-1)-y)
            dk = 2*c*(6-3*y+3*c*(5*y-8))+4*c3*(13-11*y+c*(3*y-2)+2*c2*(1+y)) + 3*(1-2*c)**2*(2-y+2*c*(y-1))*np.log(1-2*c)
            k = 8*c5/5*nk/dk
        elif ell == 3:
            nk = (1 - 2*c)**2*(-3 - 3*c*(-2 + y) + 2*c2*(-1 + y) + y)
            dk = 2*c*(15*(-3 + y) + 4*c5*(1 + y) - 45*c*(-5 + 2*y) - 20*c3*(-9 + 7*y) + 2*c4*(-2 + 9*y) + 5*c2*(-72 + 37*y)) - 15*(1 - 2*c)**2*(-3 - 3*c*(-2 + y) + 2*c**2*(-1 + y) + y)*np.log(1.0/(1 - 2*c))
            k = 8*c7/7*nk/dk
        elif ell == 4:
            nk = (1 - 2*c)**2*(-7*(-4 + y) + 28*c*(-3 + y) - 34*c2*(-2 + y) + 12*c3*(-1 + y))
            dk = (2*c*(c2*(5360 - 1910*y) + c4*(1284 - 996*y) - 105*(-4 + y) + 8*c6*(1 + y) + 105*c*(-24 + 7*y) + 40*c3*(-116 + 55*y) + c5*(-8 + 68*y)) - 15*(1 - 2*c)**2*(-7*(-4 + y) + 28*c*(-3 + y) - 34*c2*(-2 + y) + 12*c3*(-1 + y))*np.log(1.0/(1 - 2*c)))
            k = 32*c9/147*nk/dk
        elif ell == 5:
            nk = (32*(1 - 2*c)**2*c11*(3*(-5 + y) - 15*c*(-4 + y) + 26*c2*(-3 + y) - 18*c3*(-2 + y) + 4*c4*(-1 + y)))
            dk = 99.*(2*c*(315*(-5 + y) + 8*c7*(1 + y) - 315*c*(-35 + 8*y) + 4*c6*(-2 + 27*y) - 56*c5*(-60 + 47*y) - 210*c3*(-170 + 57*y) + 105*c2*(-278 + 75*y) + 56*c4*(-345 + 158*y)) - 105*(1 - 2*c)**2*(3*(-5 + y) - 15*c*(-4 + y) + 26*c2*(-3 + y) - 18*c3*(-2 + y) + 4*c4*(-1 + y))*np.log(1.0/(1 - 2*c)))
            k = nk/dk
        elif ell == 6:
            nk = (1024*(1 - 2*c)**2*c13*(-33*(-6 + y) + 198*c*(-5 + y) - 444*c2*(-4 + y) + 456*c3*(-3 + y) - 208*c4*(-2 + y) + 32*c5*(-1 + y)))
            dk = 14157.*(2*c*(-3465*(-6 + y) + 32*c8*(1 + y) + 10395*c*(-16 + 3*y) + 16*c7*(-2 + 39*y) + 2016*c5*(-122 + 55*y) - 64*c6*(-457 + 362*y) - 210*c2*(-2505 + 541*y)  + 210*c3*(-3942 + 1015*y) - 84*c4*(-7917 + 2567*y)) - 105*(1 - 2*c)**2*(-33*(-6 + y) + 198*c*(-5 + y) - 444*c2*(-4 + y) + 456*c3*(-3 + y) - 208*c4*(-2 + y) + 32*c5*(-1 + y))*np.log(1.0/(1 - 2*c)))
            k = nk/dk
        elif ell == 7:
            nk = 1024*(1 - 2*c)**2*c15*(143*(-7 + y) - 1001*c*(-6 + y) + 2750*c2*(-5 + y) - 3740*c3*(-4 + y) + 2600*c4*(-3 + y) - 848*c5*(-2 + y) + 96*c6*(-1 + y))
            dk = 20449.*(2*c*(45045*(-7 + y) + 160*c9*(1 + y) - 45045*c*(-63 + 10*y) + 80*c8*(-2 + 53*y) - 432*c7*(-651 + 521*y) - 4620*c3*(-4333 + 902*y)  + 1155*c2*(-9028 + 1621*y) + 96*c6*(-33964 + 15203*y) + 126*c4*(-168858 + 42239*y) - 84*c5*(-144545 + 45971*y)) - 315*(1 - 2*c)**2*(143*(-7 + y) - 1001*c*(-6 + y) + 2750*c2*(-5 + y) - 3740*c3*(-4 + y) + 2600*c4*(-3 + y) - 848*c5*(-2 + y) + 96*c6*(-1 + y))*np.log(1.0/(1 - 2*c)))
            k = nk/dk
        elif ell == 8:
            nk = (16*(1 - 2*c)**2*(2*c*(c*(2*c*(2*c*(-737*(-4 + y) + 374*c*(-3 + y) - 92*c2*(-2 + y) + 8*c**3*(-1 + y)) + 1573*(-5 + y)) - 1859*(-6 + y)) + 572*(-7 + y)) - 143*(-8 + y)))
            dk = (286*c*(4*(90090 + c*(-900900 + c*(3768765 + c*(-8528520 + c*(11259633 + 2*c*(-4349499 + c*(1858341 + 8*c*(-47328 + c*(3092 + (-1 + c)*c))))))))) + (-1 + c)*(-1 + 2*c)*(-45045 + 4*c*(90090 + c*(-285285 + c*(450450 + c*(-365211 + 4*c*(34881 + c*(-4887 + 2*c*(36 + c))))))))*y) - 45045*(1 - 2*c)**2*(2*c*(c*(2*c*(2*c*(-737*(-4 + y) + 374*c*(-3 + y) - 92*c2*(-2 + y) + 8*c**3*(-1 + y)) + 1573*(-5 + y)) - 1859*(-6 + y)) + 572*(-7 + y)) - 143*(-8 + y))*np.log(1.0/(1 - 2*c)))
            k = 256/2431 * c17 * nk/dk
        else:
            # https://bitbucket.org/bernuzzi/tov/src/master/ComputeLegendre.m
            Pl2, dPl2, Ql2, dQl2 = self.__compute_legendre(c, ell)
            k = -1/2*c**(2*ell+1)*(dPl2-c*y*Pl2)/(dQl2-c*y*Ql2)
        return k
        
    def __compute_shape(self,ell,c,y):
        """
        Compute even shape numbers given 
        * the multipolar index ell
        * the compactness c
        * the ratio y = R H(R)'/H(R) 
        Eq.(95) of Damour & Nagar, Phys. Rev. D 80 084035 (2009)
        """
        c2 = c**2
        c3 = c*c2
        c4 = c*c3
        c5 = c*c4
        c6 = c*c5
        c7 = c*c6
        c8 = c*c7
        c9 = c*c8
        c10 = c*c9
        c11 = c*c10
        c13 = c2*c11
        c15 = c2*c13
        c17 = c2*c15
        h = 0.
        if ell < 2: return h
        if ell == 2:
            nh = (-2 + 6*c + 2*c3*(1 + y) - c2*(6 + y))
            dh = (2*c*(6 + c2*(26 - 22*y) - 3*y + 4*c4*(1 + y) + 3*c*(-8 + 5*y) + c3*(-4 + 6*y)) - 3*(1 - 2*c)**2*(2 + 2*c*(-1 + y) - y)*np.log(1.0/(1 - 2*c)))
            h = -8*c5*nh/dh
        # elif ell == 3:
        #     nh = -5 + 15*c + 2*c3*(1 + y) - c2*(12 + y)
        #     dh = (5.*(2*c*(15*(-3 + y) + 4*c5*(1 + y) - 45*c*(-5 + 2*y) - 20*c3*(-9 + 7*y) + 2*c4*(-2 + 9*y) + 5*c2*(-72 + 37*y)) - 15*(1 - 2*c)**2*(-3 - 3*c*(-2 + y) + 2*c2*(-1 + y) + y)*np.log(1.0/(1 - 2*c))))
        #     h = 16*c7*nh/dh
        # elif ell == 4:
        #     nh = -9 + 27*c + 2*c3*(1 + y) - c2*(20 + y)
        #     dh = (21.*(2*c*(c2*(5360 - 1910*y) + c4*(1284 - 996*y) - 105*(-4 + y) + 8*c6*(1 + y) + 105*c*(-24 + 7*y)  + 40*c3*(-116 + 55*y) + c5*(-8 + 68*y)) - 15*(1 - 2*c)**2*(-7*(-4 + y) + 28*c*(-3 + y) - 34*c2*(-2 + y) + 12*c3*(-1 + y))*np.log(1.0/(1 - 2*c))))
        #     h = -64*c9*nh/dh
        else:
            Pl2, dPl2, Ql2, dQl2 = self.__compute_legendre(c,ell)
            term1 = (1-2*c)/c
            term2 = 1/(ell-1)/(ell+2) * (2*c*y + ell*(ell+1) + 4*c**2/(1-2*c) - 2*(1-2*c))
            factor = c**(ell+1)*Pl2 * (1-(dPl2/Pl2-c*y)/(dQl2/Ql2-c*y))
            h = (term1 + term2) * factor
        return h
    
    def Compute_Lambda(self,ell,k,C):
        r"""
        Compute tidal polarizability $\Lambda_\ell$
        from Love numbers and compactness
        Note: Yagi's $\bar{\lambda}_\ell$ is $\Lambda_\ell$
        """
        div = 1.0/(factorial2(2*ell-1)*C**(2*ell+1))
        return 2.*k*div


