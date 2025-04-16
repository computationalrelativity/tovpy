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

import os
import numpy as np
import matplotlib.pyplot as plt

from . import eos
from . import units
from . import tov
from .eos import EOS
from .units import Units
from .tov import TOV

uts = Units()

class Utils:
    
    """

    Utility class for saving data and quick visualisation of results.

    """

    def __init__(self, eos, p, path=None):
        """
        Parameters
        ----------
        path : str, optional
            Path to the directory where data will be saved. If None, uses the current working directory.
        """
        self.path = path if path else os.getcwd()
        if not os.path.exists(self.path):
            # Create the directory if it doesn't exist
            os.makedirs(self.path, exist_ok=True)
        if not eos:
            raise ValueError("Must provide a EOS")
        self.eos = eos
        if len(p) == 0:
            raise ValueError("Must provide a pressure array")
        self.p = np.array(p)

    def eos_plot(self, savefigon= False, filename=None):
        """
        Plot the equation of state (EOS) for a given central pressure.

        Parameters
        ----------
        p : array_like
            Array of central pressures.
        savefigon : bool, optional
            If True, saves the figure to a file. Default is False.
        filename : str, optional
            Filename to save the figure. If None, uses a default filename. Default is None.
        """
        # Compute the two sets of values from pressure array `p`
        e = np.array([self.eos.EnergyDensity_Of_Pressure(self.p[i]) for i in range(len(self.p))])
        h = np.array([self.eos.PseudoEnthalpy_Of_Pressure(self.p[i]) for i in range(len(self.p))])

        # Create the figure and primary axis
        fig, ax1 = plt.subplots()

        # Plot Energy Density on the primary y-axis in log-log scale
        ax1.loglog(self.p, e, label='Energy Density', color='blue', marker='.')
        ax1.set_xlabel('Pressure in [Geo]')
        ax1.set_ylabel('Energy Density [Geo]', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        ax2.loglog(self.p, h, label='PseudoEnthalpy', color='red', marker='.')
        ax2.set_ylabel('PseudoEnthalpy [Geo]', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.title('Equation of State')
        if savefigon:
            if filename is None:
                filename = os.path.join(self.path, 'eos_plot.pdf')
            else:
                filename = os.path.join(self.path, filename)
            plt.savefig(filename, bbox_inches='tight')
        plt.show()

    def eos_txt(self, filename=None):
        """
        Save the EOS data to a file.

        Parameters
        ----------
        p : array_like
            Array of central pressures.
        filename : str, optional
            Filename to save the EOS data. If None, uses a default filename. Default is None.
        """
        # Compute the two sets of values from pressure array `p`
        e = np.array([self.eos.EnergyDensity_Of_Pressure(self.p[i]) for i in range(len(self.p))])
        h = np.array([self.eos.PseudoEnthalpy_Of_Pressure(self.p[i]) for i in range(len(self.p))])
        dedp = np.array([self.eos.EnergyDensityDeriv_Of_Pressure(self.p[i]) for i in range(len(self.p))])
        data = np.column_stack((self.p, e, h, dedp))
        # Save the data to a file
        if filename is None:
            filename = os.path.join(self.path, 'eos_data.txt')
        else:
            filename = os.path.join(self.path, filename)
        np.savetxt(filename, data, header='Pressure EnergyDensity PseudoEnthalpy EnergyDensityDeriv', delimiter='\t')

    def MR_plot(self, savefigon= False, filename=None):
        """
        Plot the mass-radius relation for a given central pressure.

        Parameters
        ----------
        p : array_like
            Array of central pressures.
        savefigon : bool, optional
            If True, saves the figure to a file. Default is False.
        filename : str, optional
            Filename to save the figure. If None, uses a default filename. Default is None.
        """
        # Compute the two sets of values from pressure array `p`

        this_tov = TOV(eos = self.eos, #ode_method='RK45',
                        ode_atol=1e-10, 
                        ode_rtol=1e-10, 
                        dhfact=-1e-12)
        m_list, r_list, c_list = np.zeros(len(self.p)), np.zeros(len(self.p)), np.zeros(len(self.p))
        for i, pc in enumerate(self.p):
            m, r, c = this_tov.solve(pc)[:3]
            # Convert radius to km and mass to solar masses
            r *= 1./1e3
            m *= 1./uts.constant['MRSUN_SI'][0]
            m_list[i], r_list[i], c_list[i] = m, r, c
        # Create the figure and axis
        fig, ax1 = plt.subplots()
        ax1.plot(r_list, m_list, label='Mass', color='blue', marker='.')
        ax1.set_xlabel('Radius [km]')
        ax1.set_ylabel(r'Mass [M$_\odot$]', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a twin y-axis for the compactness vs. radius plot
        ax2 = ax1.twinx()
        ax2.plot(r_list, c_list, label='Compactness', color='red', marker='.')
        ax2.set_ylabel('Compactness', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Set the title and combine legends from both axes
        plt.title('Mass-Radius & Compactness-Radius Relation')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
       
        if savefigon:
            if filename is None:
                filename = os.path.join(self.path, 'MR_plot.pdf')
            else:
                filename = os.path.join(self.path, filename)
            plt.savefig(filename)
        plt.show()

    def MR_txt(self, filename=None):
        """
        Save the mass-radius data to a file.

        Parameters
        ----------
        p : array_like
            Array of central pressures.
        filename : str, optional
            Filename to save the mass-radius data. If None, uses a default filename. Default is None.
        """
        # Compute the two sets of values from pressure array `p`
        this_tov = TOV(eos = self.eos, #ode_method='RK45',
                        ode_atol=1e-10, 
                        ode_rtol=1e-10, 
                        dhfact=-1e-12)
        m_list, r_list, c_list = np.zeros(len(self.p)), np.zeros(len(self.p)), np.zeros(len(self.p))
        for i, pc in enumerate(self.p):
            m, r, c = this_tov.solve(pc)[:3]
            # Convert radius to km and mass to solar masses
            r *= 1./1e3
            m *= 1./uts.constant['MRSUN_SI'][0]
            m_list[i], r_list[i], c_list[i] = m, r, c
        # Save the data to a file
        if filename is None:
            filename = os.path.join(self.path, 'MR_data.txt')
        else:
            filename = os.path.join(self.path, filename)
        np.savetxt(filename, np.column_stack((r_list, m_list, c_list)), header='Radius Mass Compactness', delimiter='\t')

    def Love_plot(self, leven, lodd, savefigon=False, filename=None):
        """
        Plot the Love number and moment of inertia for a given central pressure.

        Parameters
        ----------
        p : array_like
            Array of central pressures.
        savefigon : bool, optional
            If True, saves the figure to a file. Default is False.
        filename : str, optional
            Filename to save the figure. If None, uses a default filename.
        """
        # Compute the two sets of values from the pressure array `p`
        this_tov = TOV(eos=self.eos, leven=leven, lodd=lodd,  # ode_method='RK45',
                    ode_atol=1e-10, 
                    ode_rtol=1e-10, 
                    dhfact=-1e-12)
        
        c_list = np.zeros(len(self.p))
        k_vars, h_vars = {}, {}
        for l in leven:
            k_vars['k' + str(l)] = np.zeros(len(self.p))
            h_vars['h' + str(l)] = np.zeros(len(self.p))
        j_vars = {}
        for l in lodd:
            j_vars['j' + str(l)] = np.zeros(len(self.p))
        
        for i, pc in enumerate(self.p):
            _, _, c, k, h, j = this_tov.solve(pc)
            c_list[i] = c
            for l in leven:
                k_vars['k' + str(l)][i] = k[l]
                h_vars['h' + str(l)][i] = h[l]
            for l in lodd:
                j_vars['j' + str(l)][i] = j[l]
        
        # Define a list of linestyles to differentiate each l value.
        linestyles = ['-', '--', '-.', ':']
        
        # Create two subplots: one for k and j, and one for the shape h.
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
        
        # --- Top subplot: Plot Love numbers k and j ---
        # Plot Love numbers k using different linestyles.
        for idx, l in enumerate(leven):
            ls = linestyles[idx % len(linestyles)]
            ax1.plot(c_list, k_vars['k' + str(l)],
                    label=f'Love Number k{l}', marker='.', color='blue', linestyle=ls)
        ax1.set_ylabel('Love Number, k', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create a twin y-axis for j values.
        ax1_twin = ax1.twinx()
        for idx, l in enumerate(lodd):
            # For j, we also assign a linestyle from the same list (optional).
            ls = linestyles[idx % len(linestyles)]
            ax1_twin.plot(c_list, j_vars['j' + str(l)],
                        label=f'Love Number j{l}', marker='.', color='red', linestyle=ls)
        ax1_twin.set_ylabel('Love Number, j', color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('Love Numbers')
        # Combine legends from both y-axes.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # --- Bottom subplot: Plot the shape h ---
        for idx, l in enumerate(leven):
            ls = linestyles[idx % len(linestyles)]
            ax2.plot(c_list, h_vars['h' + str(l)],
                    label=f'Shape h{l}', marker='.', color='green', linestyle=ls)
        ax2.set_xlabel('Compactness')
        ax2.set_ylabel('Shape h')
        ax2.legend(loc='best')
        
        plt.tight_layout()
        if savefigon:
            if filename is None:
                filename = os.path.join(self.path, 'Love_plot.pdf')
            else:
                filename = os.path.join(self.path, filename)
            plt.savefig(filename, bbox_inches='tight')
        plt.show()

    def Love_txt(self, leven, lodd, filename=None):
        """
        Save the Love number and moment of inertia data to a file.

        Parameters
        ----------
        p : array_like
            Array of central pressures.
        filename : str, optional
            Filename to save the Love number and moment of inertia data. If None, uses a default filename. Default is None.
        """
        # Compute the two sets of values from pressure array `p`
        this_tov = TOV(eos = self.eos, leven=leven, lodd=lodd, #ode_method='RK45',
                        ode_atol=1e-10, 
                        ode_rtol=1e-10, 
                        dhfact=-1e-12)
        m_list, r_list, c_list = np.zeros(len(self.p)), np.zeros(len(self.p)), np.zeros(len(self.p))
        k_vars, h_vars = {} , {}
        for l in leven:
            k_vars['k' + str(l)] = np.zeros(len(self.p))
            h_vars['h' + str(l)] = np.zeros(len(self.p))
        j_vars = {}
        for l in lodd:
            j_vars['j' + str(l)] = np.zeros(len(self.p))
        for i, pc in enumerate(self.p):
            m, r, c, k, h, j = this_tov.solve(pc)
            r *= 1./1e3
            m *= 1./uts.constant['MRSUN_SI'][0]
            m_list[i], r_list[i], c_list[i] = m, r, c
            for l in leven:
                k_vars['k' + str(l)][i] = k[l]
                h_vars['h' + str(l)][i] = h[l]
            for l in lodd:
                j_vars['j' + str(l)][i] = j[l]
        # Save the data to a file
        if filename is None:
            filename = os.path.join(self.path, 'Love_data.txt')
        else:
            filename = os.path.join(self.path, filename)
        header = ('Pressure Mass Radius Compactness ' +
          ' '.join(f'k{l}' for l in leven) + ' ' +
          ' '.join(f'h{l}' for l in leven) + ' ' +
          ' '.join(f'j{l}' for l in lodd))
        np.savetxt(filename, np.column_stack((self.p, m_list, r_list, c_list, *k_vars.values(), *h_vars.values(), *j_vars.values())), header=header, delimiter='\t')
