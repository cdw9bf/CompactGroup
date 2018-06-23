
"""
compact_group.py - Part of millennium-compact-groups package

Defines CompactGroup object to handle information about a single
compact group.

Copyright(C) 2016 by
Trey Wenger; tvwenger@gmail.com
Chris Wiens; cdw9bf@virginia.edu
Kelsey Johnson; kej7a@virginia.edu

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

14 Mar 2016 - TVW Finalized version 1.0
"""
_PACK_NAME = 'millennium-compact-groups'
_PROG_NAME = 'compact_group.py'
_VERSION = 'v1.0'

# System utilities
import sys
import os
# Numerical utilities
import numpy as np
import pandas

class CompactGroup:
    """
    Compact Group Object
    """
    def __init__(self, label, members):
        """
        Initialize ComactGroup Object
        """
        self.label = label
        self.members = members
        self.median_vel = 0.0
        self.mediod = None
        self.radius = 0.0
        self.avg_mvir = 0.0
        self.avg_stellarmass = 0.0
        self.num_nearby_galaxies = 0
        self.neighbors = []
        self.annular_mass_ratio = 0.0
        self.secondtwo_mass_ratio = 0.0

    def find_dwarfs(self,dwarf_limit, mass_resolution):
        """
        Find galaxies that have a stellar mass less than dwarf_limit
        """
        # add a is_dwarf column to members
        self.members['is_dwarf'] = np.zeros(len(self.members),dtype=bool)
        self.members['lower_limit'] = np.zeros(len(self.members),dtype=bool)
        # assign dwarfs
        indx = self.members['mvir'] < mass_resolution
        ind = self.members['stellarMass'] < dwarf_limit
        self.members.ix[ind,'is_dwarf'] = True
        self.members.ix[indx,'lower_limit'] = True

    def calc_median_velocity(self):
        """
        Calculate the median velocity of galaxies in this group
        """
        good = (~self.members['lower_limit'])
        vels = (self.members['velX']*self.members['velX'] +
                self.members['velY']*self.members['velY'] +
                self.members['velZ']*self.members['velZ'])**0.5
        # add a velocity2 column to members
        self.members['vel'] = vels
        self.median_vel = np.median(vels[good])
        
    def find_flybys(self,crit_velocity):
        """
        Find galaxies that are travelling crit_velocity faster or
        slower than median velocity of group. These are "fly-bys"
        """
        # add a is_flyby column to members
        self.members['is_flyby'] = np.zeros(len(self.members),dtype=bool)
        # assign flybys
        ind = np.abs(self.members['vel'] - self.median_vel) > crit_velocity
        self.members.ix[ind,'is_flyby'] = True
        
    def calc_mediod(self):
        """
        Calculate the mediod center of this group, excluding
        dwarfs and flybys
        """
        good = ((~self.members['is_flyby'])&(~self.members['lower_limit']))
        x_med = np.median(self.members['x'][good])
        y_med = np.median(self.members['y'][good])
        z_med = np.median(self.members['z'][good])
        self.mediod = np.array([x_med,y_med,z_med])

    def calc_radius(self):
        """
        Calculate the radius of this group, defined as the
        maximum galaxy distance from the mediod, excluding
        dwarfs and flybys
        """
        good = ((~self.members['is_flyby'])&(~self.members['lower_limit']))
        xdist = self.members['x'][good]-self.mediod[0]
        ydist = self.members['y'][good]-self.mediod[1]
        zdist = self.members['z'][good]-self.mediod[2]
        dists = (xdist*xdist + ydist*ydist + zdist*zdist)**0.5
        self.radius = np.max(dists)

    def calc_avg_mvir(self):
        """
        Calculate the average virial mass of galaxies in this group
        excluding dwafs and flybys
        """
        good = ((~self.members['is_flyby'])&(~self.members['lower_limit']))
        if np.sum(good) == 0:
            self.avg_mvir = np.nan
        else:
            self.avg_mvir = np.mean(self.members['mvir'][good])

    def calc_avg_stellarmass(self):
        """
        Calculate the average stellar mass of galaxies in this group
        excluding dwafs and flybys
        """
        good = ((~self.members['is_flyby'])&(~self.members['lower_limit']))
        if np.sum(good) == 0:
            self.avg_stellarmass = np.nan
        else:
            self.avg_stellarmass = np.mean(self.members['stellarMass'][good])

    def calc_annular_mass_ratio(self,radius):
        """
        Calculate the virial mass ratio
        of neighboring galaxies within the surrounding annulus to the
        total virial mass of all galaxies within the sphere
        """
        # mass of cluster
        sphere_mass = np.sum(self.members['mvir'])
        sphere_mass = sphere_mass / (4.*np.pi/3. * self.radius**3.)
        # mass in annulus
        annulus_mass = np.sum(self.neighbors['mvir'])
        annulus_mass = annulus_mass/(4.*np.pi/3. * (radius**3. - self.radius**3.))
        self.annular_mass_ratio = annulus_mass/sphere_mass

    def calc_secondtwo_mass_ratio(self):
        """
        Calculate the ratio of the virial masses of the second largest
        members to the virial mass of the largest member
        """
        sorted_masses = np.sort(self.members['mvir'])
        self.secondtwo_mass_ratio = (sorted_masses[-2]+sorted_masses[-3])/sorted_masses[-1]
