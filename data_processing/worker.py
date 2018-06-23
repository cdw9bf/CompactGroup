"""
worker.py - Part of millennium-compact-groups package

Defines Worker object to handle clustering and analysis of individual
simulation chunk.

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
_PROG_NAME = 'worker.py'
_VERSION = 'v1.0'

# System utilities
import sys
import os
import time
import traceback
# Numerical utilities
import numpy as np
import pandas
from sklearn.cluster import MeanShift,DBSCAN
from scipy.spatial import cKDTree
# Other utilities
import compact_group
import cg_logger

def run_worker(w):
    """
    Downloads this simulation chunk, performs clustering and
    other analysis
    """
    try:
        start_time = time.time()
        w.logger.log("Working on snapnum: {0:02g}, Box {1:03g}, {2:03g}, {3:03g}".\
            format(w.snapnum,w.xbounds[0],w.ybounds[0],w.zbounds[0]))
        # If the member file and group file for this chunk already
        # exists and we are not overwriting, we're done!
        if ((os.path.exists(w.groupsfile) and os.path.exists(w.membersfile) and
            os.path.exists(w.all_groupsfile) and os.path.exists(w.all_membersfile)
            and not w.overwrite)):
            w.logger.log('Found {0} and {1}. Returning.'.format(w.groupsfile,w.membersfile))
            return
        #
        # If data exists and we're asked to overwrite it, or if data
        # doesn't exists, download it
        #
        if not os.path.exists(w.datafile):
            raise ValueError("Data file does not exist for snapnum: {0:02g}, Box {1:03g}, {2:03g}, {3:03g}".\
              format(w.snapnum,w.xbounds[0],w.ybounds[0],w.zbounds[0]))
        #
        # Read in the data
        #
        w.logger.log('Reading data.')
        w.read_data()
        w.logger.log('Done.')
        if len(w.data) == 0:
            # No galaxies found. We're done!
            w.logger.log('No galaxies found. Returning.')
            return
        #
        # Perform clustering
        #
        if (w.cluster or not os.path.exists(w.clusterfile)):
            w.logger.log('Performing clustering.')
            if w.use_dbscan:
                w.logger.log('Using DBSCAN.')
                w.dbscan()
            else:
                w.logger.log('Using MeanShift.')
                w.mean_shift()
            w.logger.log('Done.')
        #
        # Read cluster results
        #
        w.logger.log('Reading cluster results.')
        w.read_cluster()
        w.logger.log('Done.')
        # num-1 here accounts for the "-1" ungrouped cluster
        w.logger.log('Found {0} clusters.'.format(w.num_clusters-1))
        #
        # Measure the discovered groups' properties
        #
        w.logger.log('Analyzing groups.')
        w.analyze_groups()
        w.logger.log('Done.')
        #
        # Filter groups 
        #
        w.logger.log('Filtering groups.')
        w.filter_groups()
        w.logger.log('Done.')
        #
        # Save the group and member statistics
        #
        w.logger.log('Saving group and member statistics.')
        w.save()
        w.logger.log('Done.')
        # calculate run-time
        time_diff = time.time() - start_time
        hours = int(time_diff/3600.)
        mins = int((time_diff - hours*3600.)/60.)
        secs = time_diff - hours*3600. - mins*60.
        w.logger.log("Runtime: {0}h {1}m {2:.2f}s".format(hours,mins,secs))
    except Exception as e:
        w.logger.log("Caught exception in snapnum: {0:02g}, Box {1:03g}, {2:03g}, {3:03g}".\
            format(w.snapnum,w.xbounds[0],w.ybounds[0],w.zbounds[0]))
        w.logger.log(traceback.format_exc())
        raise e

class Worker:
    """
    Object to handle the organization of a single chunk of the
    simulation analysis
    """
    def __init__(self,snapnum,xbounds,ybounds,zbounds,
                 cluster=False,
                 use_dbscan=False,neighborhood=0.05,bandwidth=0.1,
                 min_members=3,dwarf_limit=0.50,mass_resolution=0.01,crit_velocity=1000.,
                 annular_radius=1.,max_annular_mass_ratio=1.e-4,min_secondtwo_mass_ratio=0.1,
                 datadir='data',outdir='results',overwrite=False,
                 verbose=False,nolog=False):
        #
        # Function arguments
        #
        self.snapnum = snapnum
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.zbounds = zbounds
        self.cluster = cluster
        self.use_dbscan = use_dbscan
        self.neighborhood = neighborhood
        self.bandwidth = bandwidth
        self.min_members = min_members
        self.dwarf_limit = dwarf_limit
        self.crit_velocity = crit_velocity
        self.annular_radius = annular_radius
        self.max_annular_mass_ratio = max_annular_mass_ratio
        self.min_secondtwo_mass_ratio = min_secondtwo_mass_ratio
        self.overwrite = overwrite
        self.mass_resolution = mass_resolution
        #
        # Set directories and files
        #
        my_string = "{0:02g}_{1:03g}_{2:03g}_{3:03g}".\
          format(self.snapnum,self.xbounds[0],self.ybounds[0],self.zbounds[0])
        directory = os.path.join(outdir,"snapnum_{0:02g}".format(self.snapnum))
        self.datafile = os.path.join(datadir,'snapnum_{0:02g}'.format(self.snapnum),
                                     'data_{0}.csv'.format(my_string))
        self.clusterfile = os.path.join(directory,'cluster','cluster_{0}.csv'.format(my_string))
        self.membersfile = os.path.join(directory,'members','members_{0}.csv'.format(my_string))
        self.groupsfile = os.path.join(directory,'groups','groups_{0}.csv'.format(my_string))
        self.all_membersfile = os.path.join(directory,'members','all_members_{0}.csv'.format(my_string))
        self.all_groupsfile = os.path.join(directory,'groups','all_groups_{0}.csv'.format(my_string))
        #
        # Setup log file
        #
        logfile = os.path.join(directory,'log_{0}_{1}.txt'.\
                               format(my_string,time.strftime('%Y%m%d%H%M%S')))
        self.logger = cg_logger.Logger(logfile, nolog=nolog, verbose=verbose)
        #
        # Other attributes we will fill
        #
        self.data = None
        self.num_clusters = 0
        self.groups = []
        self.good_groups = []
        self.galaxy_coordinates = None
    
    def read_data(self):
        """
        Read the data from the file
        """
        if not os.path.exists(self.datafile):
            raise IOError("{0} not found!".format(self.datafile))
        self.data = pandas.read_csv(self.datafile,header=0,comment='#',
                                    skip_blank_lines=True,index_col=0)
        #self.data = self.data[self.data['stellarMass'] >= 1e-4]
        self.galaxy_coordinates = np.array(list(zip(self.data['x'],
                                                    self.data['y'],
                                                    self.data['z'])))

    def dbscan(self):
        """
        Use DBSCAN to perform clustering in this chunk
        """
        # Set up DBSCAN
        db = DBSCAN(eps=self.neighborhood,
                    min_samples=self.min_members)
        # Perform the clustering
        db.fit(self.galaxy_coordinates)
        # save the labels
        np.savetxt(self.clusterfile,db.labels_,fmt='%d')

    def mean_shift(self):
        """
        Use MeanShift to perform clustering in this chunk
        """
        # Set up MeanShift
        ms = MeanShift(bandwidth=self.bandwidth,
                       min_bin_freq=self.min_members,
                       cluster_all=False)
        # Perform the clustering
        ms.fit(self.galaxy_coordinates)
        # save the labels
        np.savetxt(self.clusterfile,ms.labels_,fmt='%d')

    def read_cluster(self):
        """
        Read the saved clustering information, add column to data
        """
        labels = np.loadtxt(self.clusterfile,dtype=int)
        self.data['cluster'] = labels
        self.num_clusters = len(np.unique(labels))

    def analyze_groups(self):
        """
        Create CompactGroup object for each discovered group, measure
        their properties
        """
        labels_unique = np.unique(self.data['cluster'])
        #
        # Build a KD-Tree to perform neighborhood analyses
        # 
        # leafsize=15 gives best timing results from
        # https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/
        kdTree = cKDTree(self.galaxy_coordinates, leafsize=15)
        for l_ind,label in enumerate(labels_unique):
            # skip label -1 (ungrouped)
            if label == -1:
                continue
            #
            # Perform some basic calculations
            #
            # update label so it is unique to this snapnum+chunk
            mylabel = '{0:02g}{1:03g}{2:03g}{3:03g}{4:05g}'.\
              format(self.snapnum,self.xbounds[0],self.ybounds[0],self.zbounds[0],l_ind)
            # get all members of this group
            members = self.data.loc[self.data.cluster == label].copy()
            # create object
            cg = compact_group.CompactGroup(mylabel, members)
            self.logger.log('{0} has {1} members.'.format(cg.label,len(cg.members)))
            # identify dwarf galaxies
            cg.find_dwarfs(self.dwarf_limit, self.mass_resolution)
            self.logger.log('{0} has {1} dwarf galaxies.'.format(cg.label,np.sum(cg.members['is_dwarf'])))
            # calculate median velocity
            cg.calc_median_velocity()
            self.logger.log('{0} median velocity: {1} km/s.'.format(cg.label,cg.median_vel))
            # identify high-velocity "fly-by"s
            cg.find_flybys(self.crit_velocity)
            self.logger.log('{0} has {1} fly-by galaxies.'.format(cg.label,np.sum(cg.members['is_flyby'])))
            # calculate mediod of group
            cg.calc_mediod()
            self.logger.log('{0} mediod: {1}.'.format(cg.label,cg.mediod))
            # calculate radius of group
            cg.calc_radius()
            self.logger.log('{0} radius: {1}.'.format(cg.label,cg.radius))
            #calculate average virial mass
            cg.calc_avg_mvir()
            self.logger.log('{0} avg_mvir: {1}.'.format(cg.label,cg.avg_mvir))
            #calculate average stellar mass
            cg.calc_avg_stellarmass()
            self.logger.log('{0} avg_stellarmass: {1}.'.format(cg.label,cg.avg_stellarmass))
            #
            # Query KD-Tree for number of neighbors within radius
            # of this group's center
            #
            neighbors_ind = kdTree.query_ball_point(cg.mediod,
                                                    self.annular_radius)
            #
            # Exclude galaxies from neighbors list if they are
            # members of the group
            #
            neighbors_ind = [i for i in neighbors_ind
                            if self.data.index[i] not in cg.members.index.tolist()]
            # Save neighbors
            cg.neighbors = self.data.iloc[neighbors_ind]
            self.logger.log('{0} has {1} neighbors.'.format(cg.label,len(cg.neighbors)))
            #
            # Calculate the ratio of mass in an annulus outside of
            # the compact group and the total mass within a radius
            #
            cg.calc_annular_mass_ratio(self.annular_radius)
            self.logger.log('{0} annular_mass_ratio: {1}.'.format(cg.label,cg.annular_mass_ratio))
            #
            # Calculate the ratio of virial masses of the second two
            # largest members to the largest member
            #
            cg.calc_secondtwo_mass_ratio()
            self.logger.log('{0} secondtwo_mass_ratio: {1}.'.format(cg.label,cg.annular_mass_ratio))
            # add it to the list of groups in this chunk
            self.groups.append(cg)

    def filter_groups(self):
        """
        Filter out bad groups and save only good groups
        """
        mins = np.array([self.xbounds[0],self.ybounds[0],self.zbounds[0]])
        maxs = np.array([self.xbounds[1],self.ybounds[1],self.zbounds[1]])
        for cg in self.groups:
            # If the group has any non-dwarf galaxies, then it is eliminated
            #if np.sum((~cg.members['is_dwarf'])) > 0:
             #   self.logger.log('{0} eliminated because it contains non-dwarf members')
              #  continue
            # if the group has all fly by galaxies, then it is eliminated.
            if np.sum((~cg.members['is_flyby'])&(~cg.members['lower_limit'])) < self.min_members:
                self.logger.log('{0} eliminated because not enough members.'.format(cg.label))
                continue
            # If within annulus radius of the edge of the chunk,
            # it's bad
            if (np.any((cg.mediod-self.annular_radius) < mins) or
                np.any((cg.mediod+self.annular_radius) > maxs)):
                self.logger.log('{0} eliminated because too close to edge.'.format(cg.label))
                continue
            # If the annular mass to total mass ratio is more than
            # ax_annular_mass_ratio, it's bad
            if cg.annular_mass_ratio > self.max_annular_mass_ratio:
                self.logger.log('{0} eliminated because annular_mass_ratio too large.'.format(cg.label))
                continue
            # If the ratio of the virial mass of the 2nd+3rd most
            # massive members to that of the most massive member
            # is less than min_secondtwo_mass_ratio, it's bad
            if cg.secondtwo_mass_ratio < self.min_secondtwo_mass_ratio:
                self.logger.log('{0} eliminated because secondtwo_mass_ratio too small.'.format(cg.label))
                continue
            # If we made it this far, we have a "good" group
            self.logger.log('{0} is a good_group'.format(cg.label))
            self.good_groups.append(cg)

    def save(self):
        """
        Save group and member statistics to file
        """
        # Save group statistics
        filenames = [self.groupsfile,self.all_groupsfile]
        groups = [self.good_groups,self.groups]
        for filename,group in zip(filenames,groups):
            #          label     x          y          z          radius     vel,       num,     mvir,      stelMass,   near,   annular,     secondtwo
            fmt_row = "{0:>16}, {1:>9.5f}, {2:>9.5f}, {3:>9.5f}, {4:>9.5f}, {5:>12.5f}, {6:>11}, {7:>12.5f}, {8:>15.5f}, {9:>13}, {10:>18.5f}, {11:>20.5f}\n"
            fmt_hdr = "{0:>16}, {1:>9}, {2:>9}, {3:>9}, {4:>9}, {5:>12}, {6:>11}, {7:>12}, {8:>15}, {9:>13}, {10:>18}, {11:>20}\n"
            with open(filename,'w') as f:
                f.write(fmt_hdr.format("group_id","x","y","z","radius",
                                       "median_vel","num_members","avg_mvir",
                                       "avg_stellarMass","galaxies_near",
                                       "annular_mass_ratio",
                                       "secondtwo_mass_ratio"))
                for cg in group:
                    f.write(fmt_row.format(cg.label,cg.mediod[0],cg.mediod[1],
                                           cg.mediod[2],cg.radius,cg.median_vel,
                                           len(cg.members),cg.avg_mvir,
                                           cg.avg_stellarmass,len(cg.neighbors),
                                           cg.annular_mass_ratio,cg.secondtwo_mass_ratio))
        # Save member statistics
        filenames = [self.membersfile,self.all_membersfile]
        groups = [self.good_groups,self.groups]
        for filename,group in zip(filenames,groups):
            #          group,    mem,     x,         y,         z,         velX,      velY,      velZ,     vel,        r_mag,     mvir,       stelMass,   treeID,  is_dwarf, is_flyby
            fmt_row = "{0:>16}, {1:>15}, {2:>9.5f}, {3:>9.5f}, {4:>9.5f}, {5:>12.5f}, {6:>12.5f}, {7:>12.5f}, {8:>12.5f}, {9:>9.5f}, {10:>12.5f}, {11:>11.5f}, {12:>15}, {13:>8}, {14:>8}\n"
            fmt_hdr = "{0:>16}, {1:>15}, {2:>9}, {3:>9}, {4:>9}, {5:>12}, {6:>12}, {7:>12}, {8:>12}, {9:>9}, {10:>12}, {11:>11}, {12:>15}, {13:>8}, {14:>8}\n"
            with open(filename,'w') as f:
                f.write(fmt_hdr.format("group_id","member_id","x","y",
                                       "z","velX","velY","velZ","vel",
                                       "r_mag","mvir","stellarMass",
                                       "treeID","is_dwarf","is_flyby"))
                for cg in group:
                    for index,member in cg.members.iterrows():
                        f.write(fmt_row.format(cg.label,index,
                                               member['x'],member['y'],
                                               member['z'],member['velX'],
                                               member['velY'],member['velZ'],
                                               member['vel'],member['r_mag'],
                                               member['mvir'],member['stellarMass'],
                                               member['treeID'],
                                               member['is_dwarf'],member['is_flyby']))
