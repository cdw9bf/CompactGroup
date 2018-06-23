"""
main.py - Part of millennium-compact-groups package

Use a clustering algorithm to find compact groups in the Millennium
simulation.

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
_PROG_NAME = 'main.py'
_VERSION = 'v1.0'

# System utilities
import os
import argparse
import time
import traceback
# Numerical utilities
import numpy as np
# Other utilities
import multiprocessing as mp
import ipyparallel as ipp
import itertools
# Classes for this project
import cg_logger
import code_to_submit.data_processing.worker


def main(snapnums=np.arange(67),size=20.,
         cluster=False,
         use_dbscan=False,neighborhood=0.05,bandwidth=0.1,
         min_members=3,dwarf_limit=0.50,mass_resolution=0.01,crit_velocity=1000.,
         annular_radius=1.,max_annular_mass_ratio=1.e-4,min_secondtwo_mass_ratio=0.1,
         num_cpus=1,profile=None,
         datadir='data',outdir='results',overwrite=False,
         verbose=False,nolog=False,test=False):
    """
    Set up workers to perform clustering and calculate group and
    member statistics  
    """
    start_time = time.time()
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    #
    # Handle test case
    #
    if test:
        snapnums = np.array([50])
        size = 20.
    #
    # Open main log file
    #
    logfile = os.path.join(outdir,'log_{0}.txt'.format(time.strftime('%Y%m%d%H%M%S')))
    logger = cg_logger.Logger(logfile, nolog=nolog, verbose=verbose)
    logger.log("Using the following parameters:")
    logger.log("snapnums: {0}".format(snapnums))
    logger.log("size: {0}".format(size))
    logger.log("cluster: {0}".format(cluster))
    logger.log("use_dbscan: {0}".format(use_dbscan))
    logger.log("neighborhood: {0}".format(neighborhood))
    logger.log("bandwidth: {0}".format(bandwidth))
    logger.log("min_members: {0}".format(min_members))
    logger.log("dwarf_limit: {0}".format(dwarf_limit))
    logger.log("mass_resolution: {0}".format(mass_resolution))
    logger.log("crit_velocity: {0}".format(crit_velocity))
    logger.log("annular_radius: {0}".format(annular_radius))
    logger.log("max_annular_mass_ratio: {0}".format(max_annular_mass_ratio))
    logger.log("min_secondtwo_mass_ratio: {0}".format(min_secondtwo_mass_ratio))
    logger.log("num_cpus: {0}".format(num_cpus))
    logger.log("profile: {0}".format(profile))
    logger.log("datadir: {0}".format(datadir))
    logger.log("outdir: {0}".format(outdir))
    logger.log("overwrite: {0}".format(outdir))
    logger.log("verbose: {0}".format(verbose))
    logger.log("test: {0}".format(test))
    #
    # Set up output directories
    #
    for snapnum in snapnums:
        directory = os.path.join(outdir,"snapnum_{0:02g}".\
                                 format(snapnum))
        if not os.path.isdir(directory):
            os.mkdir(directory)
            logger.log('Created {0}'.format(directory))
        cluster_directory = os.path.join(directory,'cluster')
        if not os.path.isdir(cluster_directory):
            os.mkdir(cluster_directory)
            logger.log('Created {0}'.format(cluster_directory))
        members_directory = os.path.join(directory,'members')
        if not os.path.isdir(members_directory):
            os.mkdir(members_directory)
            logger.log('Created {0}'.format(members_directory))
        groups_directory = os.path.join(directory,'groups')
        if not os.path.isdir(groups_directory):
            os.mkdir(groups_directory)
            logger.log('Created {0}'.format(groups_directory))
    #
    # Set up simulation chunk boundaries
    #
    if test:
        mins = np.array([0])
    else:
        mins = np.arange(0,100,size)
    maxs = mins + size
    #
    # adjust mins and maxs to overlap by annular_radius, but do not
    # go beyond simulation boundaries
    #
    mins = mins - annular_radius
    mins[mins < 0.] = 0.
    maxs = maxs + annular_radius
    maxs[maxs > 100.] = 100.
    boundaries = list(zip(mins,maxs))
    #
    # Set up worker pool
    #
    jobs = []
    for snapnum,xbounds,ybounds,zbounds in \
      itertools.product(snapnums,boundaries,boundaries,boundaries):
        # Set-up a new Worker
        job = code_to_submit.data_processing.worker.Worker(snapnum, xbounds, ybounds, zbounds,
                                                           cluster=cluster,
                                                           use_dbscan=use_dbscan, neighborhood=neighborhood, bandwidth=bandwidth,
                                                           min_members=min_members, dwarf_limit=dwarf_limit, mass_resolution=mass_resolution,
                                                           crit_velocity=crit_velocity, annular_radius=annular_radius,
                                                           max_annular_mass_ratio=max_annular_mass_ratio,
                                                           min_secondtwo_mass_ratio=min_secondtwo_mass_ratio,
                                                           datadir=datadir, outdir=outdir, overwrite=overwrite,
                                                           verbose=verbose, nolog=nolog)
        # Append to list of worker arguments
        jobs.append(job)
        logger.log('Created worker for snapnum: {0:02g}, xmin: {1:03g}, ymin: {2:03g}, zmin: {3:03g}'.\
                   format(snapnum,xbounds[0],ybounds[0],zbounds[0]))
    logger.log("Found {0} jobs".format(len(jobs)))
    #
    # Set up IPython.parallel
    #
    if profile is not None:
        logger.log("Using IPython.parallel")
        engines = ipp.Client(profile=profile,block=False)
        logger.log("Found {0} IPython.parallel engines".\
              format(len(engines)))
        balancer = engines.load_balanced_view()
        balancer.block = False
        results = balancer.map(code_to_submit.data_processing.worker.run_worker, jobs)
        try:
            results.get()
        except Exception as e:
            logger.log("Caught exception")
            logger.log(traceback.format_exc())
    #
    # Set up multiprocessing
    #
    elif num_cpus > 1:
        logger.log("Using multiprocessing with {0} cpus".format(num_cpus))
        pool = mp.Pool(num_cpus)
        results = pool.map_async(code_to_submit.data_processing.worker.run_worker, jobs)
        pool.close()
        pool.join()
    #
    # One job at a time
    #
    else:
        logger.log("Not using parallel processing.")
        for job in jobs:
            code_to_submit.data_processing.worker.run_worker(job)
    logger.log("All jobs done.")
    #
    # Clean up
    #
    # calculate run-time
    time_diff = time.time() - start_time
    hours = int(time_diff/3600.)
    mins = int((time_diff - hours*3600.)/60.)
    secs = time_diff - hours*3600. - mins*60.
    logger.log("Runtime: {0}h {1}m {2:.2f}s".format(hours,mins,secs))

#=====================================================================
# Command Line Arguments
#=====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find Compact Groups in Full Millenium Simulation",
        prog=_PROG_NAME,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    # Simulation parameters
    #
    parser.add_argument('--snapnums',nargs="+",type=int,
                        default=np.arange(67),
                        help="snapnums to process. Default: All (0 to 67)")
    parser.add_argument('--size',type=int,
                        default=20,
                        help="Simulation chunk cube side length in Mpc/h. Default: 25")
    #
    # Clustering parameters
    #
    parser.add_argument('--cluster',action='store_true',
                        help='Re-do clustering even if clustering output already exists.')
    parser.add_argument('--use_dbscan',action='store_true',
                        help='If set, use DBSCAN for clustering. Default: MeanShift')
    parser.add_argument('--neighborhood',type=float,default=0.05,
                        help='Neighborhood parameter for DBSCAN. Default 0.05')
    parser.add_argument('--bandwidth',type=float,default=0.1,
                        help='Bandwidth parameter for MeanShift. Default 0.1')
    #
    # Filter parameters
    #
    parser.add_argument('--min_members',type=int,default=3,
                        help='Minimum members to be considered a group. Default: 3')
    parser.add_argument('--dwarf_limit',type=float,default=0.05,
                        help=('Stellar mass limit for dwarf galaxies in '
                               '10^10 Msun/h. Default: 0.05'))
    parser.add_argument('--crit_velocity',type=float,default=1000.0,
                        help=('Velocity difference (km/s) between a '
                              'galaxy and median group velocity to '
                              'exclude (i.e. high-velocity fly-bys). '
                              'Default: 1000.0'))
    parser.add_argument('--annular_radius',type=float,default=1.0,
                        help=('Size (in Mpc/h) of outer annular radius '
                              'for annular mass ratio calculation. Default: 1.0'))
    parser.add_argument('--max_annular_mass_ratio',type=float,default=1.e-4,
                        help=('Maximum allowed value for the ratio of mass '
                              'in annulus to total mass. Default: 1.e-4'))
    parser.add_argument('--min_secondtwo_mass_ratio',type=float,default=0.1,
                        help=('Minimum allowed value for the ratio of mass '
                              'of the second two most massive galaxies to '
                              ' the most massive galaxy. Default: 0.1'))
    parser.add_argument('--mass_resolution',type=float,default=0.01,
                        help='Lower mass limit for the simulation measured with mvir. Default:0.01')
    #
    # Multiprocessing parameters
    #
    parser.add_argument('--num_cpus',type=int,default=1,
                        help=("Number of cores to use with "
                              "multiprocessing (not "
                              "IPython.parallel). Default: 1"))
    parser.add_argument('--profile',type=str,default=None,
                        help=("IPython profile if running on computing "
                              "cluster using IPython.parallel. "
                              "Default: None (use multiprocessing "
                              "on single machine)"))
    #
    # Data parameters
    #
    parser.add_argument('--outdir',type=str,default='results',
                        help="directory to save results. Default: results/")
    parser.add_argument('--datadir',type=str,default='data',
                        help="directory where data lives. Default: data/")
    parser.add_argument('--overwrite',action='store_true',
                        help='Re-do analysis if member file and group file exists.')
    #
    # Other
    #
    parser.add_argument('--verbose',action='store_true',
                        help='Output messages along the way.')
    parser.add_argument('--nolog',action='store_true',
                        help="Do not save log files")
    parser.add_argument('--test',action='store_true',
                        help="Run a test on one chunk. (snapnum=50-60, box=0,0,0, size=100)")
    #
    # Parse the arguments and send to main function
    #
    args = parser.parse_args()
    main(snapnums=args.snapnums,size=args.size,
         cluster=args.cluster,
         use_dbscan=args.use_dbscan,neighborhood=args.neighborhood,bandwidth=args.bandwidth,
         min_members=args.min_members,dwarf_limit=args.dwarf_limit,mass_resolution=args.mass_resolution,
         crit_velocity=args.crit_velocity,annular_radius=args.annular_radius,
         max_annular_mass_ratio=args.max_annular_mass_ratio,min_secondtwo_mass_ratio=args.min_secondtwo_mass_ratio,
         num_cpus=args.num_cpus,profile=args.profile,
         datadir=args.datadir,outdir=args.outdir,overwrite=args.overwrite,
         verbose=args.verbose,nolog=args.nolog,test=args.test)
