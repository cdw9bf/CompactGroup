"""
get_data.py - Part of millennium-compact-groups package

Download data for analysis. This is meant to be run first, before
main.py.

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
_PROG_NAME = 'get_data.py'
_VERSION = 'v1.0'

# System utilities
import os
import argparse
import time
# Numerical utilities
import numpy as np
# Other utilities
import itertools
# Classes for this project
import millennium_query


def main(username,password,
         snapnums=np.arange(67),size=20.,annular_radius=1.0,
         outdir='results',overwrite=False):
    """
    Download Millennium Simulation data for Compact Group analysis
    """
    start_time = time.time()
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    #
    # Set up output directories
    #
    for snapnum in snapnums:
        directory = os.path.join(outdir,"snapnum_{0:02g}".format(snapnum))
        if not os.path.isdir(directory):
            os.mkdir(directory)
    #
    # Get Millennium Simulation cookies
    #
    cookies = millennium_query.get_cookies(username, password)
    #
    # Set up simulation chunk boundaries
    #
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
    # Get the data
    #
    # Set up SQL query
    query = ("SELECT {0} FROM {1} WHERE snapnum={2} "
             "AND x BETWEEN {3} AND {4} "
             "AND y BETWEEN {5} AND {6} "
             "AND z BETWEEN {7} AND {8} "
             "AND stellarMass > 0 "
             "AND r_mag < 99")
    # Columns to download
    columns=('galaxyID,redshift,x,y,z,velX,velY,velZ,r_mag,mvir,'
             'stellarMass,type,treeID')
    # Table to query
    table = "Guo2010a.dbo.MRII"
    #
    # Loop over all snapnum and boundary combinations
    #
    total = int(len(snapnums)*len(boundaries)**3.)
    num = 0
    for snapnum,xbounds,ybounds,zbounds in \
      itertools.product(snapnums,boundaries,boundaries,boundaries):
        num = num + 1
        job_time = time.time()
        # set up datafile
        mystring = "{0:02g}_{1:03g}_{2:03g}_{3:03g}".\
          format(snapnum,xbounds[0],ybounds[0],zbounds[0])
        datafile = os.path.join(outdir,'snapnum_{0:02g}'.format(snapnum),
                                'data_{0}.csv'.format(mystring))
        # Check if file already exists
        if os.path.exists(datafile) and not overwrite:
            print("Found: ({0}/{1}) - {2}".format(num,total,mystring))
            continue
        # set up this query
        my_query = query.format(columns,table,snapnum,
                                xbounds[0],xbounds[1],
                                ybounds[0],ybounds[1],
                                zbounds[0],zbounds[1])
        # connect and perform query
        conn = millennium_query.MillenniumQuery(username, password,
                                                cookies=cookies,
                                                maxrec=1000000000)
        conn.query(my_query)
        conn.run()
        conn.wait()
        # Save the results
        conn.save(datafile)
        # Delete and close the connection
        conn.delete()
        conn.close()
        print("Downloaded: ({0}/{1}) - {2} - Runtime: {3:.2f}s".format(num,total,mystring,time.time()-job_time))
    #
    # Clean up
    #
    # calculate run-time
    time_diff = time.time() - start_time
    hours = int(time_diff/3600.)
    mins = int((time_diff - hours*3600.)/60.)
    secs = time_diff - hours*3600. - mins*60.
    print("Runtime: {0}h {1}m {2:.2f}s".format(hours,mins,secs))

#=====================================================================
# Command Line Arguments
#=====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Millennium Simulation data for Compact Group Analysis",
        prog=_PROG_NAME,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    # Required Parameters
    #
    parser.add_argument('--username',type=str,required=True,
                        help="Username for Millennium Simulation access.")
    parser.add_argument('--password',type=str,required=True,
                        help="Password for Millennium Simulation access")
    #
    # Simulation parameters
    #
    parser.add_argument('--snapnums',nargs="+",type=int,
                        default=np.arange(67),
                        help="snapnums to process. Default: All (0 to 67)")
    parser.add_argument('--size',type=int,
                        default=25,
                        help="Simulation chunk cube side length in Mpc/h. Default: 25")
    parser.add_argument('--annular_radius',type=float,default=1.0,
                        help=('Size (in Mpc/h) of outer annular radius '
                              'for annular mass ratio calculation. This is the overlap '
                              'in the simulation chunk boundaries. Default: 1.0'))
    #
    # Data output parameters
    #
    parser.add_argument('--outdir',type=str,default='data',
                        help="directory to save data. Default: data/")
    parser.add_argument('--overwrite',action='store_true',
                        help='Re-download data file if it already exists')
    #
    # Parse the arguments and send to main function
    #
    args = parser.parse_args()
    main(args.username,args.password,
         snapnums=args.snapnums,size=args.size,annular_radius=args.annular_radius,
         outdir=args.outdir,overwrite=args.overwrite)
