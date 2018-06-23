"""
plots.py - Part of millennium-compact-groups package

Generate plots from package output.

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

24 Mar 2016 - TVW Finalized version 1.0
"""
_PACK_NAME = 'millennium-compact-groups'
_PROG_NAME = 'plots.py'
_VERSION = 'v1.1'

# System utilities
import os
import argparse
# Numerical utilities
import numpy as np
import pandas
# Other utilities
import itertools
import glob
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
 
from matplotlib.ticker import FormatStrFormatter
from astropy.cosmology import WMAP7

mpl.rc_file('/data/compactgroups/Evolution/matplotlibrc')

def main(datadir='/data',resultsdirs=['results'],plotsdir='plots',
         labels=['results'],
         counts_file='galaxy_counts.txt',snapnum_file='snapnum_redshift.csv',
         dwarf_limit=0.05,plotlabel='',evolution=False):
    """
    Plot results of Millennium Simulation compact group analysis
    """
    if not os.path.isdir(datadir):
        raise ValueError("{0} not found!".format(datadir))
    for resultsdir in resultsdirs:
        if not os.path.isdir(resultsdir):
            raise ValueError("{0} not found!".format(resultsdir))
    if not os.path.isdir(plotsdir):
        os.mkdir(plotsdir)
    if not os.path.exists(snapnum_file):
        raise ValueError("{0} not found!".format(snapnum_file))
    #
    # Read snapnum file to convert from snapnum to redshift
    #
    snap_to_z = pandas.read_csv(snapnum_file,header=0,comment='#',
                                skip_blank_lines=True,index_col=0,
                                skipinitialspace=True)
    #
    # Determine number of snapnums we have in datadir
    #
    if evolution:
    	data_snapnum_dirs = np.arange(64)
    else:
    	data_snapnum_dirs = np.sort(glob.glob(os.path.join(datadir,'snapnum_*')))
    print("Found {0} snapnum directories in {1}.".format(len(data_snapnum_dirs),datadir))
    if len(data_snapnum_dirs) == 0 and not os.path.exists(counts_file):
        return

    #
    # Count the number and standard devation of non-dwarf galaxies in raw data files
    #
    if os.path.exists(counts_file):
        print("Found {0}".format(counts_file))
    else:
    	generate_snapnum_counts(datadir, data_snapnum_dirs, data_file, dwarf_limit, counts_file)
    

    #
    # Read the galaxy counts file
    #
    galaxy_counts = pandas.read_csv(counts_file,header=0,
                                    comment='#',skip_blank_lines=True,
                                    index_col=0,skipinitialspace=True)

    #
    # Get results from each resultsdir
    #
    if evolution:
    	num_members, num_members_std = data_read_in_evolution(resultsdirs, data_snapnum_dirs)
    	
    else:
    	members, groups, num_non_dwarf_members, non_dwarf_members, num_members_std, num_non_dwarf_members_std, num_groups_std = data_read_in(
    			resultsdirs, data_snapnum_dirs, datadir)

    #
    # Convert snapnums to redshifts and to lookback times
    #
    snapnums = [snapnum for snapnum in range(len(data_snapnum_dirs))]
    redshifts = np.array([snap_to_z['Z'][snapnum] for snapnum in snapnums])
    lbtimes = WMAP7.lookback_time(redshifts).value
    plot_redshifts = [0.0,0.1,0.2,0.3,0.5,1,1.5,2,3,5,10]
    plot_lookback_times = WMAP7.lookback_time(plot_redshifts).value
    #
    colors = ['#d73027','#fc8d59','#fee090']
    symbols = ['o','s','^','*','p']
    
    #
    # Make Evolution Plots
    #
    
    if evolution:
    	plot_evolution(plotlabel, galaxy_counts, num_members, num_members_std,
    					 resultsdirs, colors, symbols, lbtimes, plot_lookback_times,
    					 plot_redshifts, plotsdir,labels)
    	return
    	
    	
    	
    #
    # Plot total number of compact groups vs. redshift
    #
    figname = os.path.join(plotsdir,'{0}_num_groups.eps'.format(plotlabel))
    num_cgs = np.array([[len(df) for df in groups[result_ind]] for result_ind in range(len(resultsdirs))])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    for ind,(num_cg,num_cg_std) in enumerate(zip(num_cgs,num_groups_std)):
        ax.plot(lbtimes,num_cg,color=colors[ind],marker=symbols[ind],label=labels[ind])
        ax.fill_between(lbtimes,num_cg-num_cg_std,num_cg+num_cg_std,
                        facecolor=colors[ind],edgecolor="none",alpha=0.6)
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel('Number of Compact Groups')
    ax.set_yscale('log')
    ax.set_xlim(0,WMAP7.age(0).to('Gyr').value)
    ax.set_ylim(1.e2,1.e5)
    ax.legend(loc='best',fontsize=12,numpoints=1)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(plot_lookback_times)
    ax2.set_xticklabels(plot_redshifts)
    ax2.set_xlabel('Redshift')
    ax2.grid(False)
    plt.savefig(figname)
    plt.close()
    #
    # Plot total number of galaxies in compact groups vs. redshift
    #
    figname = os.path.join(plotsdir,'{0}_num_members.eps'.format(plotlabel))
    num_galaxies = np.array([[len(df) for df in non_dwarf_members[result_ind]] for result_ind in range(len(resultsdirs))])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    for ind,(num_galaxy,num_galaxy_std) in enumerate(zip(num_galaxies,num_non_dwarf_members_std)):
        ax.plot(lbtimes,num_galaxy,color=colors[ind],marker=symbols[ind],label=labels[ind])
        ax.fill_between(lbtimes,num_galaxy-num_galaxy_std,num_galaxy+num_galaxy_std,
                        facecolor=colors[ind],edgecolor='none',alpha=0.6)
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel('Number of non-dwarf Compact Group Members')
    ax.set_yscale('log')
    ax.legend(loc='best',fontsize=12,numpoints=1)
    ax.set_xlim(0,WMAP7.age(0).to('Gyr').value)
    ax.set_ylim(1.e3,5.e5)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(plot_lookback_times)
    ax2.set_xticklabels(plot_redshifts)
    ax2.set_xlabel('Redshift')
    ax2.grid(False)
    plt.savefig(figname)
    plt.close()
    #
    # Plot fractional number of galaxies in compact groups vs. redshift
    #
    figname = os.path.join(plotsdir,'{0}_frac_galaxies.eps'.format(plotlabel))
    frac_galaxies = np.array([[len(df)/galaxy_counts['num_galaxies'][snapnum] for snapnum,df in enumerate(non_dwarf_members[result_ind])] for result_ind in range(len(resultsdirs))])
    frac_galaxies_std = frac_galaxies * np.sqrt(np.array([[(num_galaxy_std/len(df))**2. + (galaxy_counts['std_galaxies'][snapnum]/galaxy_counts['num_galaxies'][snapnum])**2.
                                                           for snapnum,(df,num_galaxy_std) in enumerate(zip(non_dwarf_members[result_ind],num_non_dwarf_members_std[result_ind]))] for result_ind in range(len(resultsdirs))]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    for ind,(frac_galaxy,frac_galaxy_std) in enumerate(zip(frac_galaxies,frac_galaxies_std)):
        ax.plot(lbtimes,100.*frac_galaxy,color=colors[ind],marker=symbols[ind],label=labels[ind])
        ax.fill_between(lbtimes,100.*(frac_galaxy-frac_galaxy_std),100.*(frac_galaxy+frac_galaxy_std),
                        facecolor=colors[ind],edgecolor='none',alpha=0.6)
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel('Percent of non-dwarf Galaxies in Compact Groups')
    #ax.set_yscale('log')
    ax.legend(loc='best',fontsize=12,numpoints=1)
    ax.set_xlim(0,WMAP7.age(0).to('Gyr').value)
    ax.set_ylim(0.,1.5)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(plot_lookback_times)
    ax2.set_xticklabels(plot_redshifts)
    ax2.set_xlabel('Redshift')
    ax2.grid(False)
    plt.savefig(figname)
    plt.close()
    return
    #
    # Now with redshift
    #
    #
    # Plot total number of compact groups vs. redshift
    #
    num_cgs = [len(df) for df in groups]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    ax.plot(redshifts,num_cgs,'k-')
    ax.plot(redshifts,num_cgs,'ko')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Number of Compact Groups')
    ax.set_yscale('log')
    ax.set_xlim(0,12)
    ax.set_ylim(1.e2,1.e5)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(13))
    ax2.set_xticklabels(WMAP7.lookback_time(np.arange(13)).value)
    ax2.set_xlabel('Lookback time (Gyr)')
    ax2.grid(False)
    plt.savefig('num_groups_z.eps')
    plt.close()
    #
    # Plot total number of galaxies in compact groups vs. redshift
    #
    num_galaxies = [len(df) for df in non_dwarf_members]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    ax.plot(redshifts,num_galaxies,'k-')
    ax.plot(redshifts,num_galaxies,'ko')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Number of non-dwarf Compact Group Members')
    ax.set_yscale('log')
    ax.set_xlim(0,12)
    ax.set_ylim(1.e2,1.e5)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(13))
    ax2.set_xticklabels(WMAP7.lookback_time(np.arange(13)).value)
    ax2.set_xlabel('Lookback time (Gyr)')
    ax2.grid(False)
    plt.savefig('num_members_z.eps')
    plt.close()
    #
    # Plot fractional number of galaxies in compact groups vs. redshift
    #
    frac_galaxies = [len(df)/galaxy_counts['num_galaxies'][snapnum] for snapnum,df in enumerate(non_dwarf_members)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    ax.plot(redshifts,frac_galaxies,'k-')
    ax.plot(redshifts,frac_galaxies,'ko')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Fracion of non-dwarf Galaxies in Compact Groups')
    ax.set_xlim(0,12)
    ax.set_ylim(1.e-4,1.e-2)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(13))
    ax2.set_xticklabels(WMAP7.lookback_time(np.arange(13)).value)
    ax2.set_xlabel('Lookback time (Gyr)')
    ax2.grid(False)
    plt.savefig('frac_galaxies_z.eps')
    plt.close()
    return




def generate_snapnum_counts(datadir, data_snapnum_dir, data_file, dwarf_limit, counts_file):
    with open(counts_file,'w') as f:
        f.write('snapnum, num_galaxies, std_galaxies\n')
        for snapnum,snapnum_dir in enumerate(data_snapnum_dirs):
            # Find the data files in this directory
            data_files = np.sort(glob.glob(os.path.join(snapnum_dir,'data_*')))
            print("Found {0} data files in {1}".format(len(data_files),snapnum_dir))
            galaxy_counts = np.zeros(len(data_files))
            for ind,data_file in enumerate(data_files):
                # Read data file, count non-dwarf galaxies
                data = pandas.read_csv(data_file,header=0,comment='#',
                                           skip_blank_lines=True,index_col=0,
                                           skipinitialspace=True)
                try:
                    good = (data['stellarMass']>dwarf_limit)&(data['x']>=1)&(data['x']<=499)&(data['y']>=1)&(data['y'] <= 499)&(data['z']>=1)&(data['z'] <= 499)
                    galaxy_counts[ind] = np.sum(good)
                except KeyError as e:
                    print("Problem with {0}".format(data_file))
            # Save good galaxy count
            galaxy_count = np.sum(galaxy_counts)
            galaxy_std = np.std(galaxy_counts) * len(data_files)
            f.write('{0:7}, {1:10.2f}, {2:10.2f}\n'.format(snapnum,galaxy_count,galaxy_std))
    return





def data_read_in(resultsdirs,data_snapnum_dirs, datadir):
	
	
	#
    # Create pandas dataframe for each resultsdir and available snapnum
    #
	groups = [[pandas.DataFrame() for i in range(len(data_snapnum_dirs))] for j in range(len(resultsdirs))]
	members = [[pandas.DataFrame() for i in range(len(data_snapnum_dirs))] for j in range(len(resultsdirs))]
	non_dwarf_members = [[pandas.DataFrame() for i in range(len(data_snapnum_dirs))] for j in range(len(resultsdirs))]
	num_groups_std = [np.zeros(len(data_snapnum_dirs)) for j in range(len(resultsdirs))]
	num_members_std = [np.zeros(len(data_snapnum_dirs)) for j in range(len(resultsdirs))]
	num_non_dwarf_members_std = [np.zeros(len(data_snapnum_dirs)) for j in range(len(resultsdirs))]
	
	for result_ind,resultsdir in enumerate(resultsdirs):
		results_snapnum_dirs = np.sort(glob.glob(os.path.join(resultsdir,'snapnum_*')))
		print("Found {0} snapnum directories in {1}.".format(len(results_snapnum_dirs),resultsdir))
		if len(results_snapnum_dirs) == 0:
			return
		if len(data_snapnum_dirs) != len(results_snapnum_dirs):
			raise ValueError("Different number of snapnum directories in {0} and {1}".format(datadir,resultsdir))
		for snapnum,snapnum_dir in enumerate(results_snapnum_dirs):
			member_files = np.sort(glob.glob(os.path.join(snapnum_dir,'members','members_*')))
			print("Found {0} member files in {1}".format(len(member_files),snapnum_dir))
			group_files = np.sort(glob.glob(os.path.join(snapnum_dir,'groups','groups_*')))
			print("Found {0} group files in {1}".format(len(group_files),snapnum_dir))
			# Read in members data
			num_members = np.zeros(len(member_files))
			num_non_dwarf_members = np.zeros(len(member_files))
			for ind,member_file in enumerate(member_files):
				data = pandas.read_csv(member_file,header=0,comment='#',skip_blank_lines=True,index_col=0,skipinitialspace=True)
				members[result_ind][snapnum] = members[result_ind][snapnum].append(data)
				num_members[ind] = len(data)
				non_dwarf = data.loc[data['is_dwarf'] == False].copy()
				num_non_dwarf_members[ind] = len(data)
				non_dwarf_members[result_ind][snapnum] = non_dwarf_members[result_ind][snapnum].append(non_dwarf)
			check = np.std(num_members)*len(member_files)
			if np.isnan(check):
				num_members_std[result_ind][snapnum] = 0.
			else:
				num_members_std[result_ind][snapnum] = check
			check = np.std(num_non_dwarf_members)*len(member_files)
			if np.isnan(check):
				num_non_dwarf_members_std[result_ind][snapnum] = 0.
			else:
				num_non_dwarf_members_std[result_ind][snapnum] = check
			#Read in group data
			num_groups = np.zeros(len(group_files))
			for ind,group_file in enumerate(group_files):
				data = pandas.read_csv(group_file,header=0,comment='#',skip_blank_lines=True,index_col=0,skipinitialspace=True)
				groups[result_ind][snapnum] = groups[result_ind][snapnum].append(data)
				num_groups[ind] = len(data)
			check = np.std(num_groups)*len(group_files)
			if np.isnan(check):
				num_groups_std[result_ind][snapnum] = 0.
			else:
				num_groups_std[result_ind][snapnum] = check
	return (members, groups, num_non_dwarf_members, non_dwarf_members, num_members_std,
    			 num_non_dwarf_members_std, num_groups_std)


def data_read_in_evolution(resultsdirs,data_snapnum_dirs):
    #
    # Create pandas dataframe for each resultsdir and available snapnum
    #
	num_members = [[pandas.DataFrame() for i in range(64)] for j in range(len(resultsdirs))]
	num_members_std = [np.zeros(64) for j in range(len(resultsdirs))]
	print (resultsdirs)
	for result_ind,resultsdir in enumerate(resultsdirs):
		ev_file = np.array(glob.glob(resultsdir+'/Evolution_*'))
		data = pandas.read_csv(ev_file[0],header=0,comment='#',skip_blank_lines=True,
							skipinitialspace=True) 
		for x in range(64):
			num = len(data[data['snapnum'] == x])
			num_members[result_ind][x] = num
			num_members_std[result_ind][x] = np.sqrt(num)
	
	return num_members, num_members_std
			




def plot_evolution(plotlabel, galaxy_counts, num_members, num_members_std, resultsdirs,
					colors, symbols, lbtimes, plot_lookback_times, plot_redshifts, plotsdir,labels):
	#
    # Plot fractional number of galaxies in compact groups vs. redshift
    #
    
    figname = os.path.join(plotsdir,'{0}_frac_galaxies_evolution.eps'.format(plotlabel))
    frac_galaxies = np.array([[df/galaxy_counts['num_galaxies'][snapnum] for snapnum,df in enumerate(num_members[result_ind])] for result_ind in range(len(resultsdirs))])
    frac_galaxies_std = frac_galaxies * np.sqrt(np.array([[(num_members_std/df)**2. + (galaxy_counts['std_galaxies'][snapnum]/galaxy_counts['num_galaxies'][snapnum])**2.
                                                           for snapnum,(df,num_members_std) in enumerate(zip(num_members[result_ind],num_members_std[result_ind]))] for result_ind in range(len(resultsdirs))]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    for ind,(frac_galaxy,frac_galaxy_std) in enumerate(zip(frac_galaxies,frac_galaxies_std)):
        ax.plot(lbtimes,100.*frac_galaxy,color=colors[ind],marker=symbols[ind],label=labels[ind])
        ax.fill_between(lbtimes,100.*(frac_galaxy-frac_galaxy_std),100.*(frac_galaxy+frac_galaxy_std),
                        facecolor=colors[ind],edgecolor='none',alpha=0.6)
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel('Percent of non-dwarf Galaxies in that are currently in \nor have ever been in Compact Groups')
    
    ax.legend(loc='best',fontsize=12,numpoints=1)
    ax.set_xlim(0,WMAP7.age(0).to('Gyr').value)
    m = np.max(frac_galaxies)
    ax.set_ylim(0,10)
    #ax.set_yscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(plot_lookback_times)
    ax2.set_xticklabels(plot_redshifts)
    ax2.set_xlabel('Redshift')
    ax2.grid(False)
    plt.savefig(figname)
    plt.close()
    return




#=====================================================================
# Command Line Arguments
#=====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot results of Compact Group analysis",
        prog=_PROG_NAME,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    # Data parameters
    #
    parser.add_argument('--resultsdirs',type=str,nargs='+',default=['results'],
                        help="directory where results are saved.")
    parser.add_argument('--plotlabel',type=str,default='',
                        help="label to prepend to plot name.")
    parser.add_argument('--datadir',type=str,default='/data',
                        help="directory where raw snapnum data are saved.")
    parser.add_argument('--plotsdir',type=str,default='plots',
                        help="directory to save plots.")
    parser.add_argument('--labels',type=str,nargs='+',default=['results'],
                        help="labels for each result in plots.")
    parser.add_argument('--counts_file',type=str,default='galaxy_counts.txt',
                        help="File containing good galaxy counts in each snapnum. It will be created if it doesn't exist.")
    parser.add_argument('--snapnum_file',type=str,default='snapnum_redshift.csv',
                        help=("CSV file of snapnum-redshift conversion. Use query '"
                        "SELECT * FROM MField..Snapshots'."))
    parser.add_argument('--dwarf_limit',type=float,default=0.05,
                        help=('Stellar mass limit for dwarf galaxies in '
                               '10^10 Msun/h. Default: 0.05'))
    parser.add_argument('--evolution',action='store_true',default=False,
    					help=("Create Graphs for Evolution analysis"))
    #
    # Parse the arguments and send to main function
    #
    args = parser.parse_args()
    main(datadir=args.datadir,resultsdirs=args.resultsdirs,plotsdir=args.plotsdir,
         counts_file=args.counts_file,snapnum_file=args.snapnum_file,
         dwarf_limit=args.dwarf_limit,labels=args.labels,
         plotlabel=args.plotlabel,evolution=args.evolution)
