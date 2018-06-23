"""
cg_logger.py - Part of millennium-compact-groups package

Simple class to log actions

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
_PROG_NAME = 'cg_logger.py'
_VERSION = 'v1.0'

# System utilities
import os
import time

class Logger:
    """
    Object to handle output and logging to disk
    """
    def __init__(self,logfile,nolog=False,verbose=False):
        self.logfile = logfile
        self.nolog = nolog
        self.verbose = verbose
        if not self.nolog:
            with open(self.logfile,'w') as f:
                f.write('millennium-compact-groups log started {0}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

    def log(self,message):
        line = '{0}: {1}'.format(time.strftime('%Y-%m-%d %H:%M:%S'),message)
        if not self.nolog:
            with open(self.logfile,'a') as f:
                f.write('{0}\n'.format(line))
        if self.verbose:
            print(line)
