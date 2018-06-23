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


import pandas as pd
import numpy as np
import psycopg2
import multiprocessing as mp
import argparse
from multiprocessing import Pool
import sys
import time

DB_STRING = "dbname='{dbname}' user='{username}' host='{hostname}' password='{password}'".format(dbname="MYPOSTGRESDB", username="MYPOSTGRESUSERNAME", hostname="PostgresHostname", password="DB Password")


def trace_groups(args):
    """
    Worker process. Accesses Data from a Postgres Db, then iterates through each compact group galaxy finding its
    descendants. Once it processes all galaxies in a specific treeId, it checks for duplicates.
    The data is then sent to a writer queue where the data is written to a text file. It also
    places a count in the monitoring queue after it completes a treeId.
    :param args: List of Values for each worker.
    :return: None

    Args
    0: List of Tree Ids
    1: List of Galaxies for those Tree Ids
    2: Output File Writer Queue
    3: Status Queue for Main Thread
    """

    tree_id = args[0]
    galaxies = args[1]
    writer_queue = args[2]
    job_queue = args[3]
    conn = psycopg2.connect(DB_STRING)

    for curr_tree_id in tree_id:
        try:
            trace = []
            cur = conn.cursor()
            cur.execute("SELECT * FROM snaps WHERE treeid={0};".format(curr_tree_id))
            rows = cur.fetchall()
        
            all_gal = pd.DataFrame(rows, columns=["galaxyid", "redshift", "firstprogenitorid", "descendantid", "treeid", "snapnum"])
            # Selects galaxies from initial snapshot that have same tree ID as currently being processed
            galx = galaxies[galaxies['treeID'] == curr_tree_id]
            for ix, gal in galx.iterrows():
                # Finds the matching record retrieved from the DB
                current_gal = all_gal[all_gal['galaxyid'] == gal['member_id']]
                possible_descendants = [[] for x in range(64)]
                possible_descendants[0] = current_gal
                count = 0
                # Iterates through descendants of current galaxy until it hits one without a descendant
                while current_gal.get_value(current_gal.iloc[0].name, 'descendantid') != -1:
                    count += 1
                    next_gal = all_gal[all_gal['galaxyid'] == current_gal.get_value(current_gal.iloc[0].name, 'descendantid')]
                    possible_descendants[count] = next_gal
                    current_gal = next_gal
           
                if count != 0:
                    trace.append(pd.concat(possible_descendants[0:count+1]))
            trace = pd.concat(trace)
            trace = trace.drop_duplicates(['galaxyid'], keep='first')
            writer_queue.put(trace)
        except Exception as e:
            print(e)
            print("Had Exception on {0}".format(curr_tree_id))
        finally:
            job_queue.put(1)

    return


def writing_queue(q):
    """
    Reads from the writer queue and writes the data to a csv. Once all the data has been processed, it receives a 'done'
    string which will kill the worker.
    :param q: Queue object
    :return: None
    """
    queue = q
    # POPULATE WITH CORRECT FILE LOCATION FOR YOUR SYSTEM
    with open("/tmp/all_members.csv", "w", int(16777216/4)) as f:
        f.write("galaxyid,redshift,firstprogenitorid,descendantid,treeid,snapnum\n")
        
        while True:
            item = queue.get()
            if type(item) == str and item == "done":
                break
            if type(item) == str:
                print(item)
            item.to_csv(f, header=False, index=False)
            f.flush()


def main(pool_size, group_id_file):
    """
    Reads initial file for groups at a certain Snapshot then creates workers for tracing these galaxies.
    :param pool_size:
    :param group_id_file:
    :return:
    """
    all_groups = pd.read_csv(group_id_file, header=0)
    
    tree_ids = np.unique(all_groups['treeID'])

    manager = mp.Manager()
    writer_queue = manager.Queue()
    job_status_queue = manager.Queue()
    pool = mp.Pool(pool_size)
    max_jobs = pool_size

    # Start Writer Thread
    queue_job = pool.map_async(writing_queue, [writer_queue])

    all_jobs = [[] for i in range(max_jobs)]

    print("Building Jobs")
    total_job_count = int(len(tree_ids) / pool_size)
    # Splits the Tree Ids into different jobs
    for i in range(max_jobs):
        if i < max_jobs - 1:
            tree_id = tree_ids[i*total_job_count:(i+1)*total_job_count]
        else:
            tree_id = tree_ids[i*total_job_count::]
        sys.stdout.write("\r{0:.2f}%".format(i/max_jobs * 100))
        current_ids = all_groups.loc[all_groups["treeID"].isin(tree_id)]
        all_jobs[i] = [tree_id, current_ids, writer_queue, job_status_queue]

    print("Starting Workers")
    print(len(all_jobs))
    results = pool.map_async(trace_groups, all_jobs)
    jobs_done = 0
    sys.stdout.write("\n")

    start_time = time.time()
    while jobs_done < len(tree_ids):
        try:
            # If no messages come within 30 seconds, all workers are probably be done. Configure based on your system performance
            # Should also read 100% at this point.
            jobs_done += job_status_queue.get(timeout=30)
            sys.stdout.write("\r{0:.2f}%".format(jobs_done/len(tree_ids) * 100))
        except Exception as e:
            print(e)
            print("Queue Timed out, trying to abort process now")
            break
    
    end_time = time.time() - start_time
    
    print("\n\nEnding time: {0}".format(end_time))
    # Kill signal for writer thread
    writer_queue.put("done")

    # Waits for threads to finish executing
    pool.close()
    pool.join()
    
    sys.stdout.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pool_size", help="Number of Cores to use. Minimum=2", default=8)
    parser.add_argument("--group_file_location", help="Location of file containing all group ids")
    args = parser.parse_args()

    main(pool_size=args.pool_size, group_id_file=args.group_file_location)
