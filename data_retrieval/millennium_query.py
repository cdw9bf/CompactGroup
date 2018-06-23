"""
millennium_query.py

Query the Millennium Simulation UWS/TAP client.
Similar to tapquery in gavo.votable

Copyright(C) 2016 by
Trey Wenger; tvwenger@gmail.com

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

11 Mar 2016 - v1.0.0 TVW Finalized version 1.0
16 Mar 2016 - v1.0.1 TVW added unicode support in response.iter_content
21 Mar 2016 - v1.1   TVW added delete function
"""
_PROG_NAME = 'millennium_query.py'
_VERSION = 'v1.0.1'

import os
import time
import requests
from lxml import etree

_MILL_URL = "http://galformod.mpa-garching.mpg.de/millenniumtap/async"

def get_response(session,url,method='GET',data=None,
                 cookies=None,max_attempts=5,stream=False):
    """
    Perform call to website. Check that an error wasn't thrown,
    and check that we are logged in. Return response
    """
    n_attempts = 0
    while n_attempts < max_attempts:
        try:
            if method == 'GET':
                response = session.get(url,data=data,cookies=cookies,stream=stream)
            else:
                response = session.post(url,data=data,cookies=cookies,stream=stream)
            break
        except requests.exceptions.ConnectionError:
            # try again
            n_attempts += 1
    else:
        # it never worked...
        raise RuntimeError("Problem connecting to {0} with method {1} and data {2}".format(url,method,data))
    # it worked!
    if "Login at Millennium TAP" in str(response.content):
        raise RuntimeError("Login credentials not valid for MillenniumTAP")
    return response

def get_cookies(username,password):
    """
    Connect to the host just to get the login cookies
    """
    session = requests.Session()
    session.auth = (username,password)
    response = get_response(session,_MILL_URL,method='GET')
    cookies = session.cookies
    session.close()
    return cookies

class MillenniumQuery:
    """
    Container for handling query to Millennium Simulation database
    """
    def __init__(self,username,password,job_id=None,
                 query_lang='SQL',results_format='csv',
                 maxrec=100000,cookies=None):
        """
        Set up shared variables for this container
        """
        self.url = _MILL_URL
        self.post_data = {'LANG':query_lang,
                          'FORMAT':results_format,
                          'MAXREC':maxrec,
                          'REQUEST':'doQuery',
                          'VERSION':'1.0'}
        self.job_id = job_id
        self.job_url = None
        # If we already know the Job ID, set up proper URL
        if self.job_id is not None:
            self.job_url = '{0}/{1}'.format(self.url,self.job_id)
        # Get cookies if we don't already have some
        if cookies is None:
            self.cookies = get_cookies(username,password)
        else:
            self.cookies = cookies
        # Set up session
        self.session = requests.Session()
        self.session.auth = (username,password)

    def query(self,query):
        """
        Send query to server
        """
        # If we already know the job ID, use that URL
        if self.job_url is not None:
            url = self.job_url
        else:
            url = self.url
        post_data = self.post_data.copy()
        post_data['QUERY'] = query
        # Set up job
        response = get_response(self.session,url,method='POST',
                                data=post_data,cookies=self.cookies)
        # If we don't already have one, get Job ID from output XML
        if self.job_id is None:
            root = etree.fromstring(response.content)
            self.job_id = root.find('uws:jobId',root.nsmap).text
            self.job_url = '{0}/{1}'.format(self.url,self.job_id)

    def run(self):
        """
        Start job on server
        """
        # Check that we've already run query()
        if self.job_url is None:
            raise RuntimeError("Must first call query() to set up MillenniumQuery")
        url = '{0}/{1}'.format(self.job_url,'phase')
        post_data = self.post_data.copy()
        post_data['PHASE'] = 'RUN'
        response = get_response(self.session,url,method='POST',
                                data=post_data,cookies=self.cookies)

    def wait(self,max_attempts=100):
        """
        Wait for job to finish, then get the results
        Throw error if waiting for too long
        """
        attempts = 0
        wait_time = 1. # starting wait time
        increment = 2.**0.25 # magic number for wait time
        while True:
            phase = self.phase
            if phase == 'COMPLETED':
                break
            elif phase == 'ABORTED':
                raise RuntimeError("MillenniumQuery job {0} aborted!".format(self.job_id))
            elif phase == 'ERROR':
                raise RuntimeError("{0}".format(self.error_message))
            if max_attempts:
                if attempts > max_attempts:
                    raise RuntimeError("Did not get MillenniumQuery results after {0} iterations, job {1}".format(max_attempts,self.job_id))
            attempts += 1
            # wait for 120 seconds or 2^(niter/4) seconds,
            # whichever is shorter
            wait_time = min(120.,wait_time*increment)
            time.sleep(min(120.,wait_time))

    def save(self,filename):
        """
        Save results to filename
        """
        if self.job_url is None:
            raise RuntimeError("Must first call query() to set up MillenniumQuery")
        if self.phase != 'COMPLETED':
            raise RuntimeError("Job {0} is not completed yet!".format(self.job_id))
        url = '{0}/results/result'.format(self.job_url)
        # Stream chunks from download since the file could be huge
        response = get_response(self.session,url,method='GET',
                                cookies=self.cookies,stream=True)
        with open(filename,'w') as f:
            for chunk in response.iter_content(chunk_size=1024,
                                               decode_unicode=True):
                if chunk: # filter out "keep-alive" chunks
                    f.write(str(chunk))

    def delete(self):
        """
        Delete job from server
        """
        # Check that we've already run query()
        if self.job_url is None:
            raise RuntimeError("Must first call query() to set up MillenniumQuery")
        post_data = self.post_data.copy()
        post_data['ACTION'] = 'DELETE'
        response = get_response(self.session,self.job_url,method='POST',
                                data=post_data,cookies=self.cookies)

    def close(self):
        """
        Close the session
        """
        self.session.close()
        
    @property
    def phase(self):
        """
        Returns the current phase of the job
        """
        if self.job_url is None:
            raise RuntimeError("Must first call query() to set up MillenniumQuery")
        response = get_response(self.session,self.job_url,method='GET',
                                cookies=self.cookies)
        root = etree.fromstring(response.content)
        phase = root.find('uws:phase',root.nsmap).text
        return phase

    @property
    def error_message(self):
        """
        Returns the error message for this job
        """
        if self.job_url is None:
            raise RuntimeError("Must first call query() to set up MillenniumQuery")
        response = get_response(self.session,self.job_url,method='GET',
                                cookies=self.cookies)
        root = etree.fromstring(response.content)
        message = root.find('uws:errorSummary/uws:message',root.nsmap).text
        return message
