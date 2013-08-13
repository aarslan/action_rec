#!/usr/bin/env python
"""string"""

from hmax.tools.utils import start_progressbar, update_progressbar, end_progressbar
import scipy as sp
import glob
import numpy as np
import random
from joblib import Parallel, Memory, delayed
import time
import argparse
import pylab as pl
from multiprocessing import Process

#------------------------------------------------------------------------------#

def get_contextual(conf,Wsz):
    tic = time.time()

    nEx,nBhv = conf.shape
    nCF = (5*(pow(nBhv,2)))+(pow(nBhv,2))
    cf = conf
    kk=nBhv
    for ii in range(0,nBhv):
        for jj in range(1,nBhv):
            cf = np.concatenate((cf, np.array([conf[:,ii]-conf[:,jj]]).T),axis=1)
    cf = cf[:,np.sum(cf,axis=0)!=0]
    orig=cf

    #REST OF FEATURES
    winFeats =[]
    for ii in xrange(nEx):
        #import ipdb;ipdb.set_trace()
        winFeats.append(compute_windowed(orig,ii,Wsz))
    cf = np.concatenate((cf, np.array(winFeats)),axis=1)
    #cf = normalize(cf)
    return cf
#------------------------------------------------------------------------------#

def compute_windowed(orig,ii,wsz):
    nEx,nBhv2 = orig.shape
    wsz = (wsz-1)/2
    window = orig[max(0,ii-wsz):min(nEx,ii+wsz),:]
    
    cf1 = np.append(np.mean(window,0),
                    [window[-1,:]-window[0,:],
                     np.max(window, axis=0),
                     np.min(window, axis=0),
                     np.var(window, axis=0)])
    if any(np.isinf(cf1)):
        import ipdb;ipdb.set_trace()
    return cf1
#------------------------------------------------------------------------------#
def normalize(raw):
    high = 255.0
    low = 0.0
    mins = np.min(raw, axis=0)
    maxs = np.max(raw, axis=0)
    rng = maxs - mins
    scaled_points = high - (((high - low) * (maxs - raw)) / rng)
    return scaled_points
#------------------------------------------------------------------------------#

def get_contextual_matlab(conf,Wsz):
    import platform
    from pymatlab.matlab import MatlabSession
    if platform.node() != 'g6':
        dataPath = '/Users/aarslan/Brown/auto_context'
    else:
        dataPath = '/home/aarslan/prj/data/auto_context'

    print 'computing context features for windows size: ', str(Wsz)
    tic = time.time()
    session = MatlabSession()
    session.run('cd '+ dataPath)
    session.putvalue('conf',np.squeeze(np.array([conf], dtype='float64')))
    session.putvalue('Wsz',np.array([Wsz], dtype='float64'))
    session.run('B = ctxtFeat(conf, Wsz)')
    cf = session.getvalue('B')
    session.close()
    np.array(cf, dtype='uint8')
    print "Context feature for ", str(Wsz), " size windows took ", round(time.time() - tic,2), "seconds"
    return cf

