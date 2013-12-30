#!/usr/bin/env python

import tables   as ta
import scipy    as sp
import commands
import scipy
from scipy import random
from hmax.tools.utils import start_progressbar, end_progressbar, update_progressbar
import argparse
from scipy import io
import time
import h5py


#------------------------------------------------------------------------------#
def create_empty_table(table_fname):
    
    class images(ta.IsDescription):
        features     = ta.Float64Col(shape = (930))
        actions        = ta.StringCol(64)
	names        = ta.StringCol(64)

    h5    = ta.openFile(table_fname, mode = 'w', title='list of images')
    group = h5.createGroup("/", 'input_output_data', 'images information')
    table = h5.createTable(group, 'readout', images, "readout example")
    #pp = table.row
    table.flush()
    h5.close()

#------------------------------------------------------------------------------#
def read_mat_files(feature_fname, info_fname):
    """docstring for read_mat_file"""
    
    print "reading features"
    tic = time.time()
    #f = h5py.File(feature_fname, 'r')
    import ipdb;ipdb.set_trace()
    f = io.loadmat(feature_fname)
    ff = f["feats"]
    features = sp.array(ff)
    print "time taken :", time.time() - tic, 'seconds'

    print "reading actions"
    tic = time.time()
    actions = io.loadmat(info_fname)['act']
    actions = sp.array([str(actions[i][0][0]) for i in xrange(actions.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'

    print "reading names"
    tic = time.time()
    names = io.loadmat(info_fname)['name']
    names = sp.array([str(names[i][0][0]) for i in xrange(names.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'
    
    return features, actions, names
#------------------------------------------------------------------------------#
def populate_table(table_fname, features, actions, names):
    n_samples = names.shape[0]
    pbar = start_progressbar(n_samples, '%i features to Pytable' % (n_samples))
    
    h5 = ta.openFile(table_fname, mode = 'a')
    table = h5.root.input_output_data.readout
    pp = table.row
    
    for i in xrange(n_samples):
        pp['features']    = features[i,:]
        pp['actions']       = actions[i]
	pp['names']       = names[i]
        pp.append()
        update_progressbar(pbar, i)
    
    end_progressbar(pbar)
    # save everything in the file and close it
    table.flush()
    h5.close()

#------------------------------------------------------------------------------#
def main():
    """
        
        """
    parser = argparse.ArgumentParser(description="""This file does this and that \n
        usage: python ./file.py 11 --bla 10  blabla""")
    parser.add_argument('--feature_fname', type=str, help="""string""")
    parser.add_argument('--info_fname', type=str, help="""string""")
    parser.add_argument('--table_fname', type=str, help="""string""")
    parser.add_argument('--data_path', type=str, help="""this is the path for all the data files""")
    args = parser.parse_args()
    
    data_path = args.data_path
    feature_fname = data_path+args.feature_fname
    info_fname = data_path+args.info_fname
    table_fname =  data_path+args.table_fname
    
    features, actions, names = read_mat_files(feature_fname, info_fname)
    create_empty_table(table_fname)
    populate_table(table_fname, features, actions, names)


#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

