#!/usr/bin/env python
"""string"""

import h5py
import hmax
from hmax.tools.utils import start_progressbar, update_progressbar, end_progressbar
import scipy as sp
import glob
import numpy as np
from scipy import io
import tables as ta
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import random
from  matplotlib import pyplot as plt
from joblib import Parallel, Memory, delayed
import time
import argparse
import pylab as pl
from multiprocessing import Process

### ALI's TOOLS
from confidence_computation import compute_confidence, compute_confidence_par
from contextual_computation import get_contextual, get_contextual_matlab
from classifier_wrappers import train_randomforest, train_adaboost, train_adaboost_par



#------------------------------------------------------------------------------#
def load_training(table_path, splitNo, trainOrTest):
    print 'loading ' + trainOrTest + ' features'
    h5 = ta.openFile(table_path + trainOrTest +'_' + str(splitNo) + '.h5', mode = 'r')
    table = h5.root.input_output_data.readout
    import ipdb;ipdb.set_trace()
    tic = time.time()
    features = sp.array(table.cols.features)#[:,:N_LIM]
    print "time taken:", time.time() - tic, "seconds"
    labels = sp.array(table.cols.label)
    print 'features loaded'
    return features, labels

#------------------------------------------------------------------------------#
def load_training_mats(mat_path, splitNo, trainOrTest):
    myData = [];
    labels = [];
    names = [];
    labFiles = glob.glob(mat_path + trainOrTest + '/*_labels_double.mat')
    
    features = np.array([])
    labels = np.array([])
    labFiles = [x[0:-18] for x in labFiles]
    print 'loading ' + trainOrTest + ' features'
    tic = time.time()
    
    for myFile in labFiles:
        dd = sp.io.loadmat(myFile+'_xavier_features.mat')['positon_features']
        ll = sp.io.loadmat(myFile+'_labels_double.mat')['labels_double']
        features = np.concatenate([x for x in [features, dd] if x.size > 0],axis=1)
        labels = np.concatenate([x for x in [labels, ll] if x.size > 0],axis=0)
    
            #import ipdb;ipdb.set_trace()

    features = np.array(features.T, dtype='float64')
    labels = labels[:,0]

    print "time taken:", round(time.time() - tic,2), "seconds"
    print str(features.shape[0]), ' features loaded'
    return features, labels

#------------------------------------------------------------------------------#
def main():
    """
        This is where the magic happens
        """
    parser = argparse.ArgumentParser(description="""This file does this and that \n
        usage: python ./classify_data.py --n_samples 10 --n_features 100 --features_fname ./bla.mat --labels_fname ./bla1.mat""")
    parser.add_argument('--table_path', type=str, help="""string""") ##THIS IS THE BASE NAME, PARTS WILL BE ADDED IN THE CODE
    parser.add_argument('--mat_path', type=str, default = '0', help="""string""")
    parser.add_argument('--split_no', type=int, default = 1, help="""string""")
    args = parser.parse_args()
    
    table_path = args.table_path
    mat_path = args.mat_path
    splitNo = args.split_no
    
    if mat_path == '0':
        orig_feats,orig_labels = load_training(table_path, splitNo, 'train')
    else:
        orig_feats,orig_labels = load_training_mats(mat_path, splitNo, 'train')

    orig_feats= orig_feats.astype(np.float64)
    allLearners_orig, used_labels = train_adaboost(orig_feats,orig_labels)
#    tic = time.time()
#    confidence_orig = compute_confidence_new(allLearners_orig, orig_feats)
#    print "time taken new way:", round(time.time() - tic,2), "seconds"
    tic = time.time()
    confidence_orig= compute_confidence(allLearners_orig, orig_feats)
    print "time taken old way:", round(time.time() - tic,2), "seconds"

            
    #import ipdb;ipdb.set_trace()

    orig_CF_75 = get_contextual(confidence_orig, 75)

    orig_CF_75 = get_contextual_matlab(confidence_orig, 75)
    orig_CF_185 = get_contextual_matlab(confidence_orig, 185)
    orig_CF_615 = get_contextual_matlab(confidence_orig, 615)
    CF_feats = np.concatenate([orig_CF_75,orig_CF_185,orig_CF_615], axis = 1)

    rich_feats = np.concatenate([orig_feats,CF_feats], axis=1)
    allLearners_rich = train_adaboost(rich_feats,orig_labels)

    if mat_path == '0':
        test_feats,test_labels = load_training(table_path, splitNo, 'test')
    else:
        test_feats,test_labels = load_training_mats(mat_path, splitNo, 'test')

    test_feats= test_feats.astype(np.float64)
    confidence_test = compute_confidence(allLearners_orig, test_feats)
    #confidence_test_par = compute_confidence_par(allLearners_orig, test_feats)

    
    test_CF_75 = get_contextual_matlab(confidence_test, 75)
    test_CF_185 = get_contextual_matlab(confidence_test, 185)
    test_CF_615 = get_contextual_matlab(confidence_test, 615)
    test_CF_feats = np.concatenate([test_CF_75, test_CF_185, test_CF_615], axis = 1)
    rich_test_feats = np.concatenate([test_feats, test_CF_feats], axis=1)
    
    confidence_rich_test = compute_confidence(allLearners_rich, rich_test_feats)
    #confidence_rich_test = compute_confidence_par(allLearners_rich, rich_test_feats)
    pred = np.argmax(confidence_rich_test, axis=1)

    testUnique = np.unique(test_labels)[:N_LAB]

    import ipdb;ipdb.set_trace()

    used_labs = np.sum([test_labels == lab for lab in testUnique],0)
    truth = test_labels[used_labs.astype('bool')]
    pred2 = testUnique[pred][used_labs.astype('bool')]


    cm = confusion_matrix(truth, pred2)
    norm_cm = np.divide(cm.T,sum(cm.T), dtype='float64').T
    print 'the mean across the diagonal is ' + str(np.mean(norm_cm.diagonal()))
    #    pl.matshow(norm_cm)
    #    pl.colorbar()
    #    pl.show()

    #alpha = ['ABC', 'DEF', 'GHI', 'JKL']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(norm_cm, interpolation='nearest')
    fig.colorbar(cax)

    ax.set_xticks(range(-1,len(ACTIONS)))
    ax.set_yticks(range(-1,len(ACTIONS)))
    ax.set_xticklabels(['']+list(ACTIONS), rotation='vertical')
    ax.set_yticklabels(['']+list(ACTIONS))
    ax.axis('image')

    plt.show()


    import ipdb;ipdb.set_trace()
#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()
