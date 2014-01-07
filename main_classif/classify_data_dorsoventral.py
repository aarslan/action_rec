#!/usr/bin/env python
"""string"""

import h5py
import scipy as sp

import hmax
from hmax.classification import kernel
#from shogun import Kernel, Classifier, Features
from hmax.tools.utils import start_progressbar, update_progressbar, end_progressbar
import scipy as sp
import numpy as np
from scipy import io
import tables as ta
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from  matplotlib import pyplot as plt

from set_extraction import get_multi_sets
import classify_data_monkey as cm


import time
import argparse


REGULARIZATION_VALUE = 1E4
N_SAMPLES = 1000
N_FEATURES  = 930
N_SPLIT = 15
l_c = [1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2, 1E3, 1E4]
#------------------------------------------------------------------------------#
def read_table(table_fname, n_samples = N_SAMPLES, n_features = N_FEATURES):

    h5 = ta.openFile(table_fname, mode = 'r')
    table = h5.root.input_output_data.readout

    features = sp.array(table.cols.features, dtype = 'float32')
    labels = sp.array(table.cols.actions, dtype = 'string')

    table.flush()
    h5.close()

    return features, labels
#------------------------------------------------------------------------------#
def svm_cla_sklearn(features_train, features_test, labels_train, labels_test):
    """docstring for svm_sklearn"""

    features_train = sp.array(features_train, dtype = 'float32')
    features_test = sp.array(features_test, dtype = 'float32')

    scaler = preprocessing.StandardScaler()
    features_train_prep = scaler.fit_transform(features_train)
    features_test_prep = scaler.transform(features_test)

    #for c in l_c:
    c  = 500000
    clf = SVC(gamma = 1, C=c)
    #clf = SVC(gamma=2, C=1),
    clf.fit(features_train, labels_train)
    score = clf.score(features_test, labels_test)
    print "score for C,",c, "is: ", score
    return clf.predict(features_test)

#------------------------------------------------------------------------------#
def main():

    parser = argparse.ArgumentParser(description="""This file does this and that \n
            usage: python ./classify_data.py --n_samples 10 --n_features 100 --features_fname ./bla.mat --labels_fname ./bla1.mat""")
    parser.add_argument('--n_features', type=int, default = N_FEATURES, help="""string""")
    parser.add_argument('--n_samples', type=int, default = N_SAMPLES, help="""string""")
    parser.add_argument('--table_fname', type=str, help="""string""")
    args = parser.parse_args()

    table_fname = args.table_fname
    n_features = args.n_features
    n_samples = args.n_samples
    
    features, labels = read_table(table_fname, n_samples, n_features)
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    
    all_pred=[]
    all_test=[]
    aa, bb = np.unique(labels, return_inverse='True')
    
    for ii in range(0, N_SPLIT):
        feats, labs = get_multi_sets(features, labels, np.unique(labels), min(np.bincount(bb)))
        features_train, features_test, labels_train, labels_test = train_test_split(feats, labs, test_size=0.33)
        labels_pred = svm_cla_sklearn(features_train, features_test, labels_train, labels_test)
        all_pred.append(labels_pred)
        all_test.append(labels_test)
    
    all_test = sp.array(all_test).ravel()
    all_pred = sp.array(all_pred).ravel()
    #import ipdb; ipdb.set_trace()



    cm = confusion_matrix(all_test, all_pred)
    norm_cm = np.divide(cm.T,sum(cm.T), dtype='float16').T
    print 'the mean across the diagonal is ' + str(np.mean(norm_cm.diagonal()))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(norm_cm, interpolation='nearest')
    fig.colorbar(cax)


    ax.set_xticks(range(-1,len(le.classes_)))
    ax.set_yticks(range(-1,len(le.classes_)))
    ax.set_xticklabels(['']+list(le.classes_), rotation='vertical')
    ax.set_yticklabels(['']+list(le.classes_))
    ax.axis('image')

    plt.show()
    import ipdb; ipdb.set_trace()
#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

