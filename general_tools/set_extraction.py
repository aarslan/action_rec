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
from aux_functions import confidence_par


#------------------------------------------------------------------------------#
def get_binary_sets(features, labels, target_lab, sample_n):
    #import ipdb;ipdb.set_trace()
    trainPos_idx = np.where(labels == target_lab)[0]
    trainNeg_idx = np.where(labels != target_lab)[0]
    if len(trainPos_idx)<sample_n/2:
        sample_id = np.array(trainPos_idx);
        sample_id = np.concatenate((sample_id, trainNeg_idx[random.sample(range(0,len(trainNeg_idx)), sample_n-len(trainPos_idx))]));
    else:
        sample_id = np.array(trainPos_idx[random.sample(range(0,len(trainPos_idx)), sample_n/2)])
        sample_id = np.concatenate((sample_id, trainNeg_idx[random.sample(range(0,len(trainNeg_idx)), sample_n/2)]));
    
    feats = np.array(features[sample_id,:], dtype='float64');
    labs = np.array(labels[sample_id], dtype='int8');
    posInd = labs==target_lab
    negInd = labs!=target_lab
#import ipdb;ipdb.set_trace()
    labs[posInd] = 1;
    labs[negInd] = -1; #convert labels of positive samples to +1 and labels of negative samples to -1
    #import ipdb;ipdb.set_trace()
    return feats,labs

#------------------------------------------------------------------------------#
def get_multi_sets(features, labels, used_labels, sample_n):
    from sklearn.cross_validation import train_test_split
    #size = (sample_n*used_labels.shape[0])/float(labels.shape[0])
    #feats, null1, labs, null2 = train_test_split(features, labels, train_size=size)
    
    labels_train = []
    
    #nn, bins, patches = pl.hist(labels, len(used_labels))
    #sample_n = min(nn)
    cnt = 0
    #import ipdb;ipdb.set_trace()
    features_train = np.zeros(((sample_n*used_labels.shape[0]), features.shape[1]), dtype = 'float32')
    for myLab in used_labels:
        #import ipdb;ipdb.set_trace()
        all_exemplars = np.where(labels == myLab)[0]
        try:
            selInd = np.random.choice(all_exemplars, sample_n, replace=False)
        except ValueError:
            #import ipdb;ipdb.set_trace()
            selInd = np.random.choice(all_exemplars, sample_n, replace=True)
        uzun = len(selInd)
        labels_train = labels_train + list(labels[selInd])
        #import ipdb;ipdb.set_trace()
        features_train[cnt:cnt+uzun,:] = features[selInd,:]
        cnt = cnt+uzun

    feats = features_train#[features_train[:,1]!=0, :]
    labs = np.array(labels_train)
    if labs.shape[0] != feats.shape[0]:
        import ipdb;ipdb.set_trace()
        raise ValueError('the label and feat dimensions in get_multi_set dont match')
    #import ipdb;ipdb.set_trace()
    return feats,labs
