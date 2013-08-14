#!/usr/bin/env python
"""string"""

import scipy as sp
import glob
import numpy as np
from scipy import io
import tables as ta
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import random
from joblib import Parallel, Memory, delayed
import time
from multiprocessing import Process

###
import hmax
from hmax.tools.utils import start_progressbar, update_progressbar, end_progressbar
from set_extraction import get_binary_sets, get_multi_sets

#------------------------------------------------------------------------------#
def train_adaboost(features, labels, learning_rate, n_lab, n_runs, n_estim, n_samples):
    uniqLabels = np.unique(labels)
    print 'Taking ', str(n_lab), ' labels'
    uniqLabels = uniqLabels[:n_lab]
    used_labels = uniqLabels
    pbar = start_progressbar(len(uniqLabels), 'training adaboost for %i labels' %len(uniqLabels))
    allLearners = []
    for yy ,targetLab in enumerate(uniqLabels):
        runs=[]
        for rrr in xrange(n_runs):
            #import ipdb;ipdb.set_trace()
            feats,labs = get_binary_sets(features, labels, targetLab, n_samples)
            #print 'fitting stump'
            #import ipdb;ipdb.set_trace()
            baseClf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, min_samples_split=4)
            baseClf.fit(feats, labs)
            ada_real = AdaBoostClassifier( base_estimator=baseClf, learning_rate=learning_rate,
                                      n_estimators=n_estim,
                                      algorithm="SAMME.R")
            #import ipdb;ipdb.set_trace()
            runs.append(ada_real.fit(feats, labs))
        allLearners.append(runs)
        update_progressbar(pbar, yy)
    end_progressbar(pbar)
    
    return allLearners, used_labels

#------------------------------------------------------------------------------#
def train_adaboost_par(features, labels, learning_rate, n_lab, n_runs, n_estim, n_samples):
    from joblib import Parallel, delayed
    from joblib import load, dump
    import tempfile
    import shutil
    import os
    folder = tempfile.mkdtemp()
    samples_name = os.path.join(folder, 'samples')
    dump(samples, samples_name)
    samples = load(samples_name, mmap_mode='r')
    
    uniqLabels = np.unique(labels)
    print 'Taking ', str(n_lab), ' labels'
    uniqLabels = uniqLabels[:n_lab]
    used_labels = uniqLabels
    allLearners = []
    
    try:
        out = Parallel(n_jobs=1)(delayed(adaboost_worker)(thisLab,ii, samples) for yy ,targetLab in enumerate(uniqLabels))
        all_conf = np.zeros((len(out[0][0]),len(out)), dtype='float64')
        all_learners
        for cnf in out:
            all_learners[:,cnf[1]] = rns[0]
    finally:
        shutil.rmtree(folder)
    
    for yy ,targetLab in enumerate(uniqLabels):
        runs=[]
        for rrr in xrange(n_runs):
            #import ipdb;ipdb.set_trace()
            feats,labs = get_binary_sets(features, labels, targetLab, n_samples)
            #print 'fitting stump'
            #import ipdb;ipdb.set_trace()
            baseClf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=1, min_samples_split=1)
            baseClf.fit(feats, labs)
            ada_real = AdaBoostClassifier( base_estimator=baseClf, learning_rate=learning_rate,
                                          n_estimators=n_estim,
                                          algorithm="SAMME.R")
            #import ipdb;ipdb.set_trace()
            runs.append(ada_real.fit(feats, labs))
        allLearners.append(runs)
    
    return all_learners, used_labels

#------------------------------------------------------------------------------#

def train_randomforest(features, labels, n_lab, n_runs, n_estim, n_samples):
    
    uniqLabels = np.unique(labels)
    print 'TAKING ONLY ', str(n_lab), ' LABELS FOR SPEED '
    print "using random forests"
    uniqLabels = uniqLabels[:n_lab]
    used_labels = uniqLabels
    
    allLearners = []
    #import ipdb;ipdb.set_trace()
    for rrr in xrange(n_runs):
        #import ipdb;ipdb.set_trace()
        feats,labs = get_multi_sets(features, labels, used_labels, n_samples)
        #import ipdb;ipdb.set_trace()
        rfclf = RandomForestClassifier(n_estimators=n_estim, max_depth=16, min_samples_split=16, random_state=0)
        #import ipdb;ipdb.set_trace()
        allLearners.append(rfclf.fit(feats, labs))
    
    return allLearners, used_labels
