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
def compute_confidence(allLearners, dada, classifier_type):
    #import ipdb;ipdb.set_trace()
    
    tic = time.time()
    #import ipdb;ipdb.set_trace()
    
    if classifier_type == 'adaboost':
        lab_confidence = np.zeros([dada.shape[0], len(allLearners)], dtype='float64')
        pbar = start_progressbar(len(allLearners), '%i producing weighted outputs' % len(allLearners))
        for ii,thisLab in enumerate(allLearners):
            res = np.zeros([dada.shape[0]], dtype='float64')
            for jj, thisLearner in enumerate(thisLab):
                my_weights = thisLearner.estimator_weights_
                #tic = time.time()
                for hh, thisEstimator in enumerate(thisLearner):
                    res = res+thisEstimator.predict(dada)*my_weights[hh]
                    #import ipdb;ipdb.set_trace()
            lab_confidence[:,ii] = np.float64(res)
            update_progressbar(pbar, ii)
        end_progressbar(pbar)
    
    if classifier_type == 'randomforest' or classifier_type == 'svm':
        #import ipdb;ipdb.set_trace()
        lab_confidence = np.zeros((dada.shape[0],len(allLearners[0].classes_)), dtype='float64')
        pbar = start_progressbar(len(allLearners), '%i producing weighted outputs' % len(allLearners[0].classes_))
        for ii, thisRun in enumerate(allLearners):
            lab_confidence +=  thisRun.predict_proba(dada)
            update_progressbar(pbar, ii)
        end_progressbar(pbar)

    return lab_confidence

#------------------------------------------------------------------------------#
def compute_confidence_par(allLearners, samples, classifier_type):
    from joblib import Parallel, delayed
    from joblib import load, dump
    import tempfile
    import shutil
    import os
    folder = tempfile.mkdtemp()
    samples_name = os.path.join(folder, 'samples')
    dump(samples, samples_name)
    samples = load(samples_name, mmap_mode='r')
    
    try:
        out = Parallel(n_jobs=3)(delayed(confidence_par)(thisLab,ii, samples) for ii,thisLab in enumerate(allLearners))
        all_conf = np.zeros((len(out[0][0]),len(out)), dtype='float64')
        for cnf in out:
            all_conf[:,cnf[1]] = cnf[0]
    finally:
        shutil.rmtree(folder)
    return all_conf