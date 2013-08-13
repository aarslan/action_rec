import argparse
import glob
import numpy as np
import random
import scipy as sp
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time

def classif_RBF(features_train, features_test, labels_train, labels_test, gamma, c):
    print "Starting classification for G=",gamma, ", c = ", c
    tic = time.time()
    clf = SVC(gamma=gamma, C=c)
    clf.fit(features_train, labels_train)
    score = clf.score(features_test, labels_test)
    print "selected score for G=",gamma, ", c = ", c, "is: ", score
    print "time taken:", round(time.time() - tic), "seconds"


def confidence_par(thisLab,ii, dada):
    #tic= time.time()
    res = np.zeros([dada.shape[0]])
    for jj, thisLearner in enumerate(thisLab):
        for hh, thisEstimator in enumerate(thisLearner):
            #multiply the predictions with the weight of the learner
            res = np.array(res+thisEstimator.predict(dada)*thisLearner.estimator_weights_[hh], dtype='float16')
    
    lab_confidence_perii = res
    #print "time taken to produce confidence:", round(time.time() - tic,2), "seconds"
    #import ipdb;ipdb.set_trace()
    return lab_confidence_perii, ii
