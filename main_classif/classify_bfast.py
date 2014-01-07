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
from sklearn import preprocessing
import random
from  matplotlib import pyplot as plt
from joblib import Parallel, Memory, delayed
import time
import argparse
import pylab as pl
import pickle
import os

### ALI's TOOLS
from confidence_computation import compute_confidence, compute_confidence_par
from contextual_computation import get_contextual, get_contextual_matlab
from classifier_wrappers import train_randomforest, train_adaboost, train_adaboost_par, train_svm

N_ESTIM = 40
learning_rate = 1.
N_SAMPLES = 1000
N_RUNS = 1
N_LAB = 10
CLF = 'svm' #'adaboost'#'randomforest' #
N_FEATURES = 500

#------------------------------------------------------------------------------#

def strip_features(level, features):
    
    import platform
    if platform.node() != 'g6':
        mat_path = '/Users/aarslan/Desktop/myFeats.mat'
    else:
        mat_path = '/home/aarslan/prj/data/monkey_new_ClassifData/myFeats.mat'

    ff = sp.io.loadmat(mat_path)
    all_sets = ff.keys()
    all_sets.sort()
    all_sets = all_sets[3:]
    take = np.squeeze(ff[str(all_sets[level])])
    features = features[:,np.append(take, range(-23,-1))]
    print 'stripped data have now ', str(features.shape[1]), ' features'
    return features
#------------------------------------------------------------------------------#
def get_bfast_splits(table_fname, settings, n_samples = N_SAMPLES, n_features = N_FEATURES, n_lab = N_LAB, contig_labels = True):
    
    h5_all = ta.openFile(table_fname, mode = 'r')
    table_all = h5_all.root.input_output_data.readout

    train_p = settings['train_p']
    test_p = settings['test_p']
    cams = settings['cameras']
    cur_cam = settings['cur_cam']
    lab_per_cam = {}
    
    print 'figuring out the shared labels between cams'
    for cam in cams:
        lab_per_pat = []
        for pat in train_p:
            rowz = [row['label'] for row in table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cam)) ]
            
            lab_per_pat += list(np.unique(rowz))
        lab_per_cam[cam] = np.unique(lab_per_pat)
    #import ipdb; ipdb.set_trace()
            
    aaa = lab_per_cam.values() #avoiding a possible bug here by making sure we take only the labels that exist for all cams
    labs_for_all = set(aaa[0]).intersection(*aaa) # will use this towards the end to weed out un-shared ones

    len_data = 0
    print 'figuring out how many training samples we have for the cam ', cur_cam
    for pat in train_p:
        rowz = [row['label'] for row in table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cur_cam)) ]
        len_data += len(rowz)
    
    features_train = np.empty((len_data,n_features), dtype='float64')
    labels_train = np.empty(len_data, dtype='|S24')
    
    cnt =0
    pbar = start_progressbar(len(train_p), str(len(train_p))+ ' training participants loading for cam '+cur_cam )
    for jj,pat in enumerate(train_p):
        temp = [row['features'] for row in table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cur_cam)) ]
        temp2 =  [row['label'] for row in table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cur_cam)) ]
        temp = [roww[:n_features] for roww in temp]
        if temp:
            #features_train[cnt:cnt+len(temp),:] = np.array(temp)[:,:n_features]
            features_train[cnt:cnt+len(temp),:] = temp
            labels_train[cnt:cnt+len(temp)] = temp2
            cnt = cnt+len(temp)
        update_progressbar(pbar, jj)
    end_progressbar(pbar)


    len_data = 0
    pbar = start_progressbar(len(test_p), ' now figuring out how many test samples we have' )
    for jj,pat in enumerate(test_p):
        for cam in cams:
            len_data += len([row['label'] for row in table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cam)) ])
        update_progressbar(pbar, jj)
    end_progressbar(pbar)

    features_test = np.empty((len_data,n_features), dtype='float64')
    labels_test = np.empty(len_data, dtype='|S24')

    cnt =0
    pbar2 = start_progressbar(len(test_p), str(len(test_p))+ ' testing participants loading' )
    for jj,pat in enumerate(test_p):
        for cam in cams:
            temp = [row['features'] for row in table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cam)) ]
            temp2 =  [row['label'] for row in table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cam)) ]
            temp = [roww[:n_features] for roww in temp]
            if temp:
                #features_train[cnt:cnt+len(temp),:] = np.array(temp)[:,:n_features]
                features_test[cnt:cnt+len(temp),:] = temp
                labels_test[cnt:cnt+len(temp)] = temp2
                #import ipdb; ipdb.set_trace()
                cnt = cnt+len(temp)
        update_progressbar(pbar2, jj)
    end_progressbar(pbar2)

    tic = time.time()
            
    uniqLabels = np.intersect1d(labs_for_all, np.unique(labels_test))
    #KILL UNUSED
    uniqLabels = uniqLabels[uniqLabels!='SIL']
    
    uniqLabels = uniqLabels[:n_lab]
    print 'using ',str(len(uniqLabels)),' labels in total'


    labels_train = np.array(labels_train)
    selector = np.zeros_like(labels_train, dtype= 'bool')
    excpt = 0
    
    for uL in uniqLabels:
        label_all = labels_train == uL
        label_all_subs = np.where(label_all)[0]
        if  label_all_subs.shape >= n_samples:
            label_some_subs = label_all_subs[:n_samples]
        else:
            excpt += 1
            label_some_subs = label_all_subs
        label_lim = np.zeros_like(label_all,dtype='bool')
        label_lim[label_some_subs] = True
        selector = np.squeeze(selector|[label_lim])
    labels_train = labels_train[selector]
    features_train = features_train[selector,:n_features]

    labels_test = np.array(labels_test)
    selector = np.zeros_like(labels_test, dtype= 'bool')
    for uL in uniqLabels:
        selector = np.squeeze(selector|[labels_test == uL])
    labels_test = labels_test[selector]
    features_test = features_test[selector,:n_features]
    print "Loaded features converted in ", round(time.time() - tic) , "seconds"
    print "there were ", str(excpt), " exceptions "
    
    table_all.flush()
    h5_all.close()

    labels_train = group_labels(labels_train)
    labels_test = group_labels(labels_test)
    print 'using ',str(len(labels_test)),' labels in total'
    return features_train , labels_train, features_test, labels_test 

#------------------------------------------------------------------------------#
def single_camera(table_path, settings):
    orig_feats, orig_labels, test_feats, test_labels = get_bfast_splits(
                                                       table_path, settings, 10000,
                                                       N_FEATURES, contig_labels = True,
                                                       n_lab = N_LAB)

    le = preprocessing.LabelEncoder()
    le.fit(orig_labels)
    orig_labels = le.transform(orig_labels)
    test_labels = le.transform(test_labels)

    #orig_feats= orig_feats.astype(np.float64)
    small_scaler = preprocessing.StandardScaler()
    orig_feats = small_scaler.fit_transform(orig_feats)
    #import ipdb; ipdb.set_trace()

    print 'FIRST ROUND: training with original features'
    allLearners_orig, used_labels = train_svm(orig_feats, orig_labels, N_LAB, N_RUNS, N_SAMPLES, test_feats, test_labels)
    pred = allLearners_orig[0].predict(test_feats)
    import ipdb; ipdb.set_trace()

    pred_sur = le.inverse_transform(pred)
    test_labels_sur = le.inverse_transform(test_labels)


    cm = confusion_matrix(test_labels_sur, pred_sur)
    norm_cm = np.divide(cm.T,sum(cm.T), dtype='float16').T
    test_acc = np.mean(norm_cm.diagonal())
    print 'the mean across the diagonal is ' + str(test_acc)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(norm_cm, interpolation='nearest')
    fig.colorbar(cax)

    ACTIONS = np.unique(test_labels_sur)
    ax.set_xticks(range(-1,len(ACTIONS)))
    ax.set_yticks(range(-1,len(ACTIONS)))
    ax.set_xticklabels(['']+list(ACTIONS), rotation='vertical')
    ax.set_yticklabels(['']+list(ACTIONS))
    ax.axis('image')

    #plt.show()
    #import ipdb; ipdb.set_trace()

    confidence_rich_train = compute_confidence(allLearners_rich, rich_feats, CLF)
    pred_train = np.argmax(confidence_rich_train, axis=1)

    cm = confusion_matrix(orig_labels, pred_train)
    norm_cm = np.divide(cm.T,sum(cm.T), dtype='float16').T
    train_acc = np.mean(norm_cm.diagonal())
    print 'the mean across the diagonal FOR TRAINING is ' + str(train_acc)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(norm_cm, interpolation='nearest')
    fig.colorbar(cax)

    ax.set_xticks(range(-1,len(ACTIONS)))
    ax.set_yticks(range(-1,len(ACTIONS)))
    ax.set_xticklabels(['']+list(ACTIONS), rotation='vertical')
    ax.set_yticklabels(['']+list(ACTIONS))
    ax.axis('image')

    #plt.show()

    import ipdb;ipdb.set_trace()
    return test_acc, train_acc, pred, pred_train, test_labels_sur, orig_labels, confidence_rich_test, confidence_rich_train
#------------------------------------------------------------------------------#
def group_labels(label_set):
    label_set[label_set=='take_bowl'] = 'take_container'
    label_set[label_set=='take_cup'] = 'take_container'
    label_set[label_set=='take_glass'] = 'take_container'
    label_set[label_set=='take_plate'] = 'take_container'
    label_set[label_set=='take_squeezer'] = 'take_container'
    
    label_set[label_set=='pour_juice'] = 'pour_liquid'
    label_set[label_set=='pour_coffee'] = 'pour_liquid'
    label_set[label_set=='pour_milk'] = 'pour_liquid'
    label_set[label_set=='pour_oil'] = 'pour_liquid'
    label_set[label_set=='pour_water'] = 'pour_liquid'
    
    label_set[label_set=='pour_egg2pan'] = 'pour_2pan'
    label_set[label_set=='pour_dough2pan'] = 'pour_2pan'
    
    label_set[label_set=='put_egg2plate'] = 'put_2plate'
    label_set[label_set=='put_pancake2plate'] = 'put_2plate'
    
    label_set[label_set=='fry_egg'] = 'fry_stuff'
    label_set[label_set=='fry_pancake'] = 'fry_stuff'
    
    label_set[label_set=='cut_fruit'] = 'cut_fruit'
    label_set[label_set=='cut_orange'] = 'cut_fruit'
    
    label_set[label_set=='stir_milk'] = 'stir_liquid'
    label_set[label_set=='stir_coffee'] = 'stir_liquid'
    label_set[label_set=='stir_tea'] = 'stir_liquid'
    
    label_set[label_set=='stir_cereals'] = 'stir_solid'
    label_set[label_set=='stir_dough'] = 'stir_solid'
    label_set[label_set=='stir_egg'] = 'stir_solid'
    label_set[label_set=='stir_fruit'] = 'stir_solid'
    
    label_set[label_set=='spoon_flour'] = 'spoon_powder'
    label_set[label_set=='spoon_powder'] = 'spoon_powder'
    label_set[label_set=='spoon_sugar'] = 'spoon_powder'
    
    #import ipdb; ipdb.set_trace()
    
    return label_set
#------------------------------------------------------------------------------#


def main():
    parser = argparse.ArgumentParser(description="""This file does this and that """)
    parser.add_argument('--table_path', type=str, help="""string""")
    parser.add_argument('--cams', type=str, nargs='+', default= 'webcam01')
    parser.add_argument('--out_dir', type=str, default= '/Users/aarslan/Desktop/bfast_ClassifData')
    args = parser.parse_args()
    
    table_path = args.table_path
    out_dir    = args.out_dir
    cams       = args.cams
    
    settings = {
        'train_p' :  ['P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10',
                      'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18',
                      'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26',
                      'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34',
                      'P35', 'P36', 'P37', 'P38', 'P39', 'P40', 'P41',
                      
                      #'P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49',
                      #'P50', 'P51', 'P52', 'P53', 'P54'
                      
                      ],

        'test_p': [   'P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49',
                      'P50', 'P51', 'P52', 'P53', 'P54'], #
            
        'cameras' : cams #, 'webcam02', 'cam01', 'cam02', 'stereo01'
                }
    te_acc   = {}
    tr_acc   = {}
    te_pred  = {}
    tr_pred  = {}
    te_lab   = {}
    tr_labels= {}
    te_conf  = {}
    tr_conf  = {}
    results = {}
    
    for cam in cams:
        print 'starting ',cam
        settings['cur_cam'] = cam
        te_acc[cam], tr_acc[cam], te_pred[cam], tr_pred[cam], te_lab[cam], tr_labels[cam], te_conf[cam], tr_conf[cam] = single_camera(table_path, settings)
    results['te_acc']=te_acc
    results['tr_acc']=tr_acc
    results['te_pred'] = te_pred
    results['tr_pred'] = tr_pred
    results['te_lab'] = te_lab
    results['tr_labels'] =tr_labels
    results['te_conf'] =te_conf
    results['tr_conf'] = tr_conf

    out_name= os.path.join(out_dir, 'results')
    io.savemat(out_name, results)
    import ipdb;ipdb.set_trace()
#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

#    
#    selector_path = '/home/aarslan/oldumulan'
#    if not os.path.exists(selector_path):
#        feats,labs = get_multi_sets(orig_feats, orig_labels, np.unique(orig_labels), 3000)
#        tic = time.time()
#        selector = LinearSVC(C=0.000006, penalty="l1", dual=False).fit(feats, labs)
#        print "time taken to score data is:", round(time.time() - tic) , "seconds"
#        container = {}
#        container['selector'] = selector
#        import ipdb; ipdb.set_trace()
#        pickle.dump(container, open(selector_path, 'wb'))
#    else:
#        import ipdb; ipdb.set_trace()
#        container = pickle.load(open(selector_path, 'rb'))
#        container['selector'].transform(orig_feats)
#    for range

