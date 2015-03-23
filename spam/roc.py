"""
calculate ROC curve
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/junyic/Work/Courses/4th_year/DataSci/project1/src')
import spam_database

archive_dir = '/home/junyic/Work/Courses/4th_year/DataSci/project1/trec07p_data/'
dataset_small = spam_database.DataSet(archive_dir, 1, False)

import sklearn.naive_bayes as naive_bayes
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import recall_score, roc_curve
from sklearn import svm

train_features = dataset_small.get_train_features()
train_labels   = dataset_small.get_train_labels()

do_feature_selection = True

if do_feature_selection:
    kbestfilter = SelectKBest(chi2, k=500)
    train_features = kbestfilter.fit_transform(train_features,
                                               train_labels)
    test_features = kbestfilter.transform(dataset_small.get_test_features())
else:
    test_features = dataset_small.get_test_features()
    
#classifier = naive_bayes.MultinomialNB()
classifier  = svm.LinearSVC(verbose=2)
spam_filter = classifier.fit(train_features,
                          train_labels)
#spam_probs = classifier.predict_proba(test_features)[:,1]
spam_probs = classifier.decision_function(test_features)
fpr, tpr, thresholds = roc_curve(dataset_small.get_test_labels(), 
                                 spam_probs, pos_label=1)
## save ROC                                 
npsave_dir = '/home/junyic/Work/Courses/4th_year/DataSci/project1/'
np.savez(npsave_dir + 'NBB_feature_select', tpr, fpr)