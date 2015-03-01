"""
Support vector machine
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/junyic/Work/Courses/4th_year/DataSci/project1/src')
import spam_database

archive_dir = '/home/junyic/Work/Courses/4th_year/DataSci/project1/trec07p_data/'
dataset_small = spam_database.DataSet(archive_dir, 1, False)

#train_features = dataset_small.get_train_features()
#test_features  = dataset_small.get_test_features()

from sklearn import svm
import time

## train SVM
clf = svm.SVC(verbose=2)
#clf = svm.LinearSVC(verbose=2)
start_time = time.time()

clf.fit(train_features, dataset_small.get_train_labels())
spam_pred = clf.predict(test_features)

end_time  = time.time()
print "computational time %f" %(end_time - start_time)

## cross validataion
from sklearn import cross_validation
from sklearn.metrics import recall_score
C_set = np.linspace(0.1, 1e4, 50)
fall_out = np.empty_like(C_set)
recall   = np.empty_like(C_set)
for i, Ci in enumerate(C_set):
    clf = svm.LinearSVC(verbose=2, C=Ci)
    clf.fit(train_features, dataset_small.get_train_labels())
    spam_pred = clf.predict(test_features)
    recall_nospam, recall_spam = recall_score(
        dataset_small.get_test_labels(), spam_pred, average=None)
    fall_out[i] = 1 - recall_nospam
    recall[i]   = recall_spam

##

import matplotlib.pyplot as plt
plt.plot(np.sort(fall_out), recall[fall_out.argsort()], '-or')
#plt.plot(np.linspace(0, 1., 10), np.linspace(0, 1., 10), '--k')
plt.show()
## report
import sklearn.metrics
report = sklearn.metrics.classification_report(dataset_small.get_test_labels(),
                                               spam_pred)
print report