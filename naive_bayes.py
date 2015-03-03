"""
Naive Bayes classfier
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/junyic/Work/Courses/4th_year/DataSci/project1/src')
import spam_database

archive_dir = '/home/junyic/Work/Courses/4th_year/DataSci/project1/trec07p_data/'
dataset_small = spam_database.DataSet(archive_dir, 1, False)
#dataset_large = spam_database.DataSet('/home/junyic/Work/Courses/4th_year/DataSci/project1/old_data/P1Data', 1, False)

import sklearn.naive_bayes as naive_bayes
import time

# feature selection goes here
#train_features = dataset_small.get_train_features()
#train_features = np.delete(train_features, 
#                           [8710, 8105], axis=1)

#test_features = dataset_small.get_test_features()
#test_features = np.delete(dataset_small.get_test_features(),
#                          [8710, 8105], axis=1)                 

num_trainings = [100, 500, 1000, 5000, 1e4, 2e4, 3e4, 45000]
num_repeats   = [10,  10,  5,    5,    1,   1,   1,   1]
fall_out = np.zeros(len(num_trainings))
recall   = np.zeros(len(num_trainings))
do_feature_selection = True

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import recall_score, roc_curve
from sklearn import svm

for i, i_num in enumerate(num_trainings):
    train_features_all = dataset_small.get_train_features()
    for i_repeat in range(0, num_repeats[i]):
        
        rand_index = np.arange(train_features_all.shape[0])
        np.random.shuffle(rand_index)
        train_features = train_features_all[rand_index,:][0:i_num,:]
        train_labels   = dataset_small.get_train_labels()[rand_index][0:i_num]
        
        if do_feature_selection:
            kbestfilter = SelectKBest(chi2,k=500)
        
            train_features = kbestfilter.fit_transform(train_features,
                                                    train_labels)
            test_features = kbestfilter.transform(dataset_small.get_test_features())
        else:
            test_features = dataset_small.get_test_features()
        
    #    train_features = np.sign(train_features)
    #    test_features  = np.sign(test_features)
        start_time = time.time()
    #    classifier = naive_bayes.BernoulliNB()
    #    classifier  = naive_bayes.GaussianNB()
    #    classifier  = naive_bayes.MultinomialNB(fit_prior=False)
        classifier  = svm.LinearSVC(verbose=2)
        spam_filter = classifier.fit(train_features,
                            train_labels)
    #    spam_probs = classifier.predict_proba(test_features)
        
        spam_pred = spam_filter.predict(test_features)
        end_time  = time.time()
        print "computational time %f" %(end_time - start_time)
        #spam_pred = spam_filter.predict(dataset_large.get_test_features(cross_validation_id))
        
        recall_nospam, recall_spam = recall_score(
            dataset_small.get_test_labels(), spam_pred, average=None)
        fall_out[i] += 1 - recall_nospam
        recall[i]   += recall_spam
    fall_out[i] /= num_repeats[i]
    recall[i]   /= num_repeats[i]
    
## save ROC curve
npsave_dir = '/home/junyic/Work/Courses/4th_year/DataSci/project1/'
np.savez(npsave_dir + 'SVC_feature_select', recall, fall_out)


## reports
import sklearn.metrics
report = sklearn.metrics.classification_report(dataset_small.get_test_labels(),
                                               spam_pred)
print report

##

## find out which words contributes most to spam detection
from pandas import DataFrame
prob_ifnospam = np.exp(bnb.feature_log_prob_[0])
prob_ifspam   = np.exp(bnb.feature_log_prob_[1])

words_vs_freq = DataFrame({'vocab': dataset_small.vocabs, 
                           'freq': prob_ifspam - prob_ifnospam,
                           'freq_spam': prob_ifspam,
                           'freq_nospam': prob_ifnospam})
words_vs_freq.sort(columns = 'freq', inplace = True, ascending=False)

## visualize emails spam/no spam
import matplotlib.pylab as plt
fig = plt.figure()
ax1  = fig.add_subplot(121)
ax1.imshow(np.sign(train_features[-500:, words_vs_freq.index]), cmap='gray')
ax1.set_xlabel('word')
ax1.set_ylabel('email')
ax1.set_xlim([0, 499])

ax2 = fig.add_subplot(122)
ax2.imshow(np.sign(train_features[-500:, words_vs_freq.index]), cmap='gray')
ax2.set_xlabel('word')
ax2.set_xlim([9579-500, 9579])

#plt.colorbar()
plt.show()
##
my_pred = np.zeros_like(spam_pred)
for i in range(0, my_pred.shape[0]):
    A = np.sum(prob_ifspam[test_words[i,:]>0])+np.sum(np.log(1-np.exp(prob_ifspam[test_words[i,:]==0])))
    B = np.sum(prob_ifnospam[test_words[i,:]>0])+np.sum(np.log(1-np.exp(prob_ifnospam[test_words[i,:]==0])))
    my_pred[i] = 1 if A > B else 0