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
train_features = np.sign(train_features)
test_features  = np.sign(test_features)
start_time = time.time()
bnb = naive_bayes.BernoulliNB()
#gnb  = naive_bayes.GaussianNB()
#mnb  = naive_bayes.MultinomialNB(fit_prior=False)
spam_filter = bnb.fit(train_features, 
                      dataset_small.get_train_labels())
#spam_filter = gnb.fit(train_words, labels)
#spam_filter = mnb.fit(train_features, 
#                      dataset_small.get_train_labels())
#spam_filter = gnb.fit(train_features, 
#                      dataset_small.get_train_labels())

spam_pred = spam_filter.predict(test_features)
end_time  = time.time()
print "computational time %f" %(end_time - start_time)
#spam_pred = spam_filter.predict(dataset_large.get_test_features(cross_validation_id))

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