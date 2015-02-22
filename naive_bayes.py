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

import sklearn.naive_bayes as naive_bayes
bnb = naive_bayes.BernoulliNB()
#gnb  = naive_bayes.GaussianNB()
cross_validation_id = 0
spam_filter = bnb.fit(np.sign(dataset_small.get_train_features(cross_validation_id)), 
                      dataset_small.get_train_labels(cross_validation_id))
#spam_filter = gnb.fit(train_words, labels)

spam_pred = spam_filter.predict(dataset_small.get_test_features(cross_validation_id))

##
pred_rate = float(np.sum(spam_pred == dataset_small.get_test_labels(cross_validation_id), 0))/float(dataset_small.num_tests)
print pred_rate

## find out which words contributes most to spam detection
prob_ifnospam = bnb.feature_log_prob_[0]
prob_ifspam   = bnb.feature_log_prob_[1]
#high_spamprob_index = (prob_ifspam-prob_ifnospam)>0.1
##
my_pred = np.zeros_like(spam_pred)
for i in range(0, my_pred.shape[0]):
    A = np.sum(prob_ifspam[test_words[i,:]>0])+np.sum(np.log(1-np.exp(prob_ifspam[test_words[i,:]==0])))
    B = np.sum(prob_ifnospam[test_words[i,:]>0])+np.sum(np.log(1-np.exp(prob_ifnospam[test_words[i,:]==0])))
    my_pred[i] = 1 if A > B else 0