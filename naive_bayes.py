"""
Naive Bayes classfier
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/junyic/Work/Courses/4th_year/DataSci/project1/src')
from email_process import read_bagofwords_dat
import matplotlib.pyplot as plt

archive_dir = '/home/junyic/Work/Courses/4th_year/DataSci/project1/trec07p_data/'
train_dir = archive_dir + '/Train/'
test_dir  = archive_dir + '/Test/'

train_words = read_bagofwords_dat(train_dir + 'train_emails_bag_of_words_200.dat', 22500*2)

test_words  = read_bagofwords_dat(test_dir + 'test_emails_bag_of_words_0.dat', 2500*2)

##
labels = pd.read_csv(train_dir + 'train_emails_classes_200.txt', header=None, true_values=['NotSpam'], false_values=['Spam'])
labels = np.array(labels).astype(int)
labels.shape = (labels.shape[0],)

test_labels = pd.read_csv(test_dir + 'test_emails_classes_0.txt', header=None, true_values=['NotSpam'], false_values=['Spam'])
test_labels = np.array(test_labels).astype(int)
test_labels.shape = (test_labels.shape[0],)

vocabs = pd.read_csv(train_dir + 'train_emails_vocab_200.txt', header=None,index_col=False)
##
import sklearn.naive_bayes as naive_bayes
bnb = naive_bayes.BernoulliNB()
#gnb  = naive_bayes.GaussianNB()
spam_filter = bnb.fit(np.sign(train_words), labels)
#spam_filter = gnb.fit(train_words, labels)

spam_pred = spam_filter.predict(test_words)

##
pred_rate = np.sum((spam_pred == test_labels).astype(int), 0)/5000.0
print pred_rate