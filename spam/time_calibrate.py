"""
test run time in order to calibrate different machines to produce a consistent
run time summary
"""
import sklearn.naive_bayes as naive_bayes
import numpy as np

import time
start_time = time.time()
classifier = naive_bayes.BernoulliNB()
train_features = np.random.randint(0,2,(10000, 5000))
train_labels   = np.random.randint(0,2, 10000)
test_features  = np.random.randint(0,2,(5000, 5000))
spam_filter = classifier.fit(train_features,
                             train_labels)
spam_pred = spam_filter.predict(test_features)
end_time  = time.time()

print "run time is %f", end_time-start_time
                         