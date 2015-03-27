# linear regression

## prepare X, Y
import chromo_db
chrome_num = 1
db = chromo_db.ChromoData(chrome_num)
train_X = db.train_X(missing_X_mode='neighbors_ave')
train_Y = db.train_Y()

test_X = db.test_X(missing_X_mode='neighbors_ave')
test_Y = db.test_Y()

## Ordinary linear regression
from sklearn import linear_model
import time

start_time = time.time()
num_tests = 10
for i in range(num_tests):
    clf = linear_model.LinearRegression()
    valid_train_samples = ~np.isnan(train_Y)
    clf.fit(train_X[valid_train_samples,:], train_Y[valid_train_samples])
    
    valid_test_samples = ~np.isnan(test_Y)
    predicted_Y = clf.predict(test_X)
end_time = time.time()
print "predition time %f" %((end_time - start_time)/num_tests)
pred_score = clf.score(test_X[valid_test_samples,:], test_Y[valid_test_samples])

## OLR but using log(1-X) to predict log(1-Y)
import scipy.stats
valid_train_samples = ~np.isnan(train_Y)
start_time = time.time()
num_tests = 1
for i in range(num_tests):
    clf.fit(np.log(1.-train_X[valid_train_samples,:]), 
            np.log(1.-train_Y[valid_train_samples]))
    
    valid_test_samples = ~np.isnan(test_Y)
    predicted_log1mY = clf.predict(np.log(1-test_X))
    predicted_Y = 1. - np.exp(predicted_log1mY)
end_time = time.time()
print "predition time %f" %((end_time - start_time)/num_tests)