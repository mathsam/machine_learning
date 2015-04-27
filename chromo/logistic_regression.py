# Logistic regression
from sklearn.linear_model import LogisticRegression
import time
start_time = time.time()
num_tests = 1
for i in range(num_tests):
    logistic = LogisticRegression(solver='lbfgs')
    valid_train_samples = ~np.isnan(train_Y)
    logistic.fit(train_X[valid_train_samples,:], train_Y[valid_train_samples])
    batch_num = 10000
    predicted_Y = np.zeros(test_X.shape[0])
    for i in range(0, test_X.shape[0], batch_num):
        predicted_Y[i:i+batch_num] = logistic.predict(test_X[i:i+batch_num,:])
end_time = time.time()
print "predition time %f" %((end_time - start_time)/num_tests)