# Logistic regression
from sklearn.linear_model import LogisticRegression
import time
start_time = time.time()
num_tests = 1
for i in range(num_tests):
    logistic = LogisticRegression()
    logistic.fit(train_X, train_Y)
    predicted_Y = logistic.predict_proba(test_X)
end_time = time.time()
print "predition time %f" %((end_time - start_time)/num_tests)