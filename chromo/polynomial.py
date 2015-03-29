## polynomial preprocessing
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
valid_train_samples = ~np.isnan(train_Y)
train_X = train_X[valid_train_samples,:]
train_X = poly.fit_transform(train_X)[:,1:]
train_Y = train_Y[valid_train_samples]

##
import time
from sklearn import linear_model
start_time = time.time()

clf = linear_model.LinearRegression()
clf.fit(train_X, train_Y)

batch_num = 10000
predicted_Y = np.zeros(test_X.shape[0])
for i in range(0, test_X.shape[0], batch_num):
    train_X_poly = poly.fit_transform(test_X[i:i+batch_num,:])[:,1:]
    predicted_Y[i:i+batch_num] = clf.predict(train_X_poly)

end_time = time.time()
print "predition time %f" %(end_time - start_time)