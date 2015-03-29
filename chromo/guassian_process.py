from sklearn import gaussian_process
gp = gaussian_process.GaussianProcess(storage_mode='full')


import time
start_time = time.time()
valid_train_samples = ~np.isnan(train_Y)
lasso_selected_features = np.array([False, False, False,  True, False,  True,  True,  True,  True,
        True,  True,  True,  True,  True, False, False, False, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
       False, False,  True, False, False,  True,  True])
gp.fit(train_X[np.ix_(valid_train_samples,lasso_selected_features)], train_Y[valid_train_samples])

valid_test_samples = ~np.isnan(test_Y)
predicted_Y = gp.predict(test_X[:,lasso_selected_features], batch_size=2000)
zeros_index = np.where(predicted_Y == 0)[0]
predicted_Y_more = gp.predict(test_X[np.ix_(zeros_index,lasso_selected_features)])
predicted_Y[zeros_index] = predicted_Y_more
end_time = time.time()
print "predition time %f" %(end_time - start_time)