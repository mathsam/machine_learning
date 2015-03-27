import	numpy	as	np
import	time
import	chromo_db
from	sklearn.ensemble	import	RandomForestClassifier
from	sklearn.metrics		import	classification_report
from 	sklearn.metrics		import	recall_score
from	sklearn.metrics		import	roc_curve, auc
from	sklearn.naive_bayes	import	MultinomialNB
from	sklearn			import	svm
from	sklearn.neighbors	import	KNeighborsClassifier
from	sklearn.metrics		import	roc_curve, auc

## prepare X, Y
chrome_num = 1
db = chromo_db.ChromoData(chrome_num)
train_X = db.train_X(missing_X_mode='neighbors_ave')
train_Y = db.train_Y()
valid_train_samples = ~np.isnan(train_Y)

test_X = db.test_X(missing_X_mode='neighbors_ave')
test_Y = db.test_Y()
valid_test_samples = ~np.isnan(test_Y)

## Convert to binary 
cutoff_level = 0.5
ylabel_train = train_Y[valid_train_samples] > 0.5
ylabel_train = ylabel_train.astype('float')
ylabel_test = test_Y[valid_test_samples] > 0.5
ylabel_test = ylabel_test.astype('float')

'''
## Classifier -- Random Forests
nTree = 1000
label = ['No', 'Complete']
time_start      = time.time()
clf_RF		= RandomForestClassifier(n_estimators=nTree,random_state=0)
clfRFFit	= clf_RF.fit(train_X[valid_train_samples,:], ylabel_train)
time_tr         = time.time()
pre_te_RF	= clf_RF.predict(test_X)
time_te         = time.time()
probas_		= clfRFFit.predict_proba(test_X)
fpr, tpr, thresholds		= roc_curve(ylabel_test, probas_[:,1][valid_test_samples])
roc_auc		= auc(fpr, tpr)
report		= classification_report(ylabel_test, pre_te_RF[valid_test_samples], target_names=label)
print 'Random Forests:'
print report
print "AUC: %f" % (roc_auc)
print "Training time: %f" % (time_tr - time_start)
print "Testing time: %f" % (time_te - time_tr)
print "Total time: %f" % (time_te - time_start)
'''

## Classifier -- MultinomialNB
label		= ['No', 'Complete']
time_start      = time.time()
clf_Mul		= MultinomialNB()
clfMulFit	= clf_Mul.fit(train_X[valid_train_samples,:], ylabel_train)
time_tr         = time.time()
pre_te_Mul	= clfMulFit.predict(test_X)
time_te         = time.time()
probas_		= clfMulFit.predict_proba(test_X)
fpr, tpr, thresholds		= roc_curve(ylabel_test, probas_[:,1][valid_test_samples])
roc_auc		= auc(fpr, tpr)
report		= classification_report(ylabel_test, pre_te_Mul[valid_test_samples], target_names=label)

print 'Multinomial Naive Bayes:'
print report
print "AUC: %f" % (roc_auc)
print "Training time: %f" % (time_tr - time_start)
print "Testing time: %f" % (time_te - time_tr)
print "Total time: %f" % (time_te - time_start)

'''
## Classifier -- Linear SVM
label		= ['No', 'Complete']
time_start      = time.time()
clf_LSVM	= svm.LinearSVC()
clfLSVMFit	= clf_LSVM.fit(train_X[valid_train_samples,:], ylabel_train)
time_tr         = time.time()
pre_te_LSVM	= clfLSVMFit.predict(test_X)
time_te         = time.time()
probas_		= clfLSVMFit.decision_function(test_X)
fpr, tpr, thresholds		= roc_curve(ylabel_test, probas_[valid_test_samples])
roc_auc		= auc(fpr, tpr)
report		= classification_report(ylabel_test, pre_te_LSVM[valid_test_samples], target_names=label)

print 'Linear SVM:'
print report
print "AUC: %f" % (roc_auc)
print "Training time: %f" % (time_tr - time_start)
print "Testing time: %f" % (time_te - time_tr)
print "Total time: %f" % (time_te - time_start)

## Classifier -- KNN
nNeigh		= 20
label		= ['No', 'Complete']
time_start      = time.time()

clf_KNN		= KNeighborsClassifier(n_neighbors=nNeigh)
clfKNNFit	= clf_KNN.fit(train_X[valid_train_samples,:], ylabel_train)
time_tr         = time.time()
pre_te_KNN	= clfKNNFit.predict(test_X)
time_te         = time.time()
probas_		= clfKNNFit.predict_proba(test_X)
fpr, tpr, thresholds		= roc_curve(ylabel_test, probas_[:,1][valid_test_samples])
roc_auc		= auc(fpr, tpr)
report		= classification_report(ylabel_test, pre_te_KNN[valid_test_samples], target_names=label)

print 'KNN (n=%s):' % (nNeigh)
print report
print "AUC: %f" % (roc_auc)
print "Training time: %f" % (time_tr - time_start)
print "Testing time: %f" % (time_te - time_tr)
print "Total time: %f" % (time_te - time_start)
'''
