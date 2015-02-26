"""
do feature selection
"""
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

kbestfilter = SelectKBest(chi2,k=500)

train_features = kbestfilter.fit_transform(dataset_small.get_train_features(),
                     dataset_small.get_train_labels())
test_features = kbestfilter.transform(dataset_small.get_test_features())                     


##
threshold = 0.8*(1-0.8)
sel_var = VarianceThreshold(threshold = threshold)
sel_var.fit(np.sign(dataset_small.get_train_features()))

train_selected_features = sel_var.transform(dataset_small.get_train_features())
test_selected_features = sel_var.transform(dataset_small.get_test_features())

## train naive bayes
import sklearn.naive_bayes as naive_bayes
bnb = naive_bayes.BernoulliNB()

spam_filter = bnb.fit(np.sign(train_selected_features), 
                      dataset_small.get_train_labels())
spam_pred   = spam_filter.predict(test_selected_features)

## evaluate goodness of prediction
import sklearn.metrics
report = sklearn.metrics.classification_report(dataset_small.get_test_labels(),
                                     spam_pred)