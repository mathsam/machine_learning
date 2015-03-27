# evaulate regression performance
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error

valid_test_samples = ~np.isnan(test_Y)
y_true = test_Y[valid_test_samples]
y_pred = predicted_Y[valid_test_samples]
print "r2 = %f" %r2_score(y_true, y_pred)
#print "explained variance is %f" %explained_variance_score(y_true, y_pred)
print "SMSE = %f" %np.sqrt(mean_squared_error(y_true, y_pred))

## evaulate classification performance
import sklearn.metrics
cutoff_level = 0.5
ylabel_pred = y_pred > cutoff_level
ylabel_true = y_true > cutoff_level
report = sklearn.metrics.classification_report(ylabel_true, ylabel_pred)
print report

auc = sklearn.metrics.roc_auc_score(ylabel_true, y_pred)
print "AUC = %f" %auc

fpr, tpr, thresholds = sklearn.metrics.roc_curve(ylabel_true, y_pred, pos_label=1)
plt.plot(fpr, tpr)
plt.show()