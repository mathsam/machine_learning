import pandas as pd
import scipy.sparse
test_data = '/home/junyic/Work/Courses/4th_year/DataSci/project3/testTriplets.txt'

btc_test_pd = pd.read_csv(test_data, header=None, sep = ' ')
btc_test_np = np.array(btc_test_pd)
btc_test_label = btc_test_np[:,2]

num_tests = btc_test_np[:,2].shape[0]
btc_pred = np.zeros(num_tests)
empirical = np.zeros(num_tests)

for i in range(num_tests):
    i_indx = btc_test_np[i,0]
    j_indx = btc_test_np[i,1]
    btc_pred[i] = U[i_indx,:].dot(V[:,j_indx])
    try:
        empirical[i] = np.log(len(outbounds[i_indx])*len(outbounds[j_indx]))
    except KeyError:
        empirical[i] = 0.
    
##
num_outbounds = np.zeros(444075)
for i in range(0, 444075):
    try:
        num_outbounds[i] = len(outbounds[i])
    except KeyError:
        num_outbounds[i] = 0
##
btc_pred[btc_pred<0]= 0

## trunicate the values and make them into probability
min_value = np.min(btc_pred)
max_value = np.max(btc_pred)
btc_pred = (btc_pred-min_value)/(max_value-min_value)



## ROC curve
import sklearn.metrics

fpr, tpr, thresholds = sklearn.metrics.roc_curve(btc_test_np[:,2], np.log(btc_pred), pos_label=1)
auc = sklearn.metrics.roc_auc_score(btc_test_np[:,2], btc_pred)
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Area under curve = %3f' %auc)
plt.show()

## histogram of predicted probability
tp_indx = np.logical_and(btc_test_label==1, btc_pred > 0.1106)
tn_indx = np.logi
plt.hist(btc_pred[btc_test_np[:,2]==0])
plt.gca().set_yscale('log')
plt.show()
##
max_indx = np.where((1-fpr)*tpr == np.max((1-fpr)*tpr))[0][0]
thresholds[max_indx]