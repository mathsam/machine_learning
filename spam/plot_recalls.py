"""
plot ROC curves for different classifier
"""
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
npsave_dir = '/home/junyic/Work/Courses/4th_year/DataSci/project1/'
classifier_set    = ['NBB', 'NBGaussian', 'NBMultinomial', 'SVC']
classifier_labels = ['BNB', 'GNB', 'MNB', 'Linear SVM']
colors = ['-b', '-r', '-g', '-k']

plt.clf()
for i, which_classifier in enumerate(classifier_set):
    recalls = np.load(npsave_dir + which_classifier + '.npz')
    plt.plot(recalls['arr_1'], recalls['arr_0'], colors[i], 
             label=classifier_labels[i]+ ' (area=%0.4f)' %auc(recalls['arr_1'], recalls['arr_0']))
    recalls = np.load(npsave_dir + which_classifier + '_feature_select.npz')
    plt.plot(recalls['arr_1'], recalls['arr_0'], '-' + colors[i], label=classifier_labels[i]+' (area=%0.4f)' %auc(recalls['arr_1'], recalls['arr_0']))
    
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.legend(loc='best')
plt.xlabel('False positve rate')
plt.ylabel('True positive rate')
plt.show()             