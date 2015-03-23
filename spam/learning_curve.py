"""
plot learning curves for different classifier
"""
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
npsave_dir = '/home/junyic/Work/Courses/4th_year/DataSci/project1/'
classifier_set    = ['NBB', 'NBGaussian', 'NBMultinomial', 'SVC']
classifier_labels = ['BNB', 'GNB', 'MNB', 'Linear SVM']
colors = ['-b', '-r', '-g', '-k']

num_trainings = [100, 500, 1000, 5000, 1e4, 2e4, 3e4, 45000]

plt.clf()
for i, which_classifier in enumerate(classifier_set):
    recalls = np.load(npsave_dir + which_classifier + '.npz')
    plt.plot(num_trainings, recalls['arr_0'], colors[i]+ 's',
             label=classifier_labels[i])
    recalls = np.load(npsave_dir + which_classifier + '_feature_select.npz')
    plt.plot(num_trainings, recalls['arr_0'], '-' + colors[i] + 's')
    
plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.legend(loc='best')
plt.xlabel('Num of training emails')
plt.ylabel('Recall')
plt.show()             