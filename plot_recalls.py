"""
plot ROC curves for different classifier
"""
npsave_dir = '/home/junyic/Work/Courses/4th_year/DataSci/project1/'
classifier_set    = ['NBB', 'NBGaussian', 'NBMultinomial', 'SVC']
classifier_labels = ['BNB', 'GNB', 'MNB', 'Linear SVM']
colors = ['-ob', '-xr', '-sg', '-dk']
num_trainings = [100, 500, 1000, 5000, 1e4, 2e4, 3e4, 45000]

plt.clf()
for i, which_classifier in enumerate(classifier_set):
    recalls = np.load(npsave_dir + which_classifier + '.npz')
    plt.plot(num_trainings, recalls['arr_0'], colors[i], 
             label=classifier_labels[i])
    recalls = np.load(npsave_dir + which_classifier + '_feature_select.npz')
    plt.plot(num_trainings, recalls['arr_0'], '-' + colors[i])
    
plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.legend(loc='best')
plt.xlabel('num of training emails')
plt.ylabel('recall')
plt.show()             