import os
import pandas as pd
import numpy as np

class ChromoData(object):
    _data_path = '/home/junyic/Work/Courses/4th_year/DataSci/project2/data/data'
    
    def __init__(self, chromonum):
        if not isinstance(chromonum, int):
            raise TypeError('enter an integer from 1 to 22')
        self.trainfile  = 'intersected_final_chr%d_cutoff_20_train_revised.bed' %chromonum
        self.samplefile = 'intersected_final_chr%d_cutoff_20_sample.bed' %chromonum
        self.testfile   = 'intersected_final_chr%d_cutoff_20_test.bed' %chromonum
        
    def train_X(self, missing_X_mode='raw', include_strand=False):
        train_pd = pd.read_csv(os.path.join(ChromoData._data_path,
                                            self.trainfile),
                               sep='\t', header=None,
                               true_values=['-'], false_values=['+'])
        if include_strand:
            train_np = np.array(train_pd.iloc[:, 3:-1]).astype(float) 
            #last column is whether presented in 450k chip
        else:
            train_np = np.array(train_pd.iloc[:, 4:-1]).astype(float)
        if missing_X_mode == 'raw':
            return train_np
        if missing_X_mode == 'neighbors_ave':
            nan_is, nan_js = np.where(np.isnan(train_np))
            print "%d elements are missing out of a total of %d" %(
                nan_is.shape[0], train_np.shape[0]*train_np.shape[1])
            filled_values = np.zeros_like(nan_is, dtype=float)
            for k in range(0, nan_is.shape[0]):
                i = nan_is[k]
                j = nan_js[k]
                ave_width = 1
                while True:
                    ave = np.nanmean(
                        train_np[np.max([0, i-ave_width]):i+ave_width+1, 
                             np.max([0, j-ave_width]):j+ave_width+1].flatten())
                    if not np.isnan(ave):
                        break
                    ave_width += 1
                filled_values[k] = ave
            train_np[nan_is, nan_js] = filled_values
            return train_np
        
    def train_Y(self):
        pass
        
    def test_X(self):
        pass
        
    def test_Y(self):
        pass