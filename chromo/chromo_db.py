import os
import pandas as pd
import numpy as np

class ChromoData(object):
    _data_path = './data/data'
    
    def __init__(self, chromonum):
        if not isinstance(chromonum, int):
            raise TypeError('enter an integer from 1 to 22')
        self.trainfile  = 'intersected_final_chr%d_cutoff_20_train_revised.bed' %chromonum
        self.samplefile = 'intersected_final_chr%d_cutoff_20_sample.bed' %chromonum
        self.testfile   = 'intersected_final_chr%d_cutoff_20_test.bed' %chromonum
        
        train_Y_pd = pd.read_csv(os.path.join(ChromoData._data_path,
                                              self.samplefile),
                                 sep='\t', header=None)
        whether_in_450chip = np.array(train_Y_pd.iloc[:,-1])
        beta_Y = np.array(train_Y_pd.iloc[:,-2])
        self._which_to_predict = np.logical_and(np.isnan(beta_Y),
                                                whether_in_450chip == 0)
        
    def train_X(self, missing_X_mode='raw', include_strand=False):
        """
        Return the features train_np array for trainning
        train_np has shape (num_samples, num_features)
        
        Args:
            missing_X_mode: 'raw'|'neighbors_ave'
                'raw': does not deal with missing data (nan)
                'neighbors_ave': fill in the missing data by averaged value from
                                 it neighbors
            include_strand: True|False
                whether or not include strand indicator (1 for '-', 0 for '+')
                as the first column
                
        Returns:
            train_np: a numpy array
        """
        try:
            return self.__dict__[missing_X_mode + str(include_strand)].copy()
        except KeyError:
            self.train_pd = pd.read_csv(os.path.join(ChromoData._data_path,
                                                self.trainfile),
                                sep='\t', header=None,
                                true_values=['-'], false_values=['+'])
            if include_strand:
                train_np = np.array(self.train_pd.iloc[:, 3:-1]).astype(float) 
                #last column is whether presented in 450k chip
            else:
                train_np = np.array(self.train_pd.iloc[:, 4:-1]).astype(float)
            if missing_X_mode == 'raw':
                self.__dict__[missing_X_mode + str(include_strand)] = train_np
                return train_np.copy()
            if missing_X_mode == 'neighbors_ave':
                nan_is, nan_js = np.where(np.isnan(train_np))
                print "%d elements are missing out of a total of %d" %(
                    nan_is.shape[0], train_np.shape[0]*train_np.shape[1])
                print "now fill in the missing value using its neighbours"
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
                self.__dict__[missing_X_mode + str(include_strand)] = train_np
                return train_np.copy()
        
    def train_Y(self):
        train_Y_pd = pd.read_csv(os.path.join(ChromoData._data_path,
                                              self.samplefile),
                                 sep='\t', header=None)
        train_Y_np = np.array(train_Y_pd.iloc[:, 4])
        return train_Y_np
        
    def test_X(self, missing_X_mode='raw', include_strand=False):
        train_X = self.train_X(missing_X_mode, include_strand)
        return train_X[self._which_to_predict,:]
        
    def test_Y(self):
        test_Y_pd = pd.read_csv(os.path.join(ChromoData._data_path,
                                             self.testfile),
                                 sep='\t', header=None)
        test_Y_np = np.array(test_Y_pd.iloc[:,4])
        return test_Y_np[self._which_to_predict]

class FeatureExtend(ChromoData):
    
    def __init__(self, chromonum):
        ChromoData.__init__(self, chromonum)

    def train_X_extend(self, missing_X_mode='raw', include_strand=False, extend_mode='neighbour'):
        """
        Extend the features for trainning
        main features are inherited from Class ChromoData 
        
        Args:
            extend_mode: neighbour|distance|neighbour_distance
                         whether or not include neighbouring sites, distance in the features
        """
        train_X_normal = self.train_X(missing_X_mode, include_strand)
        self.up_stream = train_X_normal[:-2]
        self.dw_stream = train_X_normal[2:]
        self.distance = np.array(self.train_pd.iloc[1:,1])-np.array(self.train_pd.iloc[:-1,1])

        if extend_mode == 'neighbour':
            return np.column_stack([self.up_stream, train_X_normal[1:-1], self.dw_stream])

        if extend_mode == 'distance':
            return np.column_stack([self.distance[:-1], train_X_normal[1:-1], self.distance[1:]])

        if extend_mode == 'neighbour_distance':
            return np.column_stack([self.up_stream, self.distance[:-1], train_X_normal[1:-1], self.distance[1:], self.dw_stream])

    def test_X_extend(self, missing_X_mode='raw', include_strand=False, extend_mode='neighbour'):
        train_X_extend = self.train_X_extend(missing_X_mode, include_strand, extend_mode)
	print self._which_to_predict.shape
	return train_X_extend[self._which_to_predict[1:-1],:]
