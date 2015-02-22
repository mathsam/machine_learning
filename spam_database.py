import numpy as np
import pandas as pd
from email_process import read_bagofwords_dat
import os

def find_abspath(dir, substr):
    """
    given a directory, find the file the directory whose filename contains
    `substr`
    """
    from os.path import join
    files = os.listdir(dir)
    for ifile in files:
        if substr in ifile:
            return join(dir, ifile)
    raise ValueError("substr does not exist in all strs")

class DataSet(object):
    """Provide cross-validation dataset for spam/nonspam
    
    Train and test datasets are accessed as object attributes, but they are
    actually implemented as descriptors.
    """
    def __init__(self, datadir, num_partitions = 5, shuffle = False):
        """
        If `num_partitions`=1, use default partion between Train and Test. 
        Otherwise combine Train and Test together and them partion them into
        `num_partitions` partions, and then turn them into `num_partitions` 
        train/test pairs.
        
        Args:
            datadir: str for where /Test, /Train directories are stored
            num_partitions: 1 partition is used as test, num_partitions-1 are
                            used as train
            shuffle: if randomly shuffle the dataset
        """
        from os.path import join
        self._datadir = datadir
        self.num_partitions = num_partitions
        self._shuffle = shuffle
        
        train_dir = join(datadir, 'Train')
        test_dir  = join(datadir, 'Test')
        
        train_bagofwords, train_numspams, train_numnospams = \
            DataSet.read_datfile(train_dir)
        test_bagofwords, test_numspams, test_numnospams = \
            DataSet.read_datfile(test_dir)
        self._orig_numtrains = train_numspams + train_numnospams
        self._orig_numtests  = test_numspams  + test_numnospams
        self._bagofwords = np.concatenate((train_bagofwords, test_bagofwords),0)
        
        train_labels_file = find_abspath(train_dir, 'classes')
        test_labels_file  = find_abspath(test_dir,  'classes')
        
        train_labels = pd.read_csv(train_labels_file, header=None, true_values=['Spam'], false_values=['NotSpam'])[0]
        train_labels = np.array(train_labels).astype(int)
        
        test_labels  = pd.read_csv(test_labels_file, header=None, true_values=['Spam'], false_values=['NotSpam'])[0]
        test_labels  = np.array(test_labels).astype(int)
        
        self._labels = np.concatenate((train_labels, test_labels), 0)
        
        vocab_file = find_abspath(train_dir, 'vocab')
        vocabs = pd.read_csv(vocab_file, header=None, index_col=False)[0]
        self.vocabs = np.array(vocabs)
        
        if self._labels.shape[0] != self._bagofwords.shape[0]:
            raise ValueError("labels and bagofwords does not match")
            
        total_cases = self._labels.shape[0]
            
        if total_cases%num_partitions != 0 and num_partitions != 1:
            raise ValueError("Dataset cannot be equally partioned into %d parts"
                              %num_partitions)
        self.total_cases = total_cases
        
        if num_partitions != 1:
            self.num_trains = total_cases - total_cases/num_partitions                              
            self.num_tests  = total_cases - self.num_trains
        else:
            self.num_trains = train_numspams + train_numnospams
            self.num_tests  = test_numspams  + test_numnospams
                              
        if shuffle:
            shuffled_index = np.arange(0, self._labels.shape[0])
            np.random.shuffle(shuffled_index)
            self._bagofwords = self._bagofwords[shuffled_index]
            self._labels     = self._labels[shuffled_index]
            
    def get_test_features(self, test_id=0):
        """
        get the `test_id`th features
        
        Args:
            test_id: int, must less than `num_partitions`
                      starting from 0
        
        Returns:
            features: a numpy array with shape (num_cases, num_features)
            
        For example. num_partitions = 5, when test_id = 0, returns the 5th 
        partition as the test
        """
        if self.num_partitions == 1:
            return self._bagofwords[self.num_trains:,:]
        elif test_id >= self.num_partitions:
            raise ValueError("test_id exceed num_partitions")
        group_id = self.num_partitions - test_id - 1
        start_index = (self.total_cases/self.num_partitions)*group_id
        end_index   = start_index + (self.total_cases/self.num_partitions)
        return self._bagofwords[start_index:end_index,:]

    def get_test_labels(self, test_id=0):
        """
        get the `test_id`th labels
        
        Args:
            test_id: int, must less than `num_partitions`
                      starting from 0
        
        Returns:
            features: a numpy array with shape (num_cases,)
            
        For example. num_partitions = 5, when test_id = 0, returns the 5th
        partitions as test
        """
        if self.num_partitions == 1:
            return self._labels[self.num_trains:]
        elif test_id >= self.num_partitions:
            raise ValueError("test_id exceed num_partitions")
        group_id = self.num_partitions - test_id - 1
        start_index = (self.total_cases/self.num_partitions)*group_id
        end_index   = start_index + (self.total_cases/self.num_partitions)
        return self._labels[start_index:end_index]
        
    def get_train_features(self, train_id=0):
        """
        get the `test_id`th features
        
        Args:
            test_id: int, must less than `num_partitions`
                      starting from 0
        
        Returns:
            features: a numpy array with shape (num_cases, num_features)
            
        For example. num_partitions = 5, when test_id = 0, returns the 1 to 4
        partitions as the test
        """
        if self.num_partitions == 1:
            return self._bagofwords[:self.num_trains,:]
        elif train_id >= self.num_partitions:
            raise ValueError("test_id exceed num_partitions")
        group_index = np.ones(self.total_cases, dtype=bool)
        group_id = self.num_partitions - train_id - 1
        start_index = (self.total_cases/self.num_partitions)*group_id
        end_index   = start_index + (self.total_cases/self.num_partitions)
        group_index[start_index:end_index] = False
        return self._bagofwords[group_index,:]

    def get_train_labels(self, train_id=0):
        """
        get the `test_id`th labels
        
        Args:
            test_id: int, must less than `num_partitions`
                      starting from 0
        
        Returns:
            features: a numpy array with shape (num_cases)
            
        For example. num_partitions = 5, when test_id = 0, returns the 1 to 4
        partitions as the test
        """
        if self.num_partitions == 1:
            return self._labels[:self.num_trains]
        elif train_id >= self.num_partitions:
            raise ValueError("test_id exceed num_partitions")
        group_index = np.ones(self.total_cases, dtype=bool)
        group_id = self.num_partitions - train_id - 1
        start_index = (self.total_cases/self.num_partitions)*group_id
        end_index   = start_index + (self.total_cases/self.num_partitions)
        group_index[start_index:end_index] = False
        return self._labels[group_index]
        
    @staticmethod
    def read_datfile(datfile_dir):
        """
        read bag_of_words.data file given its directory
        
        Returns:
            A tuple a numpy array and two integers: 
            (bagofwords, numspams, numnospams)
        """
        datfile = find_abspath(datfile_dir, 'bag_of_words')
        
        #find out how many spam and nospam emails in Train
        spam_file   = find_abspath(datfile_dir, 'Spam')
        nospam_file = find_abspath(datfile_dir, 'NotSpam')
        numspams   = len(os.listdir(spam_file))
        numnospams = len(os.listdir(nospam_file))
        bagofwords = read_bagofwords_dat(datfile, 
                                         numnospams + numspams)
        return bagofwords, numspams, numnospams                                        