import sys
sys.path.append('/Users/yangxiu/Dropbox/Courses/COS424/HW1/')
sys.path.append('/Users/yangxiu/Dropbox/Courses/COS424/HW1/src/machine_learning/')
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
from email_process import *

def disp_feature(column, threshold=5, maxcount=10):
    if np.sum(column) <= threshold:
        return 0, 0
    spamcount = [0]*maxcount
    nonspamcount = [0]*maxcount
    for i in range(len(column)):
        count = column[i]
        if i >= 22500:
            if count >= maxcount:
                spamcount[maxcount-1] += 1
            elif count > 0:
                spamcount[count] += 1
        else:
            if count >= maxcount:
                nonspamcount[maxcount-1] += 1
            elif count > 0:
                nonspamcount[count] += 1

    return spamcount, nonspamcount

def main(argv):
    feat = read_bagofwords_dat("../../trec07p_data/Train/_bag_of_words_200.dat", 45000)
    vocab = pd.read_csv("../../trec07p_data/Train/_vocab_200.txt", header = None)
    vocab = np.array(vocab)
    n = raw_input("Press enter:")
    while n.strip() != 'q':
        colnum = rd.random() * feat.shape[1]
        #plt.hist(feat[:,colnum])
        #plt.show()
        maxcount = 20
        spamcount, nonspamcount = disp_feature(feat[:,colnum], 10, maxcount)

        x = list(range(1, maxcount))
        if spamcount != 0 and nonspamcount != 0:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)

            ax1.scatter(x, spamcount[1:], color='red', marker='s', label='spam')
            ax1.scatter(x, nonspamcount[1:], color='blue', marker='o', label='nonspam')

            plt.title(vocab[colnum][0])
            plt.show()
        n = raw_input("Press enter:")



if __name__ == "__main__":
    main("")
    
