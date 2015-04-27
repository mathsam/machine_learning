import pandas as pd
import scipy.sparse
train_data = '/home/junyic/Work/Courses/4th_year/DataSci/project3/txTripletsCounts.txt'

btc_train_pd = pd.read_csv(train_data, header=None, sep = ' ')
btc_train_np = np.array(btc_train_pd)

## create spare matrix to represent bit coin transactions
prob = np.ones_like(btc_train_np[:,2])

btc_train_sm = scipy.sparse.coo_matrix((prob, (btc_train_np[:,0], btc_train_np[:,1])), dtype=np.int32, shape=(444075, 444075))

## SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=15, random_state = 50)
svd.fit(btc_train_sm)
U = svd.transform(btc_train_sm)
V = svd.components_

## explained variance ratio
explained_variance_ratio = svd.explained_variance_ratio_
k = np.arange(1, explained_variance_ratio.size+1)
plt.plot(k, explained_variance_ratio, '-o')
plt.xlabel('Components')
plt.ylabel('Explained variance ratio')
plt.show()
## 