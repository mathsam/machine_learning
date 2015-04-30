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
## see what latent structure captured by SVD
inbounds = {}
for i in range(0, btc_train_np.shape[0]):
    current_k = btc_train_np[i,1]
    if inbounds.has_key(current_k):
        inbounds[current_k].append(btc_train_np[i,0])
    else:
        inbounds[current_k] = [btc_train_np[i,0]]
num_inbounds = np.zeros(444075)
for i in range(0, 444075):
    try:
        num_inbounds[i] = len(inbounds[i])
    except KeyError:
        num_inbounds[i] = 0
##
component_num = 14
num_interactions = num_outbounds * num_inbounds
SingularVector = V.copy()
valid_index = np.logical_and(num_interactions>0, SingularVector[component_num,:]>0)
r1 = pearsonr(np.log(num_interactions[valid_index]), np.log(SingularVector[component_num,valid_index]))
print 'r1', r1[0]
valid_index = np.logical_and(num_inbounds>0, SingularVector[component_num,:]>0)
r2 = pearsonr(np.log(num_inbounds[valid_index]), np.log(SingularVector[component_num,valid_index]))
print 'r2', r2[0]