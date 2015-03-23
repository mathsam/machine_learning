import chromo_db
chrome_num = 1
db = chromo_db.ChromoData(chrome_num)
train_data = db.train_X(missing_X_mode='neighbors_ave')
## visualize raw train data in 2d
import matplotlib
font = {'size'   : 22}
matplotlib.rc('font', **font)

x_width = 30
feature_num = train_data.shape[1]
fig = plt.figure(figsize=[10,3])
ax_common = fig.add_subplot(111)
ax1 = fig.add_subplot(131)
x1start = np.random.randint(0, train_data.shape[0])
x1 = np.arange(x1start, x1start+x_width)
ax1.imshow(train_data[x1,:], interpolation='none',
           extent=[0, feature_num-1, x1start+x_width-1, x1start])
ax1.set_ylabel('position number')

ax2 = fig.add_subplot(132)
x2start = np.random.randint(0, train_data.shape[0])
x2 = np.arange(x2start, x2start+x_width)
ax2.imshow(train_data[x2,:], interpolation='none',
           extent=[0, feature_num-1, x2start+x_width-1, x2start])

ax3 = fig.add_subplot(133)
x3start = np.random.randint(0, train_data.shape[0])
x3 = np.arange(x3start, x3start+x_width)
ax3.imshow(train_data[x3,:], interpolation='none',
           extent=[0, feature_num-1, x3start+x_width-1, x3start])

ax_common.spines['top'].set_color('none')
ax_common.spines['bottom'].set_color('none')
ax_common.spines['left'].set_color('none')
ax_common.spines['right'].set_color('none')
ax_common.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax_common.set_xlabel('beta for each sample')

plt.show()

## visualize the 2d lagged correlation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure(figsize=(8,6))

ax = fig.gca(projection='3d')
#X, Y = np.meshgrid(np.arange(-5, 6), np.arange(-20, 21))
Z = r.copy()

surf = ax.plot_surface(X, Y, Z, rstride=4, cstride=1, alpha=0.3, linewidth=0.5)
cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-5, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='y', offset=20, cmap=cm.coolwarm)
#plt.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('sample lag')
ax.set_ylabel('position lag')
ax.set_zlabel('correlation')
plt.show()

##
ax.set_xlim3d(-pi, 2*pi);
ax.set_ylim3d(0, 3*pi);
ax.set_zlim3d(-pi, 2*pi);