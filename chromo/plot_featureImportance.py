### Plot feature importances

importances= clf_RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_RF.estimators_], axis=0)
indices = np.argsort(importances)

plt.figure(figsize=(5.5,8.5))
plt.title("Random Forest Regressor", fontsize=18)
plt.barh(range(33), importances[indices], color="r", align="center")
#plt.barh(range(33), importances[indices], color="r", xerr=std[indices], align="center")
plt.yticks(range(33), indices)
plt.ylim([17.5, 32.5])
plt.ylabel('Feature number',fontsize=18)
plt.savefig('./feature_importance_RFR.png', format='PNG')
plt.show()
