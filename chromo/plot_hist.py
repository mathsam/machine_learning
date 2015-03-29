import matplotlib.pyplot as plt

valid_test_samples = ~np.isnan(test_Y)
y_true = test_Y[valid_test_samples]
y_pred = probas_[:,1][valid_test_samples]

plt.figure(figsize=(10,6))
plt.hist(y_pred,80,normed=True,alpha=0.5,label='prediction')
plt.hist(y_true,80,normed=True,alpha=0.5,label='True')
plt.xlabel('$\\beta$')
plt.ylabel('Normalized frequency')
plt.title('Random Forest')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig('./hist_RF.png',format='PNG')
plt.show()

