x = np.arange(0, 33)
plt.plot(x, olr_coeff, '--o', label='OLR')
plt.plot(x, ridge_coef, '--o', label='Ridge')
plt.plot(x, lassoaic_coef, '--o', label='Lasso AIC')
plt.plot(x, lassobic_coef, '--o', label='Lasso BIC')
plt.plot(x, lasscv_coef, '--o', label='Lasso CV')
plt.legend(loc='best')
plt.show()
