# linear regression with regulization
## ridge regression
from sklearn import linear_model
start_time = time.time()
num_tests = 10
for i in range(num_tests):
    clf = linear_model.RidgeCV(alphas=[i/100.0 for i in range(1, 100)])
    
    valid_train_samples = ~np.isnan(train_Y)
    clf.fit(train_X[valid_train_samples,:], train_Y[valid_train_samples])
    
    valid_test_samples = ~np.isnan(test_Y)
    predicted_Y = clf.predict(test_X)

end_time = time.time()
print "predition time %f" %((end_time - start_time)/num_tests)

print "alpha = %f" %clf.alpha_

## Lasso BIC
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

num_tests = 10
start_time = time.time()
for i in range(num_tests):
    model_bic = LassoLarsIC(criterion='bic')
    valid_train_samples = ~np.isnan(train_Y)
    model_bic.fit(train_X[valid_train_samples,:], train_Y[valid_train_samples])
    predicted_Y = model_bic.predict(test_X)

end_time = time.time()
print "predition time %f" %((end_time - start_time)/num_tests)
## Lasso AIC
num_tests = 10
start_time = time.time()
for i in range(num_tests):
    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(train_X[valid_train_samples,:], train_Y[valid_train_samples])
    predicted_Y = model_aic.predict(test_X)

end_time = time.time()
print "predition time %f" %((end_time - start_time)/num_tests)
##
plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend(loc='best')
plt.show()
##
def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.semilogx((alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline((alpha_), color=color, linewidth=3,
                label='alpha: %s estimate = %3e' % (name, alpha_))
    plt.xlabel('(alpha)')
    plt.ylabel('criterion')
    
## Lasso CV
num_tests = 10
start_time = time.time()
for i in range(num_tests):
    model = LassoCV(cv=20, eps=1e-6)
    model.fit(train_X[valid_train_samples,:], train_Y[valid_train_samples])
    predicted_Y = model.predict(test_X)
    
end_time = time.time()
print "predition time %f" %((end_time - start_time)/num_tests)
##
plt.semilogx(model.alphas_, model.mse_path_, ':')
plt.semilogx(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline((model.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate = %1e' %model.alpha_)

plt.legend(loc='best')

plt.xlabel('alpha')
plt.ylabel('Mean square error')
plt.show()