from sklearn import metrics
from sklearn.cross_validation import train_test_split

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(train_X[valid_train_samples], ylabel_train, test_size=0.4, random_state=0)

### Cross validation to find the optimum k value
    for k in xrange(1,100):
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_cv, y_train_cv)
        predicted = knn.predict(X_test_cv)
        print metrics.classification_report(y_test_cv, predicted)


