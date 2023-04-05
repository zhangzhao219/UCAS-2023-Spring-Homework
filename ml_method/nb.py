from sklearn.naive_bayes import MultinomialNB


def NBMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test):

    nb = MultinomialNB()

    nb.fit(vec.transform(x_train),y_train)

    y_train_hat=nb.predict(vec.transform(x_train))
    y_val_hat=nb.predict(vec.transform(x_val))
    y_test_hat=nb.predict(vec.transform(x_test))

    return y_train_hat,y_val_hat,y_test_hat
