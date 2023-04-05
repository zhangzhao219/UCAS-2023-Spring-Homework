from sklearn.tree import DecisionTreeRegressor

def DTMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test):

    tree_reg=DecisionTreeRegressor()

    tree_reg.fit(vec.transform(x_train),y_train)

    y_train_hat=tree_reg.predict(vec.transform(x_train))
    y_val_hat=tree_reg.predict(vec.transform(x_val))
    y_test_hat=tree_reg.predict(vec.transform(x_test))

    return y_train_hat,y_val_hat,y_test_hat