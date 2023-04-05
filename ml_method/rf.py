from sklearn.ensemble import RandomForestRegressor

def RFMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test):

    forest_reg=RandomForestRegressor()
    forest_reg.fit(vec.transform(x_train),y_train)

    y_train_hat=forest_reg.predict(vec.transform(x_train))
    y_val_hat=forest_reg.predict(vec.transform(x_val))
    y_test_hat=forest_reg.predict(vec.transform(x_test))

    return y_train_hat,y_val_hat,y_test_hat