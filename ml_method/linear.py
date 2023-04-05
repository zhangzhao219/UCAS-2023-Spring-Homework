from sklearn.linear_model import LinearRegression

def LinearMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test):

    lin_reg=LinearRegression()

    lin_reg.fit(vec.transform(x_train),y_train)

    y_train_hat=lin_reg.predict(vec.transform(x_train))
    y_val_hat=lin_reg.predict(vec.transform(x_val))
    y_test_hat=lin_reg.predict(vec.transform(x_test))
    
    return y_train_hat,y_val_hat,y_test_hat
