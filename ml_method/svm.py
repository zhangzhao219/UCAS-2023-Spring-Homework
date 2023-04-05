##----------4.使用SVM模型进行训练----##
from sklearn import svm 




def SVMMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test):

    svmm = svm.SVC(kernel='linear')
    svmm.fit(vec.transform(x_train),y_train)

    y_train_hat=svmm.predict(vec.transform(x_train))
    y_val_hat=svmm.predict(vec.transform(x_val))
    y_test_hat=svmm.predict(vec.transform(x_test))

    return y_train_hat,y_val_hat,y_test_hat