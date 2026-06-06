
def Evaluate(model,X_test_std,y_test):
    loss,accuracy = model.evaluate(X_test_std,y_test)
    print('Test Accuracy = ',accuracy)
    return loss,accuracy
    
def Predict(model,X_test_std):
    y_pred = model.predict(X_test_std)
    return y_pred
