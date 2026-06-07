def Optimize_model(model,X_train_std,y_train,epochs):
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
        )
    Training = model.fit(X_train_std,y_train,validation_split=0.1,epochs = epochs)
    return Training