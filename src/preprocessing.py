from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessing(df):
    X = df.drop('Class', axis = 1)
    y = df['Class']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
    
    Scaler  = StandardScaler()
    X_train_std  = Scaler.fit_transform(X_train)
    X_test_std = Scaler.transform(X_test)
    
    return X_train_std,X_test_std,y_train,y_test