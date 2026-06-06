import sklearn.datasets
import pandas as pd

def load_data():
    dataset = sklearn.datasets.load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
    return df
