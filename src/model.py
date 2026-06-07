import tensorflow as tf
from tensorflow import keras

def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (30,)),
        keras.layers.Dense(15,activation='relu'),
        keras.layers.Dense(10,activation='relu'),
        keras.layers.Dense(2,activation='softmax')
        ])


    return model