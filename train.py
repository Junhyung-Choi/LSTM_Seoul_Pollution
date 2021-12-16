# 1. 패키지 불러오기

import datasets
from tensorflow import keras

from keras import layers, models, regularizers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

import tensorflow as tf

#1. 데이터 로드

price_dataset = datasets.PriceDataset()
[tr_x, tr_y, val_x, val_y] = price_dataset.getDataset()

print("Data Loader")

# 2. 파라미터 (유닛) 설정
Nin = 18
Nh_l = [64,128]
Nout = 1

## loss function
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

## Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch == 150:
        return lr * 0.1
    else:
        return lr
    

# 3. DNN 분류 모델 구현
class DNN(models.Model):
    
    def __init__(self, Nin, Nh_l, Nout):

        x = layers.Input(shape=(Nin,))
        h = layers.Dense(Nh_l[0], activation='relu')(x)
        h = layers.Dense(Nh_l[1], activation='relu')(h)
        h = layers.Dense(Nh_l[1], activation='relu')(h)
        h = layers.Dense(Nh_l[0], activation='relu')(h)
        y = layers.Dense(Nout)(h)

        super().__init__(x, y)

        self.compile(loss= root_mean_squared_error, optimizer='adam', 
                     metrics= tf.keras.metrics.RootMeanSquaredError())

def train():
    model = DNN(Nin, Nh_l, Nout)
    # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(tr_x, tr_y, epochs=300,
                        batch_size=32, validation_data = (val_x, val_y))
    model3.summary()
    model.save_weights('./models/weights_final.h5')

    
train()