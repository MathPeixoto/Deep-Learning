import pandas as pd
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from keras.utils import np_utils

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
 
classificador = Sequential([
tf.keras.layers.Dense(units=8, activation = 'elu',
                      kernel_initializer = 'random_uniform', input_dim=4),
tf.keras.layers.Dense(units=3, activation = 'softmax')])
classificador.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])

classificador.fit(previsores, classe_dummy, batch_size = 10, epochs = 200)

classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_iris.h5')