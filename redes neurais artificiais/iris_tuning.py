import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import backend as k # atualizado: tensorflow==2.0.0-beta1
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede(optimizer, kernel_initializer, activation, neurons): # atualizado: tensorflow==2.0.0-beta1
    k.clear_session()
    classificador = Sequential([
    tf.keras.layers.Dense(units=neurons, activation = activation,
                          kernel_initializer = kernel_initializer, input_dim=4),
    tf.keras.layers.Dense(units=3, activation = 'softmax')])
    classificador.compile(optimizer = optimizer, loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede)

parametros = {'batch_size': [10, 15],
              'epochs': [200, 300],
              'optimizer': ['adam', 'sgd'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'elu'],
              'neurons': [4, 8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_