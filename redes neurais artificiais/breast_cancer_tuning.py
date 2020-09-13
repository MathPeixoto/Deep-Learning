import pandas as pd
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import backend as k # atualizado: tensorflow==2.0.0-beta1

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons): # atualizado: tensorflow==2.0.0-beta1
    k.clear_session()
    classificador = Sequential([
               tf.keras.layers.Dense(units=neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim=30),
               tf.keras.layers.Dropout(0.2),
               tf.keras.layers.Dense(units=neurons, activation = activation, kernel_initializer = kernel_initializer),
               tf.keras.layers.Dropout(0.2),
               tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
    classificador.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_