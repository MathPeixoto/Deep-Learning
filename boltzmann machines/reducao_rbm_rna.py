import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

base = datasets.load_digits()
previsores = np.asarray(base.data, 'float32')
classe = np.asarray(base.target)

normalizador = MinMaxScaler()
previsores = normalizador.fit_transform(previsores)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.2, random_state=0)

rbm = BernoulliRBM(random_state=0)
rbm.iter = 50           # Epochs
rbm.n_components = 50   # Quantidade de neurônios nas camadas escondidas
mlp_rbm = MLPClassifier(hidden_layer_sizes = (40, 40), activation='relu',
                          batch_size=30, max_iter=4000, verbose=1, learning_rate_init=0.0001)

classificador_rbm = Pipeline(steps = [('rbm', rbm), ('mlp', mlp_rbm)])
classificador_rbm.fit(previsores_treinamento, classe_treinamento)

# Predição com a redução de dimensionalidade usando RBM
previsoes_rbm = classificador_rbm.predict(previsores_teste)
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste)

# Predição sem a redução de dimensionalidade com RBM
mlp_simples = MLPClassifier(hidden_layer_sizes = (40, 40), activation='relu',
                          batch_size=30, max_iter=4000, verbose=1, learning_rate_init=0.0001)
mlp_simples.fit(previsores_treinamento, classe_treinamento)
previsoes_simples = mlp_simples.predict(previsores_teste)
precisao_simples = metrics.accuracy_score(previsoes_simples, classe_teste)