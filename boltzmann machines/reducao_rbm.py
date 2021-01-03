import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

base = datasets.load_digits()
previsores = np.asarray(base.data, 'float32')
classe = np.asarray(base.target)

normalizador = MinMaxScaler()
previsores = normalizador.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.2, random_state=0)

rbm = BernoulliRBM(random_state=0)
rbm.iter = 50           # Epochs
rbm.n_components = 50   # Quantidade de neurônios nas camadas escondidas
naive_rbm = GaussianNB()
classificador_rbm = Pipeline(steps = [('rbm', rbm), ('naive', naive_rbm)])
classificador_rbm.fit(previsores_treinamento, classe_treinamento)

plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
    
plt.show()

# Predição com a redução de dimensionalidade usando RBM
previsoes_rbm = classificador_rbm.predict(previsores_teste)
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste)

# Predição sem a redução de dimensionalidade
naive_simples = GaussianNB()
naive_simples.fit(previsores_treinamento, classe_treinamento)
previsoes_simples = naive_simples.predict(previsores_teste)
precisao_simples = metrics.accuracy_score(previsoes_simples, classe_teste)