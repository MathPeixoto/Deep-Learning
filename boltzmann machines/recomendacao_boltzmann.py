from rbm import RBM
import numpy as np

rbm = RBM(num_visible=6, num_hidden=2)

# Essa base indica 1 caso o usuário tenha assistido um filme e 
# 0 caso o usuário não tenha assistido ou não tenha gostado de um filme.
base = np.array([[1, 1, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 0, 1],
                 [0, 0, 1, 1, 0, 1]])

filmes = ['A bruxa', 'Invocacao do Mal', 'O Chamado',
          'Se Beber não Case', 'Gente Grande', 'American Pie']

rbm.train(base, max_epochs = 5000)
rbm.weights

usuario1 = np.array([[1, 1, 0, 1, 0, 0]])
# o segundo neuroônio, o neurônio do filme de terror, foi o neurônio ativado
camada_escondida1 = rbm.run_visible(usuario1)

recomendacao = rbm.run_hidden(camada_escondida1)

filme_recomendado = [filmes[i] for i in range(len(usuario1[0])) if (usuario1[0, i] == 0) and (recomendacao[0, i] == 1)]
print(filme_recomendado)
