from rbm import RBM
import numpy as np

rbm = RBM(num_visible=6, num_hidden=3)

# Essa base indica 1 caso o usuário tenha assistido um filme e 
# 0 caso o usuário não tenha assistido ou não tenha gostado de um filme.
base = np.array([[0, 1, 1, 1, 0, 1],
                 [1, 1, 0, 1, 1, 1],
                 [0, 1, 0, 1, 0, 1],
                 [0, 1, 1, 1, 0, 1],
                 [1, 1, 0, 1, 0, 1],
                 [1, 1, 0, 1, 1, 1]])

leonardo_filmes = np.array([[0, 1, 0, 1, 0, 0]])

filmes = ['Freddy x Jason', 'O Ultimato Bourne', 'Star Trek',
          'O Exterminador do Futuro', 'Norbit', 'Star Wars']

rbm.train(base, max_epochs = 5000)
rbm.weights

# Gera a camada escondida
camada_escondida = rbm.run_visible(leonardo_filmes)

# Gera novamente a camada de entrada
recomendacao = rbm.run_hidden(camada_escondida)

filme_recomendado = [filmes[i] for i in range(len(leonardo_filmes[0])) if (leonardo_filmes[0, i] == 0) and (recomendacao[0, i] == 1)]
print(filme_recomendado)