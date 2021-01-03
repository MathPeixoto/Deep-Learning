import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from minha_tarefa.deep_autoencoder_cifar import DeepAutoencoder
from minha_tarefa.conv_autoencoder_cifar import ConvAutoencoder

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Normalização dos dados
(previsores_treinamento, classe_treinamento), (previsores_teste, classe_teste) = cifar10.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

previsores_treinamento_simples = previsores_treinamento.reshape((len(previsores_treinamento),
                                                            np.prod(previsores_treinamento.shape[1:])))
previsores_teste_simples = previsores_teste.reshape((len(previsores_teste),
                                                        np.prod(previsores_teste.shape[1:])))


classe_dummy_treinamento = np_utils.to_categorical(classe_treinamento)
classe_dummy_teste = np_utils.to_categorical(classe_teste)


def acoesDeepAutoencoder():
    # Cria uma instancia DeepAutoncoder
    dp = DeepAutoencoder(previsores_treinamento_simples, previsores_teste_simples)
    
    # Faz o treinamento do autoencoder
    dp_autoencoder = dp.deepAutoencoder()
    
    # Recupera somente um encoder
    dp_encoder = dp.getEncoder(dp_autoencoder)

    # Codifica as imagens teste
    imagens_codificadas = dp_encoder.predict(previsores_teste_simples)
    imagens_decodificadas = dp_autoencoder.predict(previsores_teste_simples)
    
    # Plota as imagens
    dp.plotImagens(imagens_codificadas, imagens_decodificadas)
    
    # Previsores codificados
    previsores_treinamento_codificadas = dp_encoder.predict(previsores_treinamento_simples)
    previsores_teste_codificadas = dp_encoder.predict(previsores_teste_simples)
    
    # Realiza as classificações usando o deep autoencoder
    dp.classificar(previsores_treinamento_codificados=previsores_treinamento_codificadas,
                   previsores_teste_codificados=previsores_teste_codificadas,
                   classe_dummy_treinamento=classe_dummy_treinamento,
                   classe_dummy_teste=classe_dummy_teste)


def acoesConvAutoencoder():
    
    # Dados específicos para o conv encoder
    previsores_treinamento_conv = previsores_treinamento.reshape((len(previsores_treinamento), 64, 48, 1))
    previsores_teste_conv = previsores_teste.reshape((len(previsores_teste), 64, 48, 1))
    
    # Cria uma instancia ConvAutoencoder
    conv = ConvAutoencoder(previsores_treinamento=previsores_treinamento_conv, previsores_teste=previsores_teste_conv)
    
    # Faz o treinamento do autoencoder
    conv_autoencoder = conv.convAutoencoder()
    
    # Recupera somente um encoder
    conv_encoder = conv.getEncoder(conv_autoencoder)
    
    # Codifica as imagens teste
    imagens_codificadas = conv_encoder.predict(previsores_teste_conv)
    imagens_decodificadas = conv_autoencoder.predict(previsores_teste_conv)

    # Plota as imagens
    conv.plotImagens(imagens_codificadas, imagens_decodificadas)
    
    # Previsores codificados
    previsores_treinamento_codificadas = conv_encoder.predict(previsores_treinamento_conv)
    previsores_teste_codificadas = conv_encoder.predict(previsores_teste_conv)
    
    
    # Realiza as classificações usando o deep autoencoder
    conv.classificar(previsores_treinamento_codificados=previsores_treinamento_codificadas,
                     previsores_teste_codificados=previsores_teste_codificadas,
                   classe_dummy_treinamento=classe_dummy_treinamento,
                   classe_dummy_teste=classe_dummy_teste)
    
    
acoesDeepAutoencoder()
acoesConvAutoencoder()

# Sem redução de dimensionalidade
c1 = Sequential()
c1.add(Dense(units = 3072 / 2, activation='relu', input_dim=3072))
c1.add(Dense(units = 3072 / 2, activation='relu'))
c1.add(Dense(units = 10, activation='softmax'))
c1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
c1.fit(previsores_treinamento_simples, classe_dummy_treinamento, epochs=50, 
                batch_size=256, validation_data=(previsores_teste_simples, classe_dummy_teste))
