import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
import numpy as np
from tensorflow.keras import backend as k

class DeepAutoencoder:

    def __init__(self, previsores_treinamento, previsores_teste, save_model = True):
        self.previsores_treinamento = previsores_treinamento
        self.previsores_teste = previsores_teste
        self.save_model = save_model

    def deepAutoencoder(self):
        k.clear_session()
        dimensao_imagem = 3072
        autoencoder = Sequential()
        # Encode
        autoencoder.add(Dense(units = dimensao_imagem / 2, activation = 'relu', input_dim = 3072))
        autoencoder.add(Dense(units = dimensao_imagem / 4, activation = 'relu'))
        autoencoder.add(Dense(units = dimensao_imagem / 8, activation = 'relu'))
        
        # Decode
        autoencoder.add(Dense(units = dimensao_imagem / 4, activation = 'relu'))
        autoencoder.add(Dense(units = dimensao_imagem / 2, activation = 'relu'))
        autoencoder.add(Dense(units = dimensao_imagem, activation = 'sigmoid'))
        
        autoencoder.summary()
        
        autoencoder.compile(optimizer = 'adam', loss = 'mse',
                            metrics = ['mse'])
        autoencoder.fit(self.previsores_treinamento, self.previsores_treinamento,
                        epochs = 50, batch_size = 256, 
                        validation_data = (self.previsores_teste, self.previsores_teste))
        
        return autoencoder
    
    
    def getEncoder(self, autoencoder):
        dimensao_original = Input(shape=(3072,))
        camada_encoder1 = autoencoder.layers[0]
        camada_encoder2 = autoencoder.layers[1]
        camada_encoder3 = autoencoder.layers[2]
        
        encoder = Model(dimensao_original,
                        camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original))))
        return encoder

    
    def plotImagens(self, imagens_codificadas, imagens_decodificadas):
        numero_imagens = 10
        imagens_teste = np.random.randint(self.previsores_teste.shape[0], size = numero_imagens)
        plt.figure(figsize=(35, 35))
        
        for i, indice_imagem in enumerate(imagens_teste):   
            # imagem original
            plt.subplot(10,10,i + 1)
            plt.imshow(self.previsores_teste[indice_imagem].reshape(32, 32, 3))
            plt.xticks(())
            plt.yticks(())
            
            # imagem codificada
            plt.subplot(10,10,i + 1 + numero_imagens)
            plt.imshow(imagens_codificadas[indice_imagem].reshape(24, 16))
            plt.xticks(())
            plt.yticks(())
            
             # imagem reconstruída
            plt.subplot(10,10,i + 1 + numero_imagens * 2)
            plt.imshow(imagens_decodificadas[indice_imagem].reshape(32, 32, 3))
            plt.xticks(())
            plt.yticks(())
            
            
    def classificar(self, previsores_treinamento_codificados, previsores_teste_codificados, 
                    classe_dummy_treinamento, classe_dummy_teste):
        k.clear_session()
        c2 = Sequential()
        c2.add(Dense(units = 192, activation='relu', input_dim=384))
        c2.add(Dense(units = 96, activation='relu'))
        c2.add(Dense(units = 10, activation='softmax'))
        c2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        c2.fit(previsores_treinamento_codificados, classe_dummy_treinamento, epochs=50, 
        batch_size=256, validation_data=(previsores_teste_codificados, classe_dummy_teste))