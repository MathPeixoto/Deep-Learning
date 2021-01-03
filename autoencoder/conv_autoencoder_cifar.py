import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import numpy as np
from tensorflow.keras import backend as k

class ConvAutoencoder:

    def __init__(self, previsores_treinamento, previsores_teste, save_model = True):
        self.previsores_treinamento = previsores_treinamento
        self.previsores_teste = previsores_teste
        self.save_model = save_model

    def convAutoencoder(self):
        k.clear_session()
        autoencoder = Sequential()
    
        # Se mudar o input shape para somente um canal os resultados tendem a ser melhores
        # Encoder
        autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape=(64, 48, 1)))
        autoencoder.add(MaxPooling2D(pool_size = (2,2)))
        
        autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding='same'))
        autoencoder.add(MaxPooling2D(pool_size = (2,2), padding='same'))
        
        # 8, 6, 8 --> Necessário pegar as dimensões dessa camada e realizar um reshape depois do Flatten para voltar a essa dimensão
        autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding='same', strides = (2,2)))
        
        autoencoder.add(Flatten())
        
        autoencoder.add(Reshape((8, 6, 8)))
        
        # Decoder
        autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding='same'))
        autoencoder.add(UpSampling2D(size = (2,2)))
        
        autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding='same'))
        autoencoder.add(UpSampling2D(size = (2,2)))
        
        autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', padding='same'))
        autoencoder.add(UpSampling2D(size = (2,2)))
        
        autoencoder.add(Conv2D(filters = 1, kernel_size = (3,3), activation = 'sigmoid', padding='same'))
    
        autoencoder.summary()
        
        autoencoder.compile(optimizer = 'adam', loss = 'mse',
                            metrics = ['mse'])
        autoencoder.fit(self.previsores_treinamento, self.previsores_treinamento,
                        epochs = 50, batch_size = 256, 
                        validation_data = (self.previsores_teste, self.previsores_teste))
        return autoencoder
    
    def getEncoder(self, autoencoder):
        return Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('flatten').output)

    
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
        c2.add(Dense(units = 64, activation='relu', input_dim=384))
        c2.add(Dense(units = 32, activation='relu'))
        c2.add(Dense(units = 10, activation='softmax'))
        c2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        c2.fit(previsores_treinamento_codificados, classe_dummy_treinamento, epochs=50, 
        batch_size=256, validation_data=(previsores_teste_codificados, classe_dummy_teste))