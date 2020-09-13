import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
print(X_treinamento.shape)
print(y_treinamento.shape)
print(X_teste.shape)
print(y_teste.shape)

plt.imshow(X_treinamento[10000], cmap='gray')
plt.title('Class '+ str(y_treinamento[10000]))

#tensorflow can handle format: (batch,height,width,channel)
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_treinamento
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#very similar to min-max normalization: we transform the values
#within the range [0,1] as usual
previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation = 'relu'))

# depois
classificador.add(BatchNormalization())
# se fazemos o escalonamento para a camada de entrada, também podemos
# fazer o escalonamento para as camadas de convoluções, mostrar as imagens
# reduz significativamente o tempo de treinamento

classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())
classificador.add(BatchNormalization())

# fully connected
classificador.add(Dense(128, activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(Dropout(0.3))

classificador.add(Dense(10,activation="softmax"))

classificador.compile(loss='categorical_crossentropy', 
                      optimizer="adam", metrics=['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, 
                  batch_size=128, epochs=20, 
                  validation_data=(previsores_teste, classe_teste), verbose=2)

resultado = classificador.evaluate(previsores_teste, classe_teste)
