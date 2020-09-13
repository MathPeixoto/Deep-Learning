import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras import backend as k

#LOADING DATASET - 1
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

plt.imshow(x_train[2])
plt.title('Classe ' + str(y_train[2]))

#PREPROCESSING - 2
X = np.concatenate((x_train, x_test), axis=0)
classes = np.concatenate((y_train, y_test), axis=0)

predicts = X.reshape(X.shape[0], 32, 32, 3)
predicts = X.astype('float32')
predicts /= 255


#CREATE NEURAL NETWORK STRUCT - 3
def create_net():
    k.clear_session()
    classifier = Sequential()
    classifier.add(Conv2D(64, (3,3), 
                             input_shape=(32, 32, 3),
                             activation = 'relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (3,3)))
    
    classifier.add(Conv2D(64, (3,3), activation = 'relu'))
    classifier.add(BatchNormalization())
    
    classifier.add(Conv2D(64, (3,3), activation = 'relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (3,3)))
    
    classifier.add(Flatten())
    
    classifier.add(Dense(units = 1024, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 1024, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 10, 
                            activation = 'softmax'))
    
    classifier.compile(loss = 'sparse_categorical_crossentropy',
                          optimizer = 'adam', metrics = ['accuracy'])
    return classifier

#TRAINS THE NEURAL NETWORK - 4
classifier = KerasClassifier(build_fn = create_net, epochs=100, batch_size = 128)
results = cross_val_score(estimator = classifier,
                             X = predicts, y = classes,
                             cv = 5, scoring = 'accuracy')

media = results.mean()
desvio = results.std()


classifier.fit(predicts, classes)

previsao = classifier.predict(predicts)







