from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#LOADING IMAGES
train_generator = ImageDataGenerator(rescale=1./255, zoom_range=0.2, 
                                     rotation_range=7, horizontal_flip=True,
                                     height_shift_range = 0.07, shear_range=0.2)
test_generator = ImageDataGenerator(rescale=1./255)

train_base = train_generator.flow_from_directory('dataset_personagens/training_set',
                                                 target_size=(128, 128),
                                                 batch_size = 10,
                                                 class_mode='binary')
test_base = test_generator.flow_from_directory('dataset_personagens/test_set',
                                                 target_size=(128, 128),
                                                 batch_size = 10,
                                                 class_mode='binary')


#CNN STRUCT
classifier = Sequential()
classifier.add(Conv2D(128, (3,3), input_shape = (128, 128, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(128, (3,3), input_shape = (128, 128, 3), activation='relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(128, (3,3), input_shape = (128, 128, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics=['accuracy'])

#TRAINING
classifier.fit_generator(train_base, epochs=1000, steps_per_epoch=196 / 10, validation_data=test_base, validation_steps = 73 / 10)
