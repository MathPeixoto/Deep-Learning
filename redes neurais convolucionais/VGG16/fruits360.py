import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from glob import glob


#SETTING VARIABLES
IMAGE_SIZE = [100, 100]
epochs = 5
batch_size = 32

#Dataset paths
train_path = '../../datasets/fruits-360-small/Training'
test_path = '../../datasets/fruits-360-small/Test'

#useful for getting the number of traing images
image_files = glob(train_path + '/*/*.jp*g')
#useful for getting the number of test images
test_image_files = glob(test_path + '/*/*.jp*g')
#useful for getting the number of classes
folders = glob(train_path + '/*')

#CREATING MODEL
#adding preprocessing layer to the front of VGG16
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False    
x = Flatten()(vgg.output)
output = Dense(units=len(folders), activation='softmax')(x)
#create a custom model from VGG16
model = Model(inputs=vgg.input, outputs=output)
model.summary()
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

#getting images from dataset
gen = ImageDataGenerator(rescale=1./255,
                         rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         preprocessing_function=preprocess_input)
base_train = gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, batch_size=batch_size)
base_test = gen.flow_from_directory(test_path, target_size=IMAGE_SIZE, batch_size=batch_size)

#training and validating the model
r = model.fit_generator(base_train, steps_per_epoch=len(image_files)/batch_size,
                        epochs=epochs, validation_data=base_test, validation_steps=len(test_image_files)/batch_size)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

#predicting one image
test_one_image = image.load_img(test_path + '/Banana/12_100.jpg',
                              target_size = IMAGE_SIZE)
test_one_image = image.img_to_array(test_one_image)
test_one_image /= 255
test_one_image = np.expand_dims(test_one_image, axis=0)
r = model.predict(test_one_image)
p = np.argmax(r, axis=1)



#confusion matrix
# get label mapping for confusion matrix plot later
test_gen = gen.flow_from_directory(test_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices.items())
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k

# should be a strangely colored image (due to VGG weights being BGR)
for x, y in test_gen:
  print("min:", x[0].min(), "max:", x[0].max())
  plt.title(labels[np.argmax(y[0])])
  plt.imshow(x[0])
  plt.show()
  break

def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(test_path, len(test_image_files))
print(valid_cm)
