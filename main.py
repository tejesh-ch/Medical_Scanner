from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# outputs 3d feature maps (height, width, features)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

# coverts 3d features to 1d vectors
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16

# augmentation configuration used for training
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True)

# augmentation configuration for testing: only rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)

# generator that will read pictures found and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'Training/',
        target_size = (150, 150),
        batch_size = batch_size,
        class_mode= 'binary')

validation_generator = test_datagen.flow_from_directory(
        'Testing/', # target directory
        target_size=(150, 150), # all images will be resized
        batch_size=batch_size,
        class_mode='binary') # binary labels


model.fit(train_generator,
          epochs=50,
          batch_size=16,
          validation_data=validation_generator)

model.save_weights('Output/glioma_tumor')
