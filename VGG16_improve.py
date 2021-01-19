from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

conv_base = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# conv_base.summary()

train_dir = 'dvc_small/train'
test_dir = 'dvc_small/test'
validation_dir = 'dvc_small/validation'

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=40,
                                                             width_shift_range=0.2, height_shift_range=0.2,
                                                             shear_range=0.2, zoom_range=0.2,
                                                             horizontal_flip=True, fill_mode='nearest')
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                    batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150),
                                                        batch_size=32, class_mode='binary')

model = keras.models.Sequential()
model.add(conv_base)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_generator, steps_per_epoch=100, epochs=30,
                    validation_data=validation_generator,
                    validation_steps=50)

history_dict = history.history
history_dict.keys()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# epochs = range(1, len(loss) + 1)
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# plt.clf()
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()