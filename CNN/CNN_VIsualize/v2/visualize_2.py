from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

K = keras.backend

model = keras.applications.VGG16(weights='imagenet')

img_path = '/home/hjy/PycharmProjects/py4tf/CNN/dvc_small/train/dogs/dog.24.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = keras.applications.vgg16.preprocess_input(x)

preds = model.predict(x)
print(keras.applications.vgg16.decode_predictions(preds, top=3))
print(np.argmax(preds[0]))

keeshond_output = model.output[:, 261]
last_conv_layer = model.get_layer('block5_conv3')
heatmap_model = keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
with tf.GradientTape() as tape:
    conv_output, Predictions = heatmap_model(x)
    prob = Predictions[:, np.argmax(Predictions[0])]
    grads = tape.gradient(prob, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= np.max(heatmap)
plt.matshow(heatmap[0], cmap='viridis')
# plt.show()

origin_img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap[0], (origin_img.shape[1], origin_img.shape[0]), interpolation=cv2.INTER_CUBIC)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(origin_img, 0.5, heatmap, 0.5, 0)
cv2.imwrite('heatmap.jpg', superimposed_img)
plt.figure()
plt.imshow(superimposed_img)
plt.show()