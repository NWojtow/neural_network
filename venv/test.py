from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('/home/norbert/PycharmProjects/neural_network/venv/my_model.h5')

test_dir="/home/norbert/Pulpit/dog vs cat/dataset/test_set"
train_dir="/home/norbert/Pulpit/dog vs cat/dataset/training_set"

test_image = image.load_img(test_dir + '/dogs/dog.4128.jpg', target_size=(100,100))

test_image = image.img_to_array(test_image)

res_list= ["Kotel","Pjesiula"]

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

print(res_list[int(model.predict(test_image))])

print(model.predict(test_image))

