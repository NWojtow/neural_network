import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from zipfile import ZipFile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir="/home/norbert/Pulpit/dog vs cat/dataset/test_set"


test_dir_cats = test_dir + '/cats'
test_dir_dogs = test_dir + '/dogs'



model = tf.keras.models.load_model('/home/norbert/PycharmProjects/neural_network/venv/my_model.h5')

def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(x = test_image)
    print(result)
    if result[0][0]  == 1:
        prediction = 'It\'s a Dog'
    else:
        prediction = 'It\'s a Cat'
    return prediction


print(testing_image(test_dir + '/cats/cat.4128.jpg'))
