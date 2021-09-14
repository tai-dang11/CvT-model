import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
num_classes = 10
input_shape = (32, 32, 3)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(72, 72),
    ],

)
x_train = data_augmentation(x_train)
x_test = data_augmentation(x_test)

# print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# batch_size = 64
#
# dataset_name = "stanford_dogs"
# (ds_train, ds_test), ds_info = tfds.load(
#     dataset_name, split=["train", "test"], with_info=True, as_supervised=True
# )
# NUM_CLASSES = ds_info.features["label"].num_classes
# IMG_SIZE = 224
# size = (IMG_SIZE, IMG_SIZE)
# ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
# ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
#
# def normalize_img(image, label):
#     return tf.cast(image, tf.float32)/224, label


# One-hot / categorical encoding
# def input_preprocess(image, label):
#     label = tf.one_hot(label, NUM_CLASSES)
#     return image, label
#
#
# ds_train = ds_train.map(
#     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
# )
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
# ds_train = ds_train.batch(batch_size=batch_size)
# ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
#
# ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)
# ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# imagenette_map = {
#     "n01440764" : "tench",
#     "n02102040" : "springer",
#     "n02979186" : "casette_player",
#     "n03000684" : "chain_saw",
#     "n03028079" : "church",
#     "n03394916" : "French_horn",
#     "n03417042" : "garbage_truck",
#     "n03425413" : "gas_pump",
#     "n03445777" : "golf_ball",
#     "n03888257" : "parachute"
# }
#
# imagegen = ImageDataGenerator()
# train = imagegen.flow_from_directory("D:/Desktop/tensorflow/imagenette-320/train/", class_mode="categorical", shuffle=False, batch_size=64, target_size=(224, 224))
# # load val data
# val = imagegen.flow_from_directory("D:/Desktop/tensorflow/imagenette-320/train/", class_mode="categorical", shuffle=False, batch_size=64, target_size=(224, 224))

