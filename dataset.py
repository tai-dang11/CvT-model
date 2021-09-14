from tensorflow import keras
from tensorflow.keras import layers


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
x_train = data_augmentation(x_train) / 255.0
x_test = data_augmentation(x_test) / 255.0

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


