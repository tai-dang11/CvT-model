import tensorflow_addons as tfa
from dataset import x_test,x_train,y_train,y_test,num_classes
from model import Cvt
import tensorflow as tf
from cf import CFGS

CvT = Cvt(model_name='cvt-13-224x224', num_classes=10, CFGS=CFGS)
model = tf.keras.Sequential(tf.keras.Input(shape=(224, 224, 3)),
                            CvT,
                            tf.layers.Dense(num_classes, activation="softmax")
                            )

optimizer = tfa.optimizers.AdamW(
    learning_rate=0.00001, weight_decay=0.1
)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=["accuracy"])


model.fit(
    x_train,
    y_train,
    validation_data = (x_test,y_test),
    batch_size=32,
    epochs=10,
    validation_split=0.1,
    verbose=1
)
