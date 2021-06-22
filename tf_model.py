import os
import tensorflow as tf
from tensorflow import keras
import tf2onnx

# A tensorflow model whose architechture is 16 -> 10 -> 3
model = tf.keras.models.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(16,)),
    keras.layers.Dense(3)
])

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=[tf.metrics.SparseCategoricalAccuracy()])

print(model.summary())

model.save('tfmodel.h5')

model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
    model, output_path='tfmodel.onnx')
