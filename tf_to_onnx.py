import tf2onnx
import os
import tensorflow as tf
from tensorflow import keras


def tf2onnx_from_h5(filepath, onnxpath):
    model = tf.keras.models.load_model(filepath)
    print(model.summary())
    _, _ = tf2onnx.convert.from_keras(
        model, output_path=onnxpath)


tf2onnx_from_h5('tfmodel.h5', 'tfmodel_from_file.onnx')
