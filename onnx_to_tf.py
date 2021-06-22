import onnx
import sys
import os
import tensorflow as tf
from onnx_tf.backend import prepare


def onnx2pb(onnx_path, tf_dir):
    onnx_model = onnx.load(onnx_path)  # load onnx model
    tf_model = prepare(onnx_model)  # run the loaded model
    tf_model.export_graph(tf_dir)


onnx2pb("tfmodel.onnx", 'saved_model/my_model')

new_model = tf.keras.models.load_model('saved_model/my_model')
print(type(new_model))
