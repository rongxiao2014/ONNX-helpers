import onnx
import onnx.numpy_helper
import numpy as np



# Given a dictionary of model weights and onnx base model, create a new onnx model from old while retaining all the underlying computational graph and operators.
def updateonnx(dic, onnx_file, new_onnx_file):
    model = onnx.load(onnx_file)
    for weight in model.graph.initializer:
        weight.raw_data = onnx.numpy_helper.from_array(
            dic[weight.name]).raw_data
    onnx.save(model, new_onnx_file)




# Change the weights for a specific layer (provided the "name" of the layer) and output a new onnx file.
def updateonnxlayer(layer_name, weight_numpy, onnx_file, new_onnx_file):
    new_weight = onnx.numpy_helper.from_array(weight_numpy)
    model = onnx.load(onnx_file)
    for weight in model.graph.initializer:
        if(weight.name == layer_name):
            weight.raw_data = new_weight.raw_data
    onnx.save(model, new_onnx_file)