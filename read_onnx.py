import onnx
import onnx.numpy_helper as numpy_helper
import numpy as np


def onnx2dic(onnx_file):
    model = onnx.load(onnx_file)
    dic = {}
    for weight in model.graph.initializer:
        dic[weight.name] = numpy_helper.to_array(weight)
    return dic


print(onnx2dic(model))