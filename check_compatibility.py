import onnx
import onnx.numpy_helper as numpy_helper
import numpy as np


# This function checks whether model_A and model_B has the same underlying computational graph and operators.
def check_model(model_A, model_B):
    if(model_A.graph.input != model_B.graph.input):
        return False
    elif(model_A.graph.output != model_B.graph.output):
        return False
    elif(model_A.graph.node != model_B.graph.node):
        return False
    else:
        return True