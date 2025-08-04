import unittest
import time
import logging

from model import Model
from onnx import helper
from onnx import TensorProto
from onnx import load

import numpy as np

def _main():
    onnx_model_path = "/home/vaibhav/zivid/zivid-sdk/vision-machine-learning/workspace/onnx/robocolor_denoise.onnx"
    onnx_model = load(onnx_model_path)

    model = Model()
    model.BuildFromOnnxModel(onnx_model)

    device = "CUDA"
    print(f"Optimizing schedule for model", flush=True)
    time_start = time.time()
    schedule = model.OptimizeSchedule(device=device)
    time_end = time.time()
    print(f"Schedule optimization time: {time_end - time_start:.4f} seconds")
    
    input_data = np.random.rand(1, 3, 502, 502).astype(np.float32)

    print(f"Running model with input shape: {input_data.shape}", flush=True)
    time_start = time.time()
    outputs = model.run([input_data], device=device)
    time_end = time.time()
    print(f"Model compile and run time: {time_end - time_start:.4f} seconds")

if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        exit(1)