#!/usr/bin/sh
mkdir -p persistence

python3 utils/get_gpt2.py
cp persistence/model.pt models/generator_torchscript/model.pt
cp persistence/model.onnx models/generator_onnx/1/model.onnx