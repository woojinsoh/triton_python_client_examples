#!/usr/bin/sh
mkdir -p persistence
mkdir -p models/ensemble_pipeline/1
mkdir -p models/generator_onnx/1
mkdir -p models/generator_torchscript/1

python3 utils/get_gpt2.py
cp persistence/model.pt models/generator_torchscript/model.pt
cp persistence/model.onnx models/generator_onnx/1/model.onnx