#!/usr/bin/sh

mkdir -p models

#Download jasper model config
wget -N -P config https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/conf/jasper/jasper_10x5dr.yaml

#Download jasper onnx ensemble models from NGC
wget -O jasper_pyt_onnx_fp16_amp_20.10.0.zip --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/jasper_pyt_onnx_fp16_amp/versions/20.10.0/zip

#Unzip models
unzip jasper_pyt_onnx_fp16_amp_20.10.0.zip -d models
unzip models/jasper_pyt_onnx_fp16_amp.zip -d models

#Clear the mess
rm jasper_pyt_onnx_fp16_amp_20.10.0.zip
rm models/jasper_pyt_onnx_fp16_amp.zip

#Download triton model configs
wget -N -P models/decoder-ts-script https://raw.githubusercontent.com/NVIDIA/DeepLearningExamples/master/PyTorch/SpeechRecognition/Jasper/triton/model_repo_configs/fp16/decoder-ts-script/config.pbtxt
wget -N -P models/feature-extractor-ts-trace https://raw.githubusercontent.com/NVIDIA/DeepLearningExamples/master/PyTorch/SpeechRecognition/Jasper/triton/model_repo_configs/fp16/feature-extractor-ts-trace/config.pbtxt
wget -N -P models/jasper-onnx https://raw.githubusercontent.com/NVIDIA/DeepLearningExamples/master/PyTorch/SpeechRecognition/Jasper/triton/model_repo_configs/fp16/jasper-onnx/config.pbtxt
mkdir -p models/jasper-onnx-ensemble
mkdir -p models/jasper-onnx-ensemble/1
wget -N -P models/jasper-onnx-ensemble https://raw.githubusercontent.com/NVIDIA/DeepLearningExamples/master/PyTorch/SpeechRecognition/Jasper/triton/model_repo_configs/fp16/jasper-onnx-ensemble/config.pbtxt

sleep 3

# Load ensemble model using Triton Client API
#curl -H 'Content-Type: application/json' -X POST -i localhost:8000/v2/repository/models/jasper-onnx-ensemble/load
