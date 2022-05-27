#!/usr/bin/sh
mkdir -p dataset

#Download librispeech dataset
wget -N -P dataset http://www.openslr.org/resources/12/test-clean.tar.gz

#Untar dataset
tar -xvzf dataset/test-clean.tar.gz -C dataset

#Convert the Flac into Wave format
python3 utils/convert_librispeech.py \
    --input_dir dataset/LibriSpeech/test-clean \
    --dest_dir dataset/LibriSpeech/test-clean-wav \
    --output_json dataset/LibriSpeech/librispeech-test-clean-wav.json