#!/usr/bin/sh
mkdir -p dataset
mkdir -p models
mkdir -p config

apt-get update
apt-get install -y libsndfile1
apt-get install -y portaudio19-dev
apt-get install -y sox
apt-get install -y zip

pip install -r requirements.txt
