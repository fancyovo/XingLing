#!/bin/bash

# install dependencies
pip install -r requirements.txt

# download data
python data/data_downloader.py
python data/process_data.py
python data/sft_data_generator.py

# pretrain 
NUM_GPUS=4 # change this to the number of GPUs you have
torchrun --nproc_per_node=$NUM_GPUS train_pretrain.py

# SFT training
python train_sft.py

# inference
python inference/chat.py