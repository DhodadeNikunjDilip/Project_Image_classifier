# config.py

import torch


batchsize = 32
epochs = 20
learning_rate = 0.001
resize_x = 64
resize_y = 64
input_channels = 3
step_size = 5
gamma = 0.1
num_classes = 10  # Set this based on your dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
