"""
Model Training Configuration File
===============================
This file defines all command line arguments and their default values required for model training.
Main contents include:
1. Training related parameters (epochs, learning rate, batch size, etc.)
2. Hardware related parameters (GPU ID, number of worker threads, etc.)
3. Dataset related parameters
4. Model saving related parameters
"""

import argparse

# Create command line argument parser
parser = argparse.ArgumentParser('The training and evaluation script', add_help=False)

# Training related parameters
# ==============================================================================
parser.add_argument('--epoch',
                   type=int,
                   default=50,
                   help='Total training epochs, determines the number of model training iterations')

parser.add_argument('--lr',
                   type=float,
                   default=1e-4,
                   help='Initial learning rate, affects the step size of model parameter updates')

parser.add_argument('--batchsize',
                   type=int,
                   default=128,
                   help='Training batch size, affects memory usage and training speed')

# Hardware and environment related parameters
# ==============================================================================
parser.add_argument('--gpu_id',
                   type=str,
                   default='0,1,2,3,4,5,6,7',
                   help='Specify GPU IDs to use, comma-separated for multi-GPU training')

parser.add_argument('--num_work',
                   type=int,
                   default=8,
                   help='Number of worker threads for data loading, 0 means using only the main thread')

parser.add_argument('--start_epoch',
                   type=int,
                   default=1,
                   help='Starting epoch for training, used for resuming training from a checkpoint')

# Dataset related parameters
# ==============================================================================
parser.add_argument('--dataset',
                   type=str,
                   default='Houston2018',
                   help='Dataset name, currently supports Berlin dataset')

parser.add_argument('--useval',
                   type=int,
                   default=1,
                   help='Whether to use validation set, 0 means not use, 1 means use')

parser.add_argument('--save_path',
                   type=str,
                   default='./checkpoints/',
                   help='Path to save models and logs')

# Model saving related parameters
# ==============================================================================
parser.add_argument('--best_acc',
                   type=float,
                   default=1,
                   help='Best accuracy, used for saving the optimal model')

parser.add_argument('--best_epoch',
                   type=int,
                   default=1,
                   help='The training epoch corresponding to the best accuracy')

parser.add_argument('--model_path',
                   type=str,
                   default='/home/scl/dmy/open/checkpoints/Houston2018/weight/2025-05-20_00-01-28_93.19103577371048_Houston2018_Net_epoch_97.pth',
                   help='Model weight file path, used to load pre-trained model for testing')

# Parse command line arguments
opt = parser.parse_args()