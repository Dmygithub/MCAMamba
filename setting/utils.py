

import torch


def create_folder(save_path):

    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f'Created folder: [{save_path}]')
    return save_path


def compute_accuracy(predictions, ground_truth):

    predicted_classes = torch.argmax(predictions, dim=1)  # Get the class with highest probability for each sample
    accuracy = torch.sum(predicted_classes == ground_truth).item() / len(ground_truth)
    return accuracy


def random_seed_setting(seed: int = 42):

    import random
    import os
    import numpy as np
    import torch

    torch.manual_seed(seed)  # Set PyTorch CPU random seed
    torch.cuda.manual_seed(seed)  # Set current GPU random seed
    torch.cuda.manual_seed_all(seed)  # Set all GPUs random seed
    torch.backends.cudnn.deterministic = True  # Ensure the convolution algorithm is deterministic
    torch.backends.cudnn.benchmark = False  # Disable cudnn benchmark mode
    np.random.seed(seed)  # Set NumPy random seed
    random.seed(seed)  # Set Python random seed
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python hash seed

    # Example of creating folders
    create_folder('./checkpoints')
    create_folder('./checkpoints/Berlin')
    create_folder('./checkpoints/Berlin/log')
    create_folder('./checkpoints/Berlin/weight')