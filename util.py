import os
import argparse
import torch
import torch.optim as optim

BATCH_SIZE = 64
EPOCH_NUM = 10

criterion = torch.nn.CrossEntropyLoss()

def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    return optimizer, scheduler

def arg():
    parser = argparse.ArgumentParser(description='Pass log directories to main script.') 
    parser.add_argument('--output_log', type=str, help='Path to the output log directory.')
    args = parser.parse_args()
    result_dir = args.output_log 
    tensor_bd_dir = os.path.join(result_dir, 'tensorboard')
    state_dict_dir = os.path.join(result_dir, 'state_dict') 
    os.makedirs(result_dir, exist_ok=True) 
    os.makedirs(tensor_bd_dir, exist_ok=True)
    os.makedirs(state_dict_dir, exist_ok=True)
    return result_dir, state_dict_dir, tensor_bd_dir