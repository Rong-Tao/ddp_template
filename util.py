import os
import argparse
import torch
import torch.optim as optim

BATCH_SIZE = 64
EPOCH_NUM = 10
TRAIN_VAL_RATIO = 0.8 # 0-1

criterion = torch.nn.CrossEntropyLoss()

def get_optimizer(model):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    return optimizer, scheduler

def batch_logger(writer, batch_idx, step_num, loss):
    writer.add_scalar('Batch Training Loss', loss, step_num)
    #add extra if you want

def epoch_logger_saver(model, writer, epoch, mean_trainloss, validation_loss, best_loss, state_dict_dir):
    writer.add_scalar('Epoch Training Loss', mean_trainloss, epoch)
    writer.add_scalar('Epoch Validation Loss', validation_loss, epoch)
    if mean_trainloss < best_loss:
        best_loss = mean_trainloss
        model_save_path = os.path.join(state_dict_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
    return best_loss
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
