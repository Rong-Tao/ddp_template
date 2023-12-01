import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
import os
## Import our own scrips
from model import Model_Class
from dataset import Dataset_Class
from util import arg, get_optimizer, criterion, BATCH_SIZE, EPOCH_NUM

## Initialize Distributed Training #####
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return global_rank, world_size

rank, world_size = init_distributed_mode()

if rank == 0:
    result_dir, state_dict_dir, tensor_bd_dir = arg()


# Training
def train(rank, world_size):
    model = Model_Class.cuda()
    model = DDP(model)
    optimizer, scheduler = get_optimizer(model)

    train_dataset = Dataset_Class()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    best_loss = float('inf')

    # Initialize SummaryWriter for rank 0
    if rank == 0:
        writer = SummaryWriter(log_dir=tensor_bd_dir)

    for epoch in range(EPOCH_NUM):
        train_loss = 0.0
        for batch_idx, (img, gt) in enumerate(train_loader):
            img, gt = img.cuda(), gt.cuda()
            out = model(img)
            loss = criterion(out, gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if rank == 0:
                writer.add_scalar('Batch Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

        if rank == 0:
            writer.add_scalar('Epoch Training Loss', train_loss / len(train_loader), epoch)
            if train_loss < best_loss:
                best_loss = train_loss
                model_save_path = os.path.join(state_dict_dir, f"epoch_{epoch}.pth")
                torch.save(model.state_dict(), model_save_path)

    if rank == 0:
        writer.close()

@record
def main():
    train(rank, world_size)

if __name__ == "__main__":
    main()
