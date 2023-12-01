import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
import argparse
import os

## Initialize Distributed Training #####
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return global_rank, world_size

rank, world_size = init_distributed_mode()
## Arguments 
if rank == 0 :
    parser = argparse.ArgumentParser(description='Pass log directories to main script.') 
    parser.add_argument('--output_log', type=str, help='Path to the output log directory.')
    args = parser.parse_args()
##### Directories #####
result_dir = args.output_log if rank == 0 else None
tensor_bd_dir = os.path.join(result_dir, 'tensorboard') if rank == 0 else None
state_dict_dir = os.path.join(result_dir, 'state_dict') if rank == 0 else None
if rank == 0:
    os.makedirs(result_dir, exist_ok=True) 
    os.makedirs(tensor_bd_dir, exist_ok=True)
    os.makedirs(state_dict_dir, exist_ok=True)


# Training
def train(rank, world_size):
    model = BigramLanguageModel(enc.n_vocab).cuda()
    model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    train_dataset = SequenceDataset(train_data, block_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    best_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss()  # Make sure to define your loss function

    # Initialize SummaryWriter for rank 0
    if rank == 0:
        writer = SummaryWriter(log_dir=tensor_bd_dir)

    for epoch in range(10):
        train_loss = 0.0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            logits = model(xb)
            loss = criterion(logits, yb.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if rank == 0:
                # Log batch training loss
                writer.add_scalar('Batch Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

                if batch_idx % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        generated_sequence = model.module.generate(idx=torch.randint(0, 10000, (1, 1), dtype=torch.long).cuda(), max_new_tokens=30)[0]
                        generated_text = decode(generated_sequence.tolist())
                        writer.add_text('Generated Text', generated_text, epoch * len(train_loader) + batch_idx)
                    model.train()

        if rank == 0:
            # Log epoch training loss
            writer.add_scalar('Epoch Training Loss', train_loss / len(train_loader), epoch)

            # Save model if current loss is the best so far
            if train_loss < best_loss:
                best_loss = train_loss
                model_save_path = os.path.join(state_dict_dir, f"bigram_language_model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), model_save_path)

    # Close the SummaryWriter after training is complete
    if rank == 0:
        writer.close()

@record
def main():
    train(rank, world_size)

if __name__ == "__main__":
    main()
