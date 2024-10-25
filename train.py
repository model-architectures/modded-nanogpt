import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import glob
import time

import numpy as np
import torch
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from datetime import datetime  
import argparse
from optimizer.Muon import Muon


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

# Argument parsing for command-line parameters
parser = argparse.ArgumentParser(description='Training Script with Job ID')
parser.add_argument('--job_id', type=str, default='[default_job_id]', help='Unique Job ID for this run')
parser.add_argument('--input_bin', type=str, default='data/fineweb10B/fineweb_train_*.bin', help='Input .bin to train on')
parser.add_argument('--input_val_bin', type=str, default='data/fineweb10B/fineweb_val_*.bin', help='Input .bin to eval validation loss on')
parser.add_argument('--batch_size', type=int, default=8*64, help='Batch size, in sequences, across all devices')
parser.add_argument('--device_batch_size', type=int, default=64, help='Batch size, in sequences, per device')
parser.add_argument('--sequence_length', type=int, default=1024, help='Sequence length, in tokens')
parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
parser.add_argument('--embedding_size', type=int, default=768, help='Size of the embeddings')

parser.add_argument('--num_iterations', type=int, default=100000, help='Number of iterations to run')
parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
parser.add_argument('--warmup_iters', type=int, default=2000, help='Number of iterations of linear warmup')
parser.add_argument('--warmdown_iters', type=int, default=10000, help='Number of iterations of linear warmdown')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--val_loss_every', type=int, default=125, help='Every how many steps to evaluate val loss? 0 for only at the end')
parser.add_argument('--val_tokens', type=int, default=10485760, help='How many tokens of validation data?')
parser.add_argument('--save_every', type=int, default=0, help='Every how many steps to save the checkpoint? 0 for only at the end')
parser.add_argument('--model_type', type=str, default='gpt2', help='Model type to train')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use')
parser.add_argument('--scheduler', type=str, default='wsd', help='Scheduler to use')

# Parse arguments from the command line
args = parser.parse_args()
# set up DDP (distributed data parallel). torchrun sets this env variable

if args.model_type == 'gpt2':
    from model.GPT import GPT, GPTConfig
elif args.model_type == 'gpt2_dual':
    from model.GPT_dual import GPT, GPTConfig
else:
    raise ValueError(f"Unknown model type: {args.model_type}")
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0)  # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

# Initialize wandb and track hyperparameters
if master_process:
    # Get the current date in MM_DD_YY format
    current_date = datetime.now().strftime("%m_%d_%y_%H_%M")

    wandb.init(
        project="modded-nanogpt",  # Set your wandb project name
        config={
            "batch_size": args.batch_size,
            "device_batch_size": args.device_batch_size,
            "sequence_length": args.sequence_length,
            "num_iterations": args.num_iterations,
            "learning_rate": args.learning_rate,
            "warmup_iters": args.warmup_iters,
            "warmdown_iters": args.warmdown_iters,
            "weight_decay": args.weight_decay,
            "val_loss_every": args.val_loss_every,
            "val_tokens": args.val_tokens,
        }
    )
    # Use the job_id and current date to name the wandb run
    wandb.run.name = f"run_{args.optimizer}_{args.model_type}_{current_date}_jobid_{args.job_id}"  # Modified run name

    # Log the Python code and environment details
    wandb.config.update({
        "pytorch_version": torch.version.__version__,
        "cuda_version": torch.version.cuda,
    })

x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=args.num_layers, n_head=args.num_heads, n_embd=args.embedding_size))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer (AdamW for all parameters)
optimizers = []
for opt in args.optimizer.split(","):
    if opt == "adamw":
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                                            weight_decay=args.weight_decay, fused=True))
    elif opt == "muon":
        optimizers.append(Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95,
                  rank=ddp_rank, world_size=ddp_world_size))

# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    # WSD scheduler
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
    
if args.scheduler == "wsd":
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
elif args.scheduler == "cosine":
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.num_iterations) for opt in optimizers]
else:
    raise ValueError(f"Unknown scheduler: {args.scheduler}")
# begin logging
if master_process:
    run_id = f"run_{args.optimizer}_{args.model_type}_{current_date}_jobid_{args.job_id}"
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# Start the clock
torch.cuda.synchronize()
t0 = time.time()
# Begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)

    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    # Evaluate the validation dataset periodically
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)

        # Validation loop
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps

        if master_process:
            # Log validation loss to wandb
            wandb.log({
                "step": step,
                "val_loss": val_loss.item(),
                "training_time_ms": training_time_ms,
                "step_avg_time_ms": training_time_ms / (timed_steps - 1)
            })

            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')

        torch.cuda.synchronize()
        t0 = time.time()

    # Save checkpoints periodically
    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)

        # Ensure the logs directory exists before saving the checkpoint
        save_dir = f'logs/{wandb.run.id}/'
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        log = {
            "step": step,
            "code": code,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_states": [opt.state_dict() for opt in optimizers],
        }
        torch.save(log, f'logs/{wandb.run.id}/state_step{step:06d}.pt')

        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # Training section
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # Forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()

        x, y = train_loader.next_batch()

        # Backward pass
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    # Scale gradients and step optimizers
    for p in model.parameters():
        p.grad /= train_accumulation_steps

    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()

    model.zero_grad(set_to_none=True)

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        wandb.log({
            "step": step + 1,
            "train_loss": train_loss.item(),
            "training_time_ms": approx_time,
            "step_avg_time_ms": approx_time / timed_steps,
        })

        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

if master_process:
    print(f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    wandb.log({"peak_memory_mib": torch.cuda.max_memory_allocated() // 1024 // 1024})

# Clean up
dist.destroy_process_group()

if master_process:
    wandb.finish()
