import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from llm.utils.logger import setup_logging
import yaml
from llm.model import Transformer
from llm.dataloader import create_dataloader
from llm.utils.tester import text_generator
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    def __init__(self, model, dataloader, tokenizer, config):
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.config = config

        self.scaler = torch.amp.GradScaler('cuda')

    def train(self):

        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        self.device = torch.device(f"cuda:{local_rank}")
        self.model.to(self.device)

        self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)
        if global_rank == 0:
            self.logger = setup_logging(self.config['output_dir'])

        

        optimizer = optim.AdamW(self.model.module.parameters(), lr=self.config['lr'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=config["warmup_steps"], 
            num_training_steps=config["num_steps"]
        )
        criterion = nn.CrossEntropyLoss()
        total_token = 0
        tester = text_generator(self.model.module, self.tokenizer, self.config)
        mean_loss = 0
        accumulated_steps = self.config['accum_steps']
        loss_list = []
        token_count_list = []
        start_epoch = 0
        start_step = 0

        # ... 初始化 model, optimizer, scheduler ...
        checkpoint_path = os.path.join(self.config['output_dir'], 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            self.dataloader.sampler.set_epoch(start_epoch)
            total_token = checkpoint['token_count']
            checkpoint = None

        else:
            print("No checkpoint found, starting from scratch.")
        
        print("len(dataloader): ", len(self.dataloader))
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            if global_rank == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            self.dataloader.sampler.set_epoch(epoch)
            for i, x in enumerate(self.dataloader):
                if epoch == start_epoch and i < start_step:
                    if (i+1)%10000==0:
                        print(f"Skipping step {i+1}/{start_step}...")
                    continue
                is_accumulated = (i+1) % accumulated_steps == 0


                class able_context:
                    def __enter__(self): pass
                    def __exit__(self, *args): pass
                
                context = able_context() if is_accumulated else self.model.no_sync()


                x = x['input_ids'].to(self.device)  # (B, n)
                inputs = x[:, :-1]  # (B, n-1)
                labels = x[:, 1:]  # (B, n-1)
                with context:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(inputs)  # (B, n-1, V)
                        labels = labels.reshape(-1)  # (B*(n-1),)
                        logits = logits.reshape(-1, logits.size(-1))  # (B*(n-1), V)
                        loss = criterion(logits, labels) / accumulated_steps
                    
                    mean_loss += loss.item()
                    self.scaler.scale(loss).backward()
                    total_token += x.size(0) * x.size(1)
                if (i+1) % accumulated_steps == 0:
                    self.scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    loss_tensor = torch.tensor([mean_loss], device=self.device)
                    token_tensor = torch.tensor([total_token], device=self.device, dtype=torch.float) 
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
                    global_loss = loss_tensor.item() / world_size
                    global_token_count = token_tensor.item()

                    if global_rank == 0:
                        self.logger.info(f"i: {i+1}, token_count: {global_token_count / (2**30):.6f} B, Loss: {global_loss:.4f}")
                        with torch.no_grad(): 
                            torch.cuda.empty_cache()
                            text = tester.generate(prompt=None, max_len=60)
                        self.logger.info(f"Generated text: {text}")

                        loss_list.append(global_loss)
                        token_count_list.append(global_token_count/(2**30))
                        plt.plot(token_count_list, loss_list)
                        plt.xlabel('total_training_token(B)')
                        plt.ylabel('loss')
                        plt.savefig(os.path.join(self.config['output_dir'], 'loss_curve.png'))
                        plt.close()

                    dist.barrier() 
                    mean_loss = 0

                if (i+1) % config['save_steps'] == 0 and global_rank == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'step': i+1, # 当前 epoch 内的 step
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': global_loss,
                        'token_count': total_token
                    }
                    torch.save(checkpoint, os.path.join(self.config['output_dir'], f"checkpoint.pth"))
                    torch.save(self.model.module.state_dict(), os.path.join(self.config['output_dir'], f"model_base.pth"))
                    self.logger.info(f"Model saved to {os.path.join(self.config['output_dir'], f'model_base.pth')}")




if __name__ == '__main__':
    with open('./configs/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')


    model = Transformer(config)
 #   model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'model.pth'), map_location=f'cuda:{rank}'))
    dataloader, tokenizer = create_dataloader(config, rank=rank, world_size=world_size)
    trainer = Trainer(model, dataloader, tokenizer, config)
    trainer.train()

    dist.destroy_process_group()