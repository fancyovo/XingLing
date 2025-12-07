import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from llm.utils.logger import setup_logging
import yaml
from llm.model import Transformer
from llm.dataloader_sft import create_sft_dataloader
from llm.utils.tester import text_generator
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class SFT_Trainer:
    def __init__(self, model, dataloader, tokenizer, config):
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.config = config

    def train(self):
        self.device = torch.device(f"cuda")
        self.model.to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['sft_lr'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.config['sft_epochs'] * len(self.dataloader) // self.config['sft_accum_steps'] // 20,
            num_training_steps=self.config['sft_epochs'] * len(self.dataloader) // self.config['sft_accum_steps'],
        )
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()

        loss_list = []
        token_count_list = []
        total_token = 0

        

        tester = text_generator(self.model, self.tokenizer, self.config)

        self.logger = setup_logging(self.config['output_dir'], log_file='sft_training_log.log')

        for epoch in range(self.config['sft_epochs']):
            mean_loss = 0
            optimizer.zero_grad()
            for i, data in enumerate(self.dataloader):
                x = data['input_ids'].to(self.device)
                y = data['labels'].to(self.device)

                x = x.to(self.device)  # (B, n)
                inputs = x[:, :-1]  # (B, n-1)
                labels = y[:, 1:]  # (B, n-1)
                
                logits = self.model(inputs)  # (B, n-1, V)
                labels = labels.reshape(-1)  # (B*(n-1),)
                logits = logits.reshape(-1, logits.size(-1))  # (B*(n-1), V)
                loss = criterion(logits, labels)
                loss = loss / self.config['sft_accum_steps']
                loss.backward()
                mean_loss += loss.item()
                total_token += x.shape[0] * x.shape[1] / (2**20)


                if (i+1)%self.config['sft_accum_steps']==0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    loss_list.append(mean_loss)
                    token_count_list.append(total_token)
                    mean_loss = 0
                    self.logger.info(f"Epoch {epoch+1}/{self.config['sft_epochs']}, Step {i+1}/{len(self.dataloader)}, lr: {scheduler.get_last_lr()[0]:.7f}, Loss: {loss_list[-1]}")
                    with torch.no_grad(): 
                        text = tester.generate(prompt="<|im_start|>user\n星灵你好<|im_end|><|im_start|>assistant\n", max_len=50, stream=False)
                    
                    
                    self.logger.info(text)
                    plt.plot(token_count_list, loss_list)
                    plt.xlabel('Token count(M)')
                    plt.ylabel('SFT_Loss')
                    plt.savefig(os.path.join(self.config['output_dir'], 'sft_loss_curve.png'))
                    plt.clf()

            
            torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], f"model_sft.pth"))

if __name__ == '__main__':
    with open('./configs/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = Transformer(config)
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'model_base.pth'), map_location=f'cuda'))
    dataloader, tokenizer = create_sft_dataloader(config)

    trainer = SFT_Trainer(model, dataloader, tokenizer, config)
    trainer.train()



