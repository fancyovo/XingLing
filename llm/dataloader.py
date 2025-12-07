import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import yaml
import os
import glob
import random
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class PackedDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_seq_len, rank=0, world_size=1):
        """
        Args:
            dataset: text data
            tokenizer: a Hugging Face tokenizer, like "Qwen/Qwen3-8B"
            max_seq_len: maximum sequence length
            rank: rank of the current GPU
            world_size: number of GPUs used for training
        Note:
            for muilti-GPU training
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = tokenizer.eos_token_id
        self.rank = rank
        self.world_size = world_size
        

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers_per_gpu = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        total_workers = self.world_size * num_workers_per_gpu 
        global_worker_id = self.rank * num_workers_per_gpu + worker_id

        sharded_dataset = self.dataset.shard(num_shards=total_workers, index=global_worker_id)
        iterator = iter(sharded_dataset)

        buffer = []
        for sample in iterator:
            text = sample['text']
            token_ids = self.tokenizer.encode(text)
            token_ids.append(self.tokenizer.eos_token_id)
            buffer.extend(token_ids)
            while len(buffer) >= self.max_seq_len:
                yield torch.tensor(buffer[:self.max_seq_len], dtype=torch.long)
                buffer = buffer[self.max_seq_len:]

        return None
    

def create_dataloader(config, rank=0, world_size=1):
    """
    Args:
        config: a dictionary of configuration parameters
        rank: rank of the current GPU
        world_size: number of GPUs used for training
    Returns:
        a DataLoader object
        tokenizer
    """
    # 直接加载处理好的数据
    dataset = load_from_disk(os.path.join(config['pretrain_data_dir'], 'processed_data')) 
    
    # 设置格式为 torch，这样取出来直接是 tensor
    dataset.set_format(type='torch', columns=['input_ids'])
    # 使用 DistributedSampler 来处理多卡数据分配
    # 它会自动处理 shuffle，而且支持 set_epoch 来保证随机性
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], trust_remote_code=True)
    return dataloader, tokenizer

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for i, batch in enumerate(create_dataloader(config)):
        print(f"Batch {i}: {batch.shape}")
        if i == 10:
            break