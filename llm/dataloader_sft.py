import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import copy
import os
import yaml

# 定义特殊的 ChatML 格式 token
# Qwen Tokenizer 通常自带这些，但为了保险我们手动定义模板逻辑
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_line = json.loads(line)
                if line.strip() and len(line)<1000 and not ("用户" in line):
                    self.data.append(json_line)
        
        print(f"加载了 {len(self.data)} 条对话数据")
        self.im_start_id = self.tokenizer.convert_tokens_to_ids(IM_START)
        self.im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        
        input_ids = []
        labels = []
        
        # 这里的逻辑是：拼接对话，并构建掩码
        # 格式: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>\n
        
        im_start_id = self.tokenizer.convert_tokens_to_ids(IM_START)
        im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)
        nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        
        for msg in messages:
            try:
                role = msg["role"]
                content = msg["content"]
            except:
                print(f"数据格式错误，请检查 {messages}" + '-' * 100)
                continue
            # 构建当前句子的文本
            # 注意：这里手动拼接了格式，你也可以用 tokenizer.apply_chat_template
            
            role_ids = self.tokenizer.encode(role, add_special_tokens=False)
            content_ids = self.tokenizer.encode(content, add_special_tokens=False)

            # [im_start_id] + role_ids + nl_ids + content_ids + [im_end_id] + nl_ids
                    
            input_ids.extend([im_start_id])
            labels.extend([-100])

            input_ids.extend(role_ids + nl_ids + content_ids)
            msg_len = len(role_ids + nl_ids + content_ids)

            
            if role == "user":
                # 用户说的话，不计算 Loss，Label 全设为 -100
                labels.extend([-100] * msg_len)
            elif role == "assistant":
                # 模型说的话，计算 Loss，Label 就是 input_ids
                labels.extend([-100] * len(role_ids + nl_ids) + content_ids)
            else:
                # 系统提示词等，通常也不计算 Loss
                labels.extend([-100] * msg_len)

            input_ids.extend([im_end_id] + nl_ids)
            labels.extend([im_end_id] + [-100])


        # 截断
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def sft_collate_fn(batch):
    # 动态 Padding
    max_len = max([len(item["input_ids"]) for item in batch])
    
    # 假设 pad_token_id 是 0 或者 tokenizer.pad_token_id
    # Qwen 的 pad_token_id 可能是 None，需要手动指定，通常用 eos_token_id 或者 0
    pad_id = 0 
    
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    
    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]
        length = len(input_ids)
        
        # Padding input_ids
        pad_len = max_len - length
        padded_input = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
        
        # Padding labels (用 -100 填充)
        padded_labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
        
        # Attention Mask (1 for real tokens, 0 for padding)
        # 注意：如果是 Causal LM，通常不需要手动传 attention_mask 给 model，
        # 除非你的 model forward 里处理了 padding mask。
        # 简单的 GPT 类模型通常只用 causal mask，padding 部分通过 label=-100 忽略 loss 即可。
        # 但为了严谨，这里生成 mask。
        mask = torch.cat([torch.ones(length, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        
        batch_input_ids.append(padded_input)
        batch_labels.append(padded_labels)
        batch_attention_mask.append(mask)
        
    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attention_mask)
    }

def create_sft_dataloader(config):

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], trust_remote_code=True)

    DATA_PATH = os.path.join(config["sft_data_dir"], "sft.jsonl")
   # DATA_PATH = "../data/sft.jsonl"


    dataset = SFTDataset(
        data_path=DATA_PATH, 
        tokenizer=tokenizer,
        max_seq_len=config["max_seq_len"]
    )
    
    # 确保 tokenizer 有 pad_token，如果没有就指定一个
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return DataLoader(
        dataset, 
        batch_size=config["sft_batch_size"], 
        shuffle=True, 
        collate_fn=sft_collate_fn,
        num_workers=2
    ), tokenizer

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dl, tokenizer = create_sft_dataloader(config)

    for i, batch in enumerate(dl):
        print(f"Batch {i}: {batch['input_ids'].shape}")
        x = batch['input_ids'][0]
        y = batch['labels'][0]
        for j in range(x.shape[0]):
            print(f"{x[j].tolist()}, {y[j].tolist()}, {tokenizer.decode(x[j].tolist())}")
        break
        if i == 10:
            break