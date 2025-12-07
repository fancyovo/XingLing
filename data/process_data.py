import os
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing
import yaml

def main():
    with open("./configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_path = config["pretrain_data_dir"]
    save_path = os.path.join(config["pretrain_data_dir"], "processed_data")
    tokenizer_path = config["tokenizer"]
    max_seq_len = config["max_seq_len"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # 确保有 pad_token

    # 1. 加载数据 (streaming=False, 直接加载)
    # 如果你的 jsonl 文件非常多且巨大，load_dataset 会自动处理内存映射，不会爆内存
    dataset = load_dataset("json", data_dir=data_path, split="train", streaming=False)

    # 2. 定义处理函数：Tokenize + Grouping
    # 这里我们把所有文本拼起来，然后按 max_seq_len 切分，这是预训练的标准做法
    def group_texts(examples):
        # 拼接所有文本
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples['input_ids'])
        
        # 丢弃最后那一点不够 max_seq_len 的余数
        if total_length >= max_seq_len:
            total_length = (total_length // max_seq_len) * max_seq_len
            
        # 切分
        result = {
            k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    print("Start Tokenizing...")
    # 先把文本转成 ID
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=16, # 利用多核 CPU 加速
        remove_columns=["text"], # 移除原始文本，省空间
        desc="Tokenizing"
    )

    print("Start Grouping...")
    # 再把 ID 拼起来切分
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=16,
        desc="Grouping"
    )

    print(f"Saving to {save_path}...")
    lm_dataset.save_to_disk(save_path)
    print("Done!")

if __name__ == "__main__":
    main()
