import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from llm.model import Transformer
import yaml
import os

# ================= 配置区域 =================

CONFIG_PATH = "./configs/config.yaml"
MODEL_PATH = "./results/model_sft.pth" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 特殊 Token 定义
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# ===========================================

def load_model():
    print(f"正在加载配置: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], trust_remote_code=True)
    
    print(f"正在加载模型权重: {MODEL_PATH}")
    model = Transformer(config).to(DEVICE)
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if list(state_dict.keys())[0].startswith('module.'):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = new_state_dict
        
    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载完成！")
    return model, tokenizer, config

def generate(model, tokenizer, input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
    curr_ids = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(curr_ids)
            next_token_logits = logits[:, -1, :]
            
            # --- 重复惩罚 ---
            if repetition_penalty != 1.0:
                for i in range(curr_ids.shape[0]):
                    unique_seen_tokens = torch.unique(curr_ids[i])
                    seen_logits = next_token_logits[i, unique_seen_tokens]
                    updated_logits = torch.where(
                        seen_logits < 0, 
                        seen_logits * repetition_penalty, 
                        seen_logits / repetition_penalty
                    )
                    next_token_logits[i, unique_seen_tokens] = updated_logits
            
            # --- 采样 ---
            next_token_logits = next_token_logits / temperature
            
            # Top-P
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # --- 结束判断 ---
            token_id = next_token.item()
            # 兼容 EOS 和 IM_END
            if token_id == tokenizer.eos_token_id or token_id == tokenizer.convert_tokens_to_ids(IM_END):
                break
                
    return curr_ids

def main():
    model, tokenizer, config = load_model()
    
    # 预先获取特殊 Token ID，确保和 Dataloader 一致
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    
    history = [] 
    
    print("\n" + "="*30)
    print("星灵对话系统启动！(输入 'exit' 退出, 'clear' 清空历史)")
    print("="*30 + "\n")
    
    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() == 'exit':
            break
        if user_input.lower() == 'clear':
            history = []
            print("[历史记录已清空]")
            continue
        if not user_input:
            continue
            
        # --- 构建 Prompt (Token ID 拼接模式) ---
        # 这种方式能保证和训练时看到的 Token 序列 100% 一致
        input_ids = []
        
        # 1. 拼接历史
        for role, content in history:
            role_ids = tokenizer.encode(role, add_special_tokens=False)
            content_ids = tokenizer.encode(content, add_special_tokens=False)
            
            # <|im_start|> + role + \n + content + <|im_end|> + \n
            input_ids += [im_start_id] + role_ids + nl_ids + content_ids + [im_end_id] + nl_ids
        
        # 2. 拼接当前 User 输入
        user_role_ids = tokenizer.encode("user", add_special_tokens=False)
        user_content_ids = tokenizer.encode(user_input, add_special_tokens=False)
        input_ids += [im_start_id] + user_role_ids + nl_ids + user_content_ids + [im_end_id] + nl_ids
        
        # 3. 拼接 Assistant 引导头 (<|im_start|>assistant\n)
        assist_role_ids = tokenizer.encode("assistant", add_special_tokens=False)
        input_ids += [im_start_id] + assist_role_ids + nl_ids
        
        # 转 Tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
        
        # 长度保护
        if input_tensor.shape[1] > config['max_seq_len'] - 100:
            print("[警告] 上下文过长，清空历史...")
            history = []
            continue

        print("星灵: ", end="", flush=True)
        
        # --- 生成 ---
        output_ids = generate(
            model, 
            tokenizer, 
            input_tensor, 
            max_new_tokens=300, 
            temperature=0.7, 
            top_p=0.9, 
            repetition_penalty=1.1
        )
        
        # --- 解码 ---
        new_tokens = output_ids[0][len(input_ids):]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(response)
        
        history.append(("user", user_input))
        history.append(("assistant", response))

if __name__ == "__main__":
    main()
