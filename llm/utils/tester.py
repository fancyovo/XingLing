import torch
import sys

class text_generator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate(self, prompt, max_len=100, stream = False):
        if prompt is None:
            prompt = "这"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        input_ids = input_ids.reshape(1, -1)
        output_ids = input_ids
        if stream:
            print(prompt, end='')
        for i in range(max_len):
            output = self.model(output_ids)
            output = output[:, -1, :]
            output[0, self.tokenizer.eos_token_id] = -100
            output = torch.softmax(output, dim=-1)  ** (1/0.8)
            output = output / output.sum(dim=-1, keepdim=True)
            # 依赖output的概率来随机
            output_id = torch.multinomial(output, 1)
            if stream:
            #    print(f"output_id: {output_id.squeeze().tolist()}, output_text: {self.tokenizer.decode(output_id.squeeze().tolist())}, prob: {output[0, output_id.squeeze().tolist()].item()}")
                print(self.tokenizer.decode(output_id.squeeze().tolist()), end='')
                sys.stdout.flush()
            output_ids = torch.cat([output_ids, output_id], dim=1)
            # if output_ids[0, -1] == self.tokenizer.eos_token_id:
            #     break
        output_text = self.tokenizer.decode(output_ids.squeeze().tolist(), skip_special_tokens=True)
        return output_text