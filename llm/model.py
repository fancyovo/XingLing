import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class RoPE(nn.Module):
    def __init__(self):
        """
        (B, m, n, d) -> (B, m, n, d)
        B: batch size
        m: number of heads
        n: sequence length
        d: feature dimension
        d should be even
        """
        super().__init__()
    
    def forward(self, x, base = 10000):
        device = x.device
        (B, m, n, d) = x.shape
        o = torch.arange(d//2).reshape(1, d // 2).to(device)
        o = 1 / (base ** (2 * o / d))
        I = torch.arange(n).reshape(n, 1).float().to(device)
        P = I @ o # (n, d/2)
        cosP = torch.cat((torch.cos(P), torch.cos(P)), dim=-1) # (n, d)
        sinP = torch.cat((torch.sin(P), torch.sin(P)), dim=-1) # (n, d)
        
        
        # R = torch.stack((
        #     torch.stack((cosP, sinP), dim=2),
        #     torch.stack((-sinP, cosP), dim=2)),
        #     dim=3
        # ) # (n, d/2, 2, 2)

        # R = R.reshape(1, 1, n, d // 2, 2, 2)
        # x = x.reshape(B, m, n, d // 2, 2, 1)

        # x = R @ x # (B, m, n, d/2, 2, 1)
        # x = x.reshape(B, m, n, d)

        rot_half_x = torch.cat((x[:, :, :, d//2:], -x[:, :, :, :d//2]), dim=-1) # (B, m, n, d)

        return x * cosP + rot_half_x * sinP

class SwishGLU(nn.Module):
    def __init__(self, config):
        """
        (B, n, C) -> (B, n, C)
        B: batch size
        n: sequence length
        C: feature dimension
        """
        super().__init__()
        C1 = config["latent_dim"]
        C2 = config["ffn_dim"]
        self.W1 = nn.Linear(C1, C2)
        self.W2 = nn.Linear(C2, C1)
        self.V = nn.Linear(C1, C2)
        self.silu = nn.SiLU()

    def forward(self, x):
        x1 = self.W1(x)
        x1 = self.silu(x1)
        x2 = self.V(x)
        x = x1 * x2
        return self.W2(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, config):
        """
        (B, n, C) -> (B, n, C)
        B: batch size
        n: sequence length
        C: feature dimension
        """
        super().__init__()
        C = config["latent_dim"]
        self.m = config["num_heads"]
        self.Q = nn.Linear(C, C)
        self.K = nn.Linear(C, C)
        self.V = nn.Linear(C, C)
        self.rope = RoPE()    
        self.attn = F.scaled_dot_product_attention    
        self.W = nn.Linear(C, C)

    def forward(self, x):
        B, n, C = x.shape
        m = self.m
        Q = self.Q(x).reshape(B, n, m, C//m).transpose(1, 2)
        K = self.K(x).reshape(B, n, m, C//m).transpose(1, 2)
        V = self.V(x).reshape(B, n, m, C//m).transpose(1, 2)
        
        Q = self.rope(Q)
        K = self.rope(K)

        O = self.attn(Q, K, V, is_causal=True)
        O = O.transpose(1, 2).contiguous().reshape(B, n, C)
        O = self.W(O)
        return O
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        """
        (B, n, C) -> (B, n, C)
        B: batch size
        n: sequence length
        C: feature dimension
        """
        super().__init__()
        C = config["latent_dim"]
        self.attn = AttentionBlock(config)
        self.ffn = SwishGLU(config)
        self.norm1 = nn.RMSNorm(C)
        self.norm2 = nn.RMSNorm(C)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        """
        (B, n) -> (B, n, V)
        B: batch size
        n: sequence length
        V: vocabulary size
        """
        super().__init__()
        self.embedding = nn.Embedding(config["vocabulary_size"], config["latent_dim"])
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["num_layers"])])
        
        self.norm = nn.RMSNorm(config["latent_dim"])

        torch.nn.init.normal_(self.embedding.weight, std=config["latent_dim"]**-0.5)
        self.gradient_checkpointing = True

    def forward(self, x, target = None):
        x = self.embedding(x)
        for block in self.blocks:
            if self.gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        if target is None:
            return x @ self.embedding.weight.T
        else:
            return x