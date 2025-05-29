import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
import tiktoken
import time

@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50304

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.INIT_FLAG=1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x) # B, T, 3 * C
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, hs
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, hs
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, hs

        # the following 4 lines got replace by flash attention 
        wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # # B, nh, T, T
        wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # # B, nh, T, T
        out = wei @ v # B, nh, T, hs

        # out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.c_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.INIT_FLAG=1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.weight_init)

    def weight_init(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'INIT_FLAG'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, std=std, mean=0.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=std, mean=0.0)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # T
        pos_embd = self.transformer.wpe(pos) # T, C
        tok_embd = self.transformer.wte(idx) # B, T, C
        x = pos_embd + tok_embd # B, T, C
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss
            
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizer(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: n for pn, n in self.named_parameters()}
        param_dict = {pn: n for pn, n in param_dict.items() if n.requires_grad}

        decay_params = [n for pn, n in param_dict.items() if n.dim() >= 2]
        non_decay_params = [n for pn, n in param_dict.items() if n.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay}, 
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_non_decay_params = sum(p.numel() for p in non_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(non_decay_params)}, with {num_non_decay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        self.current_position = 0
        with open("shakes.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        # data = enc.encode(text[:1000])
        data = enc.encode(text)
        self.tokens = torch.tensor(data, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x, y = (buf[:-1]).view(B, T), (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x,y
    

max_steps = 30
warmup_steps = 15
max_lr = 6e-4
min_lr = 0.1 * max_lr

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    elif it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
    

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
    
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model = model.to(device)
# model = torch.compile(model)

total_batch_size = 1024
B=4
T=32
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)

dataLoader = DataLoaderLite(B=4, T=32)

# torch.set_float32_matmul_precision('high')

optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device_type=device)

for i in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for steps in range(grad_accum_steps):
        x, y = dataLoader.next_batch()
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        #     logits, loss = model(x, y)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    token_per_second = (dataLoader.B * dataLoader.T * grad_accum_steps) // (t1 - t0)
    print(f"epoch = {i}: loss = {loss_accum.item()} | norm= {norm:.4f} | lr= {lr} | dt = {dt} | tok/sec= {token_per_second}")

import sys; sys.exit(0)

print("didn't crash yay!")


model.eval()
max_length = 30
num_return_sequences = 5

enc = tiktoken.get_encoding('gpt2')
token = enc.encode("Hello, I'm a language model,")
token = torch.tensor(token, dtype=torch.long)
token = token.unsqueeze(0).repeat(num_return_sequences, 1)

x = token.to(device)

while x.shape[1] < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = F.softmax(logits, -1)
        topK, topKInd = torch.topk(probs, 50)
        ix = torch.multinomial(topK, num_samples=1)
        ids = torch.gather(topKInd, dim=-1, index=ix)
        x = torch.cat((x, ids), dim=1)

for i in range(num_return_sequences):
    tokens = x[i].tolist()
    print(enc.decode(tokens))
    print("\n") 