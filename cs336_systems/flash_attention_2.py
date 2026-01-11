import timeit
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import hydra
from omegaconf import DictConfig
from loguru import logger as log

import cs336_basics
from cs336_basics.model import BasicsTransformerLM, softmax, CausalMultiHeadSelfAttention, RotaryEmbedding
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW


@hydra.main(config_path="conf", config_name="config", version_base=None)
def pytorch_attention(cfg: DictConfig):
    d_model = cfg.pytorch_attention.d_model
    context_length = cfg.pytorch_attention.context_length
    device = cfg.pytorch_attention.device
    num_heads = 1
    rope_theta = 10000
    d_head = d_model // num_heads
    positional_encoder = RotaryEmbedding(context_length=context_length, dim=d_head, theta=rope_theta)
    model = CausalMultiHeadSelfAttention(d_model, num_heads, positional_encoder).to(device)
    loss_fn = CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=cfg.optimizer.lr)

    for i in range(10):
        x = torch.randn([8, context_length, d_model], device=device)
        label = torch.randint(low=0, high=d_model, size=[x.shape[0] * x.shape[1]], device=device)
        y = model(x).reshape(-1, d_model)
        loss = loss_fn(y, label.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        torch.cuda.synchronize()

    for i in range(100):
        x = torch.randn([8, context_length, d_model], device=device)
        label = torch.randint(low=0, high=d_model, size=[x.shape[0] * x.shape[1]], device=device)
        timeit.default_timer()
        y = model(x).reshape(-1, d_model)
        forward_alloc = torch.cuda.memory_allocated() / 1024**2
        loss = loss_fn(y, label.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
        log.info(f"loss: {loss.item():.2f}")


if __name__ == "__main__":
    pytorch_attention()