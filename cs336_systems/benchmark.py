import timeit
import math
from pathlib import Path

import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch import Tensor
from torch import nn
from jaxtyping import Float, Bool
import torch.cuda.nvtx as nvtx
from einops import einsum
import hydra
from omegaconf import DictConfig
from loguru import logger as log
import pandas as pd

import cs336_basics
from cs336_basics.model import BasicsTransformerLM, softmax
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention acores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def profile(cfg: DictConfig):
    vocab_size = cfg.model.vocab_size
    batch_size = cfg.model.batch_size
    d_model = cfg.size[cfg.model.size].d_model
    d_ff = cfg.size[cfg.model.size].d_ff
    num_layers = cfg.size[cfg.model.size].num_layers
    num_heads = cfg.size[cfg.model.size].num_heads
    rope_theta = cfg.model.rope_theta
    npa = np.arange(0, vocab_size, dtype=np.int32)
    context_length = cfg.model.context_length
    device = cfg.model.device
    model = BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta=rope_theta,
    ).to(device)
    loss_fn = CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=cfg.optimizer.lr)

    # warm-up
    for i in range(cfg.model.warmup_step):
        x, label = get_batch(npa, batch_size=batch_size, context_length=context_length, device=device)
        y = model(x).reshape(-1, vocab_size)
        loss = loss_fn(y, label.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        torch.cuda.synchronize()

    forward_times = []
    backward_times = []

    for i in range(cfg.model.meas_step):
        x, label = get_batch(npa, batch_size=batch_size, context_length=context_length, device=device)
        forward_start = timeit.default_timer()
        y = model(x).reshape(-1, vocab_size)
        loss = loss_fn(y, label.reshape(-1))
        forward_end = timeit.default_timer()
        torch.cuda.synchronize()
        optim.zero_grad()

        backward_start = timeit.default_timer()
        loss.backward()
        backward_end = timeit.default_timer()
        optim.step()
        torch.cuda.synchronize()
        forward_time = forward_end - forward_start
        backward_time = backward_end - backward_start

        forward_times.append(forward_time)
        backward_times.append(backward_time)

    times = torch.tensor([forward_times, backward_times])
    sum = times.sum(dim=-1)
    mean = times.mean(dim=-1)
    std = times.std(dim=-1)
    log.info(
        f"forward sum: {sum[0]:.5f}, forward mean: {mean[0]:.5f}, formard std: {std[0]:.5f}, backward sum: {sum[1]:.5f}, backward mean: {mean[1]:.5f}, backward std: {std[1]:.5f}"
    )
    return [round(x.item(), 5) for x in [sum[0], mean[0], std[0], sum[1], mean[1], std[1]]]


def profile_nsys(cfg: DictConfig):
    vocab_size = cfg.model.vocab_size
    batch_size = cfg.model.batch_size
    d_model = cfg.size[cfg.model.size].d_model
    d_ff = cfg.size[cfg.model.size].d_ff
    num_layers = cfg.size[cfg.model.size].num_layers
    num_heads = cfg.size[cfg.model.size].num_heads
    rope_theta = cfg.model.rope_theta
    npa = np.arange(0, vocab_size, dtype=np.int32)
    context_length = cfg.model.context_length
    device = cfg.model.device
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    model = BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta=rope_theta,
    ).to(device)
    loss_fn = CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=cfg.optimizer.lr)

    # warm-up
    with nvtx.range("warm-up"):
        for i in range(cfg.model.warmup_step):
            x, label = get_batch(npa, batch_size=batch_size, context_length=context_length, device=device)
            y = model(x).reshape(-1, vocab_size)
            loss = loss_fn(y, label.reshape(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    for i in range(cfg.model.meas_step):
        x, label = get_batch(npa, batch_size=batch_size, context_length=context_length, device=device)
        with nvtx.range("forward"):
            y = model(x).reshape(-1, vocab_size)
            loss = loss_fn(y, label.reshape(-1))
        optim.zero_grad()

        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("optim step"):
            optim.step()
        torch.cuda.synchronize()

        forward_times.append(0.0)
        backward_times.append(0.0)

    times = torch.tensor([forward_times, backward_times])
    sum = times.sum(dim=-1)
    mean = times.mean(dim=-1)
    std = times.std(dim=-1)
    log.info(
        f"forward sum: {sum[0]:.5f}, forward mean: {mean[0]:.5f}, formard std: {std[0]:.5f}, backward sum: {sum[1]:.5f}, backward mean: {mean[1]:.5f}, backward std: {std[1]:.5f}"
    )
    return [round(x.item(), 5) for x in [sum[0], mean[0], std[0], sum[1], mean[1], std[1]]]


def profile_nsys_mixed_precision(cfg: DictConfig):
    vocab_size = cfg.model.vocab_size
    batch_size = cfg.model.batch_size
    d_model = cfg.size[cfg.model.size].d_model
    d_ff = cfg.size[cfg.model.size].d_ff
    num_layers = cfg.size[cfg.model.size].num_layers
    num_heads = cfg.size[cfg.model.size].num_heads
    rope_theta = cfg.model.rope_theta
    npa = np.arange(0, vocab_size, dtype=np.int32)
    context_length = cfg.model.context_length
    device = cfg.model.device
    model = BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta=rope_theta,
    ).to(device)
    loss_fn = CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=cfg.optimizer.lr)

    # warm-up
    with nvtx.range("warm-up"):
        for i in range(cfg.model.warmup_step):
            x, label = get_batch(npa, batch_size=batch_size, context_length=context_length, device=device)
            y = model(x).reshape(-1, vocab_size)
            loss = loss_fn(y, label.reshape(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    for i in range(cfg.model.meas_step):
        with torch.autocast(device_type=device, dtype=torch.float32):
            x, label = get_batch(npa, batch_size=batch_size, context_length=context_length, device=device)
            with nvtx.range("forward"):
                y = model(x).reshape(-1, vocab_size)
                loss = loss_fn(y, label.reshape(-1))
            optim.zero_grad()

            with nvtx.range("backward"):
                loss.backward()
            with nvtx.range("optim step"):
                optim.step()
            torch.cuda.synchronize()

            forward_times.append(0.0)
            backward_times.append(0.0)

    times = torch.tensor([forward_times, backward_times])
    sum = times.sum(dim=-1)
    mean = times.mean(dim=-1)
    std = times.std(dim=-1)
    log.info(
        f"forward sum: {sum[0]:.5f}, forward mean: {mean[0]:.5f}, formard std: {std[0]:.5f}, backward sum: {sum[1]:.5f}, backward mean: {mean[1]:.5f}, backward std: {std[1]:.5f}"
    )
    return [round(x.item(), 5) for x in [sum[0], mean[0], std[0], sum[1], mean[1], std[1]]]


def benchmark(cfg: DictConfig, nsys: bool = True):
    result = []
    d_model = cfg.size[cfg.model.size].d_model
    d_ff = cfg.size[cfg.model.size].d_ff
    num_layers = cfg.size[cfg.model.size].num_layers
    num_heads = cfg.size[cfg.model.size].num_heads

    result.append(
        [
            cfg.model.size,
            d_model,
            d_ff,
            num_layers,
            num_heads,
            cfg.model.context_length,
        ]
        + (profile_nsys_mixed_precision(cfg) if nsys else profile(cfg))
    )

    df = pd.DataFrame(
        result,
        columns=[
            "Size",
            "d_model",
            "d_ff",
            "num_layers",
            "num_heads",
            "context_length",
            "forward sum",
            "forward mean",
            "forward std",
            "backward sum",
            "backward mean",
            "backward std",
        ],
    )
    print(df)
    df.to_markdown(
        Path(cfg.log.dir, f"{cfg.model.size}-{d_model}-{d_ff}-{num_layers}-{num_heads}-{cfg.model.context_length}.md")
    )


def mixed_precision_accumultion():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float16)
    print(s)


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print(f"fc1 dtype: {x.dtype}, {x.device}")
        x = self.relu(x)
        x = self.ln(x)
        print(f"ln dtype: {x.dtype}, {x.device}")
        x = self.fc2(x)
        print(f"logits dtype: {x.dtype}, {x.device}")
        return x


def benchmarking_mixed_precision():
    model: nn.Module = ToyModel(1000, 5)
    dtype: torch.dtype = torch.float32
    x: torch.Tensor = torch.randn(1000, 1000)
    y: torch.Tensor = torch.randint(0, 4, (1000, 1))
    loss_fn = CrossEntropyLoss()
    optim = AdamW(model.parameters())
    print(model.parameters())
    device = "cuda"
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    for i in range(1):
        with torch.autocast(device_type="cuda", dtype=dtype):
            y_hat = model(x)
            loss = loss_fn(y_hat, y.reshape(-1))
            optim.zero_grad()
            loss.backward()
            print(f"loss dtype: {loss.dtype} , {loss.device}")
            optim.step()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    print(name, p.grad.dtype, p.grad.device)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    Path(cfg.log.dir).mkdir(parents=True, exist_ok=True)
    # cs336_basics.model.scaled_dot_product_attention = (
    #     annotated_scaled_dot_product_attention
    # )
    # profile(cfg)
    # mixed_precision_accumultion()
    # benchmarking_mixed_precision()
    benchmark(cfg, nsys=cfg.nsys)


if __name__ == "__main__":
    main()
