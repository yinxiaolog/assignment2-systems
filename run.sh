#!/bin/bash

source .venv/bin/activate
uv run python -m cs336_systems.benchmark model.vocab_size=10000 model.d_model=768 model.num_layers=12 model.num_heads=12 model.d_ff=3072
# uv run nsys profile --python-backtrace=cuda -o result python -m cs336_systems.benchmark model.vocab_size=10000 model.d_model=768 model.num_layers=12 model.num_heads=12 model.d_ff=3072 model.warmup_step=0