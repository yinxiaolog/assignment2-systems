#!/bin/bash

source .venv/bin/activate
# uv run python -m cs336_systems.benchmark model.vocab_size=10000 model.d_model=768 model.num_layers=12 model.num_heads=12 model.d_ff=3072
# uv run nsys profile --python-backtrace=cuda -o /opt/log/ python -m cs336_systems.benchmark model.vocab_size=10000 model.d_model=768 model.num_layers=12 model.num_heads=12 model.d_ff=3072 model.warmup_step=0

sizes=("small" "medium" "large" "xl" "2.7B")
context_length=(128 256 512 1024)

function nsys_profile() {
    rm -rf /opt/log/cs336_systems/nsys_profile_*
    for size in "${sizes[@]}"; do
        for len in "${context_length[@]}"; do
            uv run \
            nsys profile --python-backtrace=cuda \
            -o "/opt/log/cs336_systems/nsys_profile_${size}_context_length_${len}" python -m \
            cs336_systems.benchmark model.size="${size}" \
            model.context_length="${len}" \
            nsys=true
        done
    done
}

function profile() {
    rm -rf /opt/log/cs336_systems/nsys_profile_*
    for size in "${sizes[@]}"; do
        for len in "${context_length[@]}"; do
            uv run \
            python -m \
            cs336_systems.benchmark model.size="${size}" \
            model.context_length="${len}" \
            nsys=false
        done
    done
}

profile