# Rust AI: Candle in Huggingface - Minimalist ML framework

## Candle Source
- https://github.com/huggingface/candle 

## How to install Candle
- Search site
- Git clone

## How to run the examples

### Phi-1, 1.5, 2
```bash
$ cargo run --example phi --release -- --prompt "def print_prime(n): "
```

### T5: t5-small, MADLAD-400
```bash
$ cargo run --example t5 --release -- --model-id "t5-small" --prompt "translate to German: A beautiful candle." --decode

$ cargo run --example t5 --release  -- \
    --model-id "jbochi/madlad400-3b-mt" \
    --prompt "<2de> How are you, my friend?" \
    --decode --temperature 0
```

## AI Training

### MNIST (model: linear, mlp, cnn)
```bash
$ cargo run --example mnist-training -- linear --epochs 10
$ cargo run --example mnist-training -- mlp --epochs 5
$ cargo run --example mnist-training -- cnn --epochs 5
$ cargo run --example mnist-training --features cuda -- cnn --epochs 5
```