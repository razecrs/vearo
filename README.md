# Vearo

A deep learning training framework written from scratch in Rust.

Right now it's a CPU reference backend, correctness first. Everything runs on the
CPU and is covered by tests. GPU support is planned but not here yet.

It's early: the tensor core and autograd work, but there's no training loop yet.

## What works

- Tensors backed by a sharded arena allocator (f32 for now)
- Broadcasting elementwise ops (add/sub/mul/div) and batched matmul
- Views: reshape, transpose, permute (no copy where possible)
- Reverse-mode autograd with a tape; backward() works
- Grad-checked backward for: add, sub, mul, div, matmul, reshape, transpose,
  permute, relu, sum, mean
- Verification baked in: every backward pass is checked against numerical
  gradients, and forward ops are checked against NumPy

## Not here yet

- GPU backends
- More ops (softmax, gelu, layernorm, embedding, cross entropy)
- Optimizers, nn modules, an actual training loop

## Build

```
cargo build
cargo test
```

No GPU or CUDA toolkit needed; it all runs on the CPU.

## Layout

```
crates/vearo-core         tensors, shapes, dtypes, the arena allocator
crates/vearo-backend-cpu  CPU op implementations (the reference backend)
crates/vearo-autograd     the tape and backward passes
crates/vearo-nn           nn modules (empty for now)
crates/vearo-optim        optimizers (empty for now)
crates/vearo              umbrella crate; call vearo::init() then use it
```

## License

AGPL-3.0-or-later. Use it, build on it, even sell it, just keep it open source
and keep the credit. See LICENSE.
