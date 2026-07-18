# Vearo

A deep learning training framework written from scratch in Rust. No PyTorch, no
LibTorch, no ONNX - the tensor library, the autograd engine, the CUDA kernels and
the optimizers are all in this repo.

It trains real models end to end on CPU and on NVIDIA GPUs, and the two backends
produce the same numbers.

## What works

Core:

- Tensors over a 32-way sharded arena allocator, f32, with refcounted storage
- Broadcasting elementwise ops, batched matmul
- Views: reshape, transpose, permute, contiguous
- Eager reverse-mode autograd with a generation-tagged tape (survives multi-step
  training loops)
- Device-keyed backend dispatch, so the same code runs on CPU or CUDA

Ops (forward and backward, all gradient-checked):

- add, sub, mul, div, matmul
- relu, gelu, softmax, layernorm
- sum, mean, embedding, cross entropy
- conv2d, maxpool2d, avgpool2d, batchnorm2d, dropout

Layers and training:

- Linear, Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d, Dropout, Embedding,
  LayerNorm, MultiHeadAttention, transformer block, a small GPT
- SGD and AdamW, cosine LR schedule, global-norm gradient clipping
- Thread-local train/eval mode, so dropout and batchnorm behave correctly at
  evaluation time

CUDA backend:

- Hand-written kernels compiled to PTX and loaded at runtime (no cuDNN)
- Every op above runs on the GPU
- CPU and CUDA training produce matching loss curves on the same model and data

## Verification

Correctness is checked, not assumed:

- Analytical gradients are checked against numerical (central difference)
  gradients for every op with a backward pass
- Forward ops are checked against a NumPy oracle
- CPU and CUDA results are compared op by op, including with non-contiguous
  inputs (a class of bug that silently corrupts training)
- Memory regression tests assert that repeated training steps do not grow host
  or device allocations
- Models are trained end to end and checked for convergence, not just for
  matching numbers on a single step

## Quickstart

```
git clone https://github.com/razecrs/vearo
cd vearo
cargo test                                       # unit + parity + regression tests
cargo run --release -p vearo --example train_mlp # trains a small MLP, prints loss falling
```

`train_mlp` needs no dataset: it generates synthetic data, trains a 2-32-32-1 MLP,
and asserts the loss decreases. If it prints a falling loss, your setup works.

Here is the whole training loop, which is the API in miniature:

```rust
use vearo::nn::{Linear, Module};
use vearo::{Device, Tensor};

vearo::backend_cpu::init();   // register a backend
vearo::autograd::init();      // and the tape

let fc = Linear::new(4, 1, true, 42);
let mut opt = vearo::optim::AdamW::new(fc.parameters(), 1e-3, 0.9, 0.999, 1e-8, 0.0);

let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [1, 4]).to(Device::Cpu);
let y = Tensor::from_f32(&[1.0], [1, 1]).to(Device::Cpu);

vearo::autograd::zero_gradients();
vearo::autograd::reset_active_tape();

let diff = fc.forward(&x).sub(&y);
let loss = diff.mul(&diff).mean(0, false);
loss.backward();
opt.step();
```

Swap `Device::Cpu` for `Device::Cuda(0)` and the same code runs on the GPU.

## Build requirements

The umbrella crate links the CUDA backend, so a CUDA toolkit and an NVIDIA GPU are
currently required to build the workspace, even for CPU-only use. (Making CUDA an
optional cargo feature is a good first contribution.)

Kernels ship as prebuilt PTX, so `nvcc` is only needed if you change `kernels.cu`:

```
nvcc --ptx crates/vearo-backend-cuda/src/kernels.cu \
     -o crates/vearo-backend-cuda/src/kernels.ptx
```

## Datasets (optional)

Most tests are self-contained. The benchmark and training tests need real datasets
and are marked `#[ignore]`, so `cargo test` stays green without them.

To run them, fetch and preprocess the data (needs the Kaggle CLI, authenticated,
and `pip install numpy pandas pillow`):

```
./scripts/setup_data.sh              # 32x32 images
./scripts/setup_data.sh --size 64    # higher resolution

export VEARO_DATA_DIR="$PWD/data/kaggle"
cargo test --release -p vearo --test kaggle_bakeoff -- --ignored --nocapture
```

Data lands in `data/`, which is gitignored. The MNIST tests expect
`data/mnist_train.csv` (the standard CSV format: label first, then 784 pixels).

## Where to start reading

```
crates/vearo/examples/train_mlp.rs   a complete training loop, start here
crates/vearo-core/src/tensor.rs      Tensor, dispatch, autograd hooks
crates/vearo-backend-cpu/src/lib.rs  reference implementation of every op
crates/vearo-autograd/src/lib.rs     the tape, backward, numerical_grad
```

Adding an op touches six places: the CPU backend, the `BackendOps` struct, the
`Tensor` dispatch method, an `OpType` plus backward arm in autograd, the CUDA
kernel and its dispatch, and a gradient check. Following an existing op such as
`avgpool2d` end to end is the fastest way to see the pattern.

Anything numerical must come with a gradient check and CPU/CUDA parity test. The
CUDA kernels assume contiguous inputs unless they explicitly handle strides, so
guard new kernels with `contiguous()` or handle strides and test it.

## Layout

```
crates/vearo-core          tensors, shapes, dtypes, the arena allocator, dispatch
crates/vearo-backend-cpu   CPU op implementations (the reference backend)
crates/vearo-backend-cuda  CUDA kernels and device memory management
crates/vearo-autograd      the tape, backward passes, numerical grad checking
crates/vearo-nn            layers and modules
crates/vearo-optim         optimizers and schedules
crates/vearo               umbrella crate; call vearo::init() then use it
```

## Status

The op set and both backends are complete and verified. Current work is on the
memory side of training: reducing the peak memory a training run needs, without
giving up throughput.

## License

AGPL-3.0-or-later. Use it, build on it, even sell it, just keep it open source
and keep the credit. See LICENSE.
