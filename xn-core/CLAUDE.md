# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

mimi-rs is a Rust tensor library implementing neural network operations with a focus on audio processing models. It uses Rust 2024 edition and supports both CPU and CUDA backends.

## Build Commands

```bash
# Build with CUDA support (default)
cargo build --release

# Build CPU-only (no CUDA required)
cargo build --release --no-default-features

# Run tests
cargo test

# Run a single test
cargo test test_name

# Run examples
cargo run --release --example mimi -- input.wav -o output.wav
cargo run --release --example llama
cargo run --release --example basic_cuda  # requires CUDA
```

## Architecture

### Core Tensor System

The library uses a generic `Tensor<T, B>` type where:
- `T: WithDType` - the element type (f32, f16, bf16, i64, u8)
- `B: Backend` - the compute backend (CPU via `()` or CUDA via `cuda_backend::Device`)

Key types:
- `CpuTensor<T>` = `Tensor<T, ()>` - convenience alias for CPU tensors
- `WithDTypeF` - trait for float types that support transcendental functions

### Backend Trait (`src/backend.rs`)

The `Backend` trait defines all low-level operations. Implementations:
- `src/cpu_backend.rs` - CPU backend using `()` as the device type
- `src/cuda_backend.rs` - CUDA backend using `cuda_backend::Device`

Operations are split into:
- Basic ops: add, mul, copy, fill, transpose
- Float ops (require `WithDTypeF`): softmax, rms_norm, layer_norm, rope, conv1d, etc.
- In-place variants in `src/inplace_ops.rs` with `_` suffix (e.g., `add_`, `matmul_`)

### CUDA Kernels (`cuda-kernels/`)

Custom CUDA kernels compiled via `bindgen_cuda`:
- `arithmetic.cu` - binary/unary ops, scale
- `fill.cu` - memory fill
- `indexing.cu` - index select operations
- `layout.cu` - transpose
- `reduce.cu` - softmax, rms_norm, layer_norm, reductions
- `rope.cu` - rotary position embeddings

Helper headers:
- `cuda_utils.cuh` - strided indexing, math helpers
- `compatibility.cuh` - CUDA FP16/BF16 includes

PTX is loaded at runtime via `src/cuda_kernels.rs`.

### Neural Network Layers (`src/nn/`)

- `Linear`, `RmsNorm` - basic layers
- `VB` (VarBuilder) - loads weights from safetensors files with automatic dtype conversion

### Models (`src/models/`)

- `llama.rs` - Llama architecture with KV-cache support
- `mimi.rs` - Mimi audio tokenizer (encoder/decoder for audio compression)
- `demucs.rs` - Demucs audio source separation model

### Streaming Support

`mimi.rs` defines streaming primitives:
- `StreamTensor<T, B>` - optional tensor for streaming contexts
- `StreamMask` - batch element mask
- `StreamingModule` trait - step-by-step processing with state

### Shape System (`src/shape.rs`)

Supports dynamic shapes with:
- `D` type for dimension indexing (supports negative indices)
- `ShapeWithOneHole` - allows `()` as placeholder in reshape (e.g., `(3, ())` infers second dim)

## Conventions

- Operations return new tensors; in-place variants have `_` suffix
- Use `cargo clippy` - the codebase has specific clippy configurations
- Tests are in `tests/tensor_tests.rs` for core tensor operations
