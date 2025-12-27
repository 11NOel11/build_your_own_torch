# build_your_own_torch

**Goal:** Learn deep learning internals by re-building the core ideas of PyTorch from scratch:

- Autograd engine
- Tensors
- Neural network modules
- Optimizers (SGD, Adam, AdamW, Muon, etc.)
- Training loops on simple tasks

This repo is meant to be educational and beginner-friendly, especially for people entering deep learning who want to understand **how** frameworks work under the hood.

---

## Roadmap

We’ll build it in stages:

1. **Stage 1 – Scalar autograd**
   - A `Value` class for scalars
   - Build a computation graph
   - Backpropagation (`.backward()`)

2. **Stage 2 – Tensor autograd (NumPy backend)**
   - `Tensor` class with basic ops
   - Autograd over vectors/matrices

3. **Stage 3 – Neural network modules**
   - `Module` base class
   - Layers like `Linear`, `ReLU`, etc.
   - Loss functions

4. **Stage 4 – Optimizers**
   - SGD, Momentum, RMSProp
   - Adam, AdamW
   - Newer optimizers (e.g. Muon)

5. **Stage 5 – Examples**
   - Linear regression
   - Tiny MLP on toy datasets

---

## Getting started

```bash
git clone https://github.com/<your-username>/build_your_own_torch.git
cd build_your_own_torch
python examples/01_scalar_autograd_demo.py
s